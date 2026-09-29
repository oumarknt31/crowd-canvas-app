"""Core algorithm for rendering an image as a Craig-Alan-style crowd mosaic."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


def prepare_sprite(image: Image.Image, remove_baked_shadows: bool = False) -> Optional[Image.Image]:
    """Prepare a transparent motif without modifying its source file.

    Existing motifs contain opaque cool-gray painted shadows. The optional
    cleanup removes that color family, not arbitrary gray pixels. This is a
    heuristic for the existing library; final motifs should be shadow-free.
    """
    image = image.convert("RGBA")
    arr = np.asarray(image).copy()
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]
    visible = alpha > 20
    if not visible.any():
        return None
    # Ignore blank/white extraction artifacts rather than placing invisible people.
    ink = visible & (rgb.min(axis=2) < 225)
    if ink.sum() < max(32, visible.sum() * 0.025):
        return None
    if remove_baked_shadows:
        from scipy import ndimage
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        shadow = (visible & (blue - red >= 13) & (blue - green >= 5)
                  & (green - red >= 3) & (red >= 105)
                  & ((blue - red) <= 65))
        arr[..., 3][shadow] = 0
        # Remove outside white fringes, preserving enclosed white clothing.
        white = (rgb.min(axis=2) > 242)
        outside = ndimage.binary_propagation(arr[..., 3] == 0,
            mask=(arr[..., 3] == 0) | white)
        arr[..., 3][outside & white] = 0
        labels, count = ndimage.label(arr[..., 3] > 20)
        if count:
            sizes = np.bincount(labels.ravel()); sizes[0] = 0
            keep = sizes >= max(12, sizes.max() * 0.002)
            arr[..., 3][~keep[labels]] = 0
    result = Image.fromarray(arr)
    if remove_baked_shadows:
        foreground = (arr[..., 3] > 20) & (rgb.min(axis=2) < 230)
        bbox = Image.fromarray(foreground.astype("uint8") * 255).getbbox()
    else:
        bbox = result.getbbox()
    if bbox is None:
        return None
    return result.crop(bbox)


def load_sprites(sprites_dir: str | Path, remove_baked_shadows: bool = False) -> List[Image.Image]:
    """Load every PNG in ``sprites_dir`` as RGBA, trimmed to its opaque bbox."""
    sprites: List[Image.Image] = []
    for fp in sorted(Path(sprites_dir).glob("*.png")):
        try:
            with Image.open(fp) as source:
                img = prepare_sprite(source, remove_baked_shadows)
        except Exception:
            continue
        if img is not None:
            sprites.append(img)
    return sprites


def _enhance_details(image: Image.Image, strength: float) -> Image.Image:
    """Sharpen edges/details so facial features (eyes, lips, hair, glasses) read better."""
    if strength <= 0:
        return image
    return image.filter(
        ImageFilter.UnsharpMask(radius=2.0, percent=int(strength * 220), threshold=2)
    )


def _density_map(image: Image.Image, gamma: float, blur: float) -> np.ndarray:
    """Return a (H, W) float32 density map in [0, 1]: darker pixels -> denser."""
    g = image.convert("L")
    if blur > 0:
        g = g.filter(ImageFilter.GaussianBlur(radius=blur))
    arr = np.asarray(g, dtype=np.float32) / 255.0
    return np.clip(1.0 - arr, 0.0, 1.0) ** gamma


def detect_face_bbox(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Detect the largest frontal face. Returns ``(x, y, w, h)`` or None.

    Uses OpenCV's Haar cascade (bundled with opencv-python; no model download).
    Returns coordinates in the *input image's* pixel space.
    """
    try:
        import cv2  # type: ignore
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            return None
        arr = np.asarray(image.convert("L"))
        min_side = max(40, min(arr.shape) // 12)
        faces = cascade.detectMultiScale(
            arr, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
        )
    except Exception as exc:
        # Face emphasis is optional; a missing/broken OpenCV must not stop artwork.
        warnings.warn(f"Face detection unavailable: {exc}", RuntimeWarning)
        return None
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return int(fx), int(fy), int(fw), int(fh)


def _face_weight_map(
    shape_hw: Tuple[int, int],
    face_bbox: Optional[Tuple[int, int, int, int]],
    boost: float,
) -> np.ndarray:
    """Return (H, W) weighting: ``boost`` near face center, smoothly fading to 1.0 elsewhere."""
    H, W = shape_hw
    if face_bbox is None or boost <= 1.0:
        return np.ones((H, W), dtype=np.float32)
    fx, fy, fw, fh = face_bbox
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    # Soft elliptical bump; falls off so cheeks/eyes/forehead get strongest boost.
    rx, ry = fw * 0.65, fh * 0.65
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist2 = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2
    return 1.0 + (boost - 1.0) * np.exp(-dist2 * 1.2)


def extract_subject_mask(
    image: Image.Image, *, model: str = "u2net_human_seg"
) -> np.ndarray:
    """Run background removal and return the subject alpha as float32 (H, W) in [0, 1].

    First call downloads the ~170MB ONNX model under ``~/.u2net``; subsequent
    calls reuse it. ``model`` defaults to a human-segmentation network; pass
    ``"u2net"`` for general subjects.
    """
    from rembg import new_session, remove  # lazy: heavy import

    session = new_session(model)
    result = remove(image.convert("RGB"), session=session, post_process_mask=True)
    if isinstance(result, (bytes, bytearray)):
        from io import BytesIO

        result = Image.open(BytesIO(result))
    alpha = result.split()[-1]
    return np.asarray(alpha, dtype=np.float32) / 255.0


def _resize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if mask.shape == target_hw:
        return mask
    img = Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8), mode="L")
    img = img.resize((target_hw[1], target_hw[0]), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _resize_for_work(image: Image.Image, longest_side: int) -> Image.Image:
    iw, ih = image.size
    if max(iw, ih) == longest_side:
        return image
    if iw >= ih:
        return image.resize((longest_side, max(1, round(longest_side * ih / iw))), Image.LANCZOS)
    return image.resize((max(1, round(longest_side * iw / ih)), longest_side), Image.LANCZOS)


def _add_paper_grain(canvas: Image.Image, intensity: float, rng: np.random.Generator) -> Image.Image:
    """Apply a subtle monochrome noise layer to simulate paper texture."""
    if intensity <= 0:
        return canvas
    arr = np.asarray(canvas, dtype=np.float32).copy()
    noise = rng.normal(0.0, intensity * 255.0, arr.shape[:2]).astype(np.float32)
    arr[..., :3] = np.clip(arr[..., :3] + noise[..., None], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode=canvas.mode)


def generate_crowd(
    input_image: Image.Image,
    sprites: Sequence[Image.Image],
    *,
    output_size: int = 2000,
    density_count: int = 4000,
    scatter_count: int = 20,
    sprite_height_pct: float = 0.011,
    scale_jitter: float = 0.12,
    gamma: float = 1.1,
    blur: float = 0.5,
    selectivity: float = 1.0,
    min_density: float = 0.035,
    detail_strength: float = 0.4,
    face_boost: float = 1.0,
    face_bbox=None,
    color_match_strength: float = 0.0,
    subject_only: bool = False,
    subject_mask=None,
    background_color=(245, 240, 230),
    paper_grain: float = 0.003,
    seed=None,
    progress_callback=None,
    **options,
) -> Image.Image:
    """Compatibility API; create a composition, then render at the requested size.

    Use create_layout/render_layout directly to reuse one arrangement at every
    resolution. Background removal is attempted once and may fail gracefully.
    """
    from crowd_layout import create_layout, render_layout
    if subject_only and subject_mask is None:
        try:
            subject_mask = extract_subject_mask(input_image)
        except Exception as exc:
            warnings.warn(f"Subject extraction unavailable: {exc}", RuntimeWarning)
    if face_boost > 1 and face_bbox is None:
        face_bbox = detect_face_bbox(input_image)
    actual_seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy) % (2**32)
    layout_keys = {"mode", "polarity", "spacing", "iterations"}
    layout_options = {k: options.pop(k) for k in list(options) if k in layout_keys}
    layout = create_layout(input_image, sprites, density_count=density_count,
        scatter_count=scatter_count, sprite_height_pct=sprite_height_pct,
        scale_jitter=scale_jitter, gamma=gamma, blur=blur, selectivity=selectivity,
        min_density=min_density, detail_strength=detail_strength,
        face_boost=face_boost, face_bbox=face_bbox, subject_mask=subject_mask,
        seed=actual_seed, progress_callback=progress_callback, **layout_options)
    return render_layout(layout, sprites, output_size=output_size,
        background_color=background_color, paper_grain=paper_grain,
        color_match_strength=color_match_strength, progress_callback=progress_callback,
        **options)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected 6-digit hex color, got {hex_color!r}")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
