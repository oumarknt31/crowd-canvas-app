"""Core algorithm for rendering an image as a Craig-Alan-style crowd mosaic."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


def load_sprites(sprites_dir: str | Path) -> List[Image.Image]:
    """Load every PNG in ``sprites_dir`` as RGBA, trimmed to its opaque bbox."""
    sprites: List[Image.Image] = []
    for fp in sorted(Path(sprites_dir).glob("*.png")):
        try:
            img = Image.open(fp).convert("RGBA")
        except Exception:
            continue
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
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
    except ImportError:
        return None
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
    density_count: int = 2800,
    scatter_count: int = 0,
    sprite_height_pct: float = 0.015,
    scale_jitter: float = 0.18,
    gamma: float = 2.0,
    blur: float = 1.0,
    selectivity: float = 3.0,
    min_density: float = 0.10,
    detail_strength: float = 0.6,
    face_boost: float = 2.4,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
    subject_only: bool = True,
    subject_mask: Optional[np.ndarray] = None,
    background_color: Tuple[int, int, int] = (245, 240, 230),
    paper_grain: float = 0.015,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Image.Image:
    """Render ``input_image`` as a crowd of small figures sampled from ``sprites``.

    The number of figures placed in any region is proportional to that region's
    darkness, so dark facial features (eyes, hair, mouth) form dense clusters
    while light areas thin out — the visual signature of Craig Alan's Populus.
    """
    if not sprites:
        raise ValueError("sprites list is empty")

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    work_img = _resize_for_work(input_image.convert("RGB"), output_size)
    enhanced = _enhance_details(work_img, detail_strength)
    density = _density_map(enhanced, gamma=gamma, blur=blur)
    H, W = density.shape

    # Gate density by subject mask so figures land on the subject, not background.
    if subject_only and subject_mask is None:
        subject_mask = extract_subject_mask(work_img)
    if subject_mask is not None and float(subject_mask.max()) >= 0.05:
        mask = _resize_mask(subject_mask, (H, W))
        density = density * mask

    # Boost density on the face so eyes/nose/mouth/glasses get more figures than the body.
    if face_boost > 1.0:
        if face_bbox is None:
            face_bbox = detect_face_bbox(work_img)
        else:
            # bbox supplied in input-image coords — scale to working resolution.
            iw, ih = input_image.size
            sx, sy = W / iw, H / ih
            fx, fy, fw, fh = face_bbox
            face_bbox = (int(fx * sx), int(fy * sy), int(fw * sx), int(fh * sy))
        density = density * _face_weight_map((H, W), face_bbox, face_boost)

    base_h = max(20, int(H * sprite_height_pct))
    base_sprites: List[Image.Image] = []
    for s in sprites:
        ratio = base_h / s.height
        new_size = (max(1, int(round(s.width * ratio))), base_h)
        base_sprites.append(s.resize(new_size, Image.LANCZOS))

    canvas = Image.new("RGBA", (W, H), background_color + (255,))
    canvas = _add_paper_grain(canvas, paper_grain, np_rng)

    placements: List[Tuple[int, int, int, float]] = []  # (x, y, sprite_idx, scale)

    dmax = float(density.max()) or 1.0
    attempts = 0
    max_attempts = max(density_count * 200, 50_000)
    while len(placements) < density_count and attempts < max_attempts:
        attempts += 1
        x = py_rng.randrange(W)
        y = py_rng.randrange(H)
        d_norm = density[y, x] / dmax
        if d_norm < min_density:
            continue
        if py_rng.random() < d_norm ** selectivity:
            scale = max(0.5, 1.0 + py_rng.uniform(-scale_jitter, scale_jitter))
            placements.append((x, y, py_rng.randrange(len(base_sprites)), scale))

    for _ in range(scatter_count):
        x = py_rng.randrange(W)
        y = py_rng.randrange(H)
        scale = max(0.5, 1.0 + py_rng.uniform(-scale_jitter, scale_jitter))
        placements.append((x, y, py_rng.randrange(len(base_sprites)), scale))

    placements.sort(key=lambda p: p[1])

    total = max(1, len(placements))
    report_every = max(1, total // 50)
    for i, (x, y, idx, scale) in enumerate(placements):
        sp = base_sprites[idx]
        if scale != 1.0:
            sp = sp.resize(
                (max(1, int(sp.width * scale)), max(1, int(sp.height * scale))),
                Image.BILINEAR,
            )
        px = x - sp.width // 2
        py = y - sp.height // 2
        canvas.alpha_composite(sp, (px, py))
        if progress_callback and (i % report_every == 0):
            progress_callback(i / total)

    if progress_callback:
        progress_callback(1.0)

    return canvas.convert("RGB")


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected 6-digit hex color, got {hex_color!r}")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
