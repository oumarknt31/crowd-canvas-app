"""Resolution-independent crowd composition and raster rendering.

Weighted centroidal Voronoi relaxation follows the approach described in
Adrian Secord, Weighted Voronoi Stippling (NPAR 2002). Figures additionally
have rectangular footprints: accepted figures cannot overlap one another.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy.spatial import cKDTree

MAX_OUTPUT_PIXELS = 160_000_000


@dataclass(frozen=True)
class Figure:
    x: float                 # center, in units of the canvas's longest side
    y: float
    height: float
    sprite: int
    mirrored: bool
    rgb: tuple[int, int, int]
    tone: float


@dataclass
class CrowdLayout:
    width: float
    height: float
    figures: list[Figure]
    requested: int
    requested_scatter: int
    seed: int


def _rgb_on_white(image: Image.Image) -> Image.Image:
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    white.alpha_composite(rgba)
    return white.convert("RGB")


def density_map(image: Image.Image, *, gamma: float = 1.1, blur: float = 0.5,
                detail_strength: float = 0.4, min_density: float = 0.035,
                mode: str = "portrait", polarity: str = "dark",
                subject_mask: Optional[np.ndarray] = None) -> tuple[Image.Image, np.ndarray]:
    """Bounded analysis grid; alpha, highlights and masks remain empty space."""
    if mode not in {"portrait", "text", "silhouette"}:
        raise ValueError("Unknown reconstruction mode")
    if polarity not in {"dark", "light"}:
        raise ValueError("Polarity must be dark or light")
    rgb = _rgb_on_white(image)
    if subject_mask is not None:
        matte = Image.fromarray((np.clip(subject_mask, 0, 1) * 255).astype("uint8"))
        matte = matte.resize(rgb.size, Image.Resampling.LANCZOS)
        isolated = Image.new("RGB", rgb.size, "white")
        isolated.paste(rgb, mask=matte)
        rgb = isolated
    rgb.thumbnail((768, 768), Image.Resampling.LANCZOS)
    gray = rgb.convert("L")
    if mode == "portrait":
        gray = ImageOps.autocontrast(gray, cutoff=1)
        if detail_strength:
            gray = gray.filter(ImageFilter.UnsharpMask(
                radius=1.2, percent=int(150 * detail_strength), threshold=3))
        if blur:
            gray = gray.filter(ImageFilter.GaussianBlur(blur))
    light = np.asarray(gray, dtype=np.float64) / 255.0
    density = (1.0 - light) if polarity == "dark" else light
    # Transparent background must never become a light-polarity subject.
    alpha_image = ImageOps.exif_transpose(image).convert("RGBA").getchannel("A")
    alpha_image = alpha_image.resize(rgb.size, Image.Resampling.LANCZOS)
    alpha = np.asarray(alpha_image, dtype=np.float64) / 255.0
    if mode == "silhouette":
        density = (density > max(0.10, min_density)).astype(np.float64)
    density = np.clip(density, 0, 1) ** max(0.1, gamma)
    if mode == "portrait" and detail_strength > 0:
        from scipy import ndimage
        smooth = ndimage.gaussian_filter(light, 0.6)
        gradient = np.hypot(ndimage.sobel(smooth, axis=0), ndimage.sobel(smooth, axis=1))
        nonzero = gradient[gradient > 0.015]
        if len(nonzero):
            normalizer = max(0.15, float(np.percentile(nonzero, 95)))
            edges = np.clip(gradient / normalizer, 0, 1) ** 1.5
            density = np.maximum(density, min(0.8, detail_strength) * edges)
    density[density < min_density] = 0
    density *= alpha
    if subject_mask is not None:
        mask = Image.fromarray((np.clip(subject_mask, 0, 1) * 255).astype("uint8"))
        mask = mask.resize(rgb.size, Image.Resampling.LANCZOS)
        density *= np.asarray(mask, dtype=np.float64) / 255.0
    return rgb, density


def _samples(weights: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(weights.ravel())
    if count == 0 or cumulative[-1] <= 0:
        return np.empty((0, 2), dtype=np.float64)
    indexes = np.searchsorted(cumulative, rng.random(count) * cumulative[-1], side="right")
    ys, xs = np.divmod(indexes, weights.shape[1])
    return np.column_stack((xs + rng.random(count), ys + rng.random(count))) / max(weights.shape)


def _relax(points: np.ndarray, density: np.ndarray, iterations: int,
           progress: Optional[Callable[[float], None]]) -> np.ndarray:
    """Discrete weighted Lloyd iterations, measured in anisotropic figure space."""
    ys, xs = np.nonzero(density > 0)
    if not len(points) or not len(xs):
        return points
    grid = np.column_stack((xs + 0.5, ys + 0.5)) / max(density.shape)
    weights = density[ys, xs] ** 2
    metric = np.array([1.8, 1.0])  # figures are narrower than they are tall
    valid_tree = cKDTree(grid * metric)
    for step in range(iterations):
        tree = cKDTree(points * metric)
        mass = np.zeros(len(points)); moment_x = mass.copy(); moment_y = mass.copy()
        for start in range(0, len(grid), 100_000):
            block = grid[start:start + 100_000]
            w = weights[start:start + 100_000]
            labels = tree.query(block * metric, workers=1)[1]
            mass += np.bincount(labels, weights=w, minlength=len(points))
            moment_x += np.bincount(labels, weights=w * block[:, 0], minlength=len(points))
            moment_y += np.bincount(labels, weights=w * block[:, 1], minlength=len(points))
        good = mass > 1e-15
        points[good, 0] = moment_x[good] / mass[good]
        points[good, 1] = moment_y[good] / mass[good]
        # A centroid can fall in a hole (e.g. the middle of an O). Project it
        # back onto the closest nonzero pixel, preserving counters and gaps.
        nearest = valid_tree.query(points * metric, workers=1)[1]
        points = grid[nearest].copy()
        if progress:
            progress(0.05 + 0.50 * (step + 1) / max(iterations, 1))
    return points


def create_layout(input_image: Image.Image, sprites: Sequence[Image.Image], *,
                  density_count: int = 4000, scatter_count: int = 20,
                  sprite_height_pct: float = 0.011, scale_jitter: float = 0.12,
                  gamma: float = 1.1, blur: float = 0.5, selectivity: float = 1.0,
                  min_density: float = 0.035, detail_strength: float = 0.4,
                  face_boost: float = 1.0, face_bbox=None,
                  subject_mask: Optional[np.ndarray] = None,
                  mode: str = "portrait", polarity: str = "dark",
                  spacing: float = 0.15, iterations: int = 6,
                  seed: int = 42,
                  progress_callback: Optional[Callable[[float], None]] = None) -> CrowdLayout:
    if not sprites:
        raise ValueError("No usable motifs found")
    if not 0 < sprite_height_pct < 0.2:
        raise ValueError("Figure size must be between 0 and 20%")
    if density_count < 0 or scatter_count < 0 or not 0 <= spacing <= 2:
        raise ValueError("Invalid crowd count or spacing")
    rng = np.random.default_rng(seed)
    rgb, density = density_map(input_image, gamma=gamma, blur=blur,
        min_density=min_density, detail_strength=detail_strength, mode=mode,
        polarity=polarity, subject_mask=subject_mask)
    if face_bbox is not None and face_boost > 1:
        from crowdcanvas import _face_weight_map
        iw, ih = input_image.size
        fx, fy, fw, fh = face_bbox
        bbox = (int(fx * rgb.width / iw), int(fy * rgb.height / ih),
                int(fw * rgb.width / iw), int(fh * rgb.height / ih))
        density *= _face_weight_map(density.shape, bbox, face_boost)
    density **= max(0.1, selectivity)
    iw, ih = ImageOps.exif_transpose(input_image).size
    canvas_w, canvas_h = iw / max(iw, ih), ih / max(iw, ih)
    layout = CrowdLayout(canvas_w, canvas_h, [], density_count, scatter_count, int(seed))
    if progress_callback:
        progress_callback(0.02)
    points = _relax(_samples(density, density_count, rng), density, iterations, progress_callback)
    # Greedy footprint filtering and weighted refill enforce actual figure gaps,
    # rather than assuming dot spacing is enough for tall/wide human motifs.
    max_h = sprite_height_pct * (1 + scale_jitter) * (1 + spacing)
    max_w = max_h * max(s.width / s.height for s in sprites)
    cell = max(max_h, max_w, 0.002)
    buckets: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    rgb_arr = np.asarray(rgb)
    sample_side = max(density.shape)

    def accept(point, scatter=False):
        x, y = float(point[0]), float(point[1])
        sy = min(density.shape[0] - 1, max(0, int(y * sample_side)))
        sx = min(density.shape[1] - 1, max(0, int(x * sample_side)))
        if not scatter and density[sy, sx] <= 0:
            return False
        if scatter and density[sy, sx] > 0:
            return False
        idx = int(rng.integers(len(sprites)))
        height = sprite_height_pct * rng.uniform(1 - scale_jitter, 1 + scale_jitter)
        width = height * sprites[idx].width / sprites[idx].height
        if x - width / 2 < 0 or x + width / 2 > canvas_w or y - height / 2 < 0 or y + height / 2 > canvas_h:
            return False
        if not scatter and mode in {"text", "silhouette"}:
            left, right = int((x - width / 2) * sample_side), math.ceil((x + width / 2) * sample_side)
            top, bottom = int((y - height / 2) * sample_side), math.ceil((y + height / 2) * sample_side)
            region = density[top:bottom, left:right]
            if not region.size or np.mean(region > 0) < 0.95:
                return False
        w, h = width * (1 + spacing), height * (1 + spacing)
        bx, by = math.floor(x / cell), math.floor(y / cell)
        for gx in range(bx - 1, bx + 2):
            for gy in range(by - 1, by + 2):
                for ox, oy, ow, oh in buckets.get((gx, gy), []):
                    if abs(x - ox) < (w + ow) / 2 and abs(y - oy) < (h + oh) / 2:
                        return False
        buckets.setdefault((bx, by), []).append((x, y, w, h))
        layout.figures.append(Figure(x, y, height, idx, bool(rng.integers(2)),
            tuple(int(v) for v in rgb_arr[sy, sx]), min(1.0, float(density[sy, sx]))))
        return True

    rejected = []
    for point in points[rng.permutation(len(points))]:
        if not accept(point):
            rejected.append(point)
    # Refill only around rejected locations. Global refill would saturate dark
    # regions, then move their missing people to highlights, flattening the image.
    for point in rejected:
        for _ in range(10):
            nearby = point + rng.normal(0, sprite_height_pct * 0.55, 2)
            if accept(nearby):
                break
    if scatter_count:
        candidates = rng.random((scatter_count * 30, 2)) * [canvas_w, canvas_h]
        scattered = 0
        for point in candidates:
            if scattered >= scatter_count:
                break
            scattered += int(accept(point, scatter=True))
    layout.figures.sort(key=lambda f: f.y)
    if progress_callback:
        progress_callback(0.65)
    return layout


def print_dimensions(width_cm: float, height_cm: float, dpi: int) -> tuple[int, int]:
    if not all(math.isfinite(v) and v > 0 for v in (width_cm, height_cm, dpi)):
        raise ValueError("Print dimensions and DPI must be positive")
    size = (max(1, round(width_cm / 2.54 * dpi)), max(1, round(height_cm / 2.54 * dpi)))
    if size[0] * size[1] > MAX_OUTPUT_PIXELS:
        raise ValueError("This export exceeds 160 megapixels. Reduce print dimensions or DPI.")
    return size


def _cast_shadow(sprite: Image.Image, angle: float, length: float,
                 opacity: float, softness: float) -> tuple[Image.Image, tuple[int, int]]:
    """Project an alpha silhouette onto a ground plane, anchored at the feet.

    This is a 2D projection, not a recovered 3D body or physically exact ray trace.
    angle is the SUN azimuth (0 right, 90 bottom); shadows point the other way.
    """
    alpha = sprite.getchannel("A")
    arr = np.asarray(alpha)
    bottom = arr[max(0, sprite.height - max(2, sprite.height // 15)):, :].sum(axis=0)
    foot_x = float(np.dot(np.arange(sprite.width), bottom) / max(float(bottom.sum()), 1))
    radians = math.radians(angle + 180)
    direction = np.array([math.cos(radians), math.sin(radians)])
    across = np.array([-direction[1], direction[0]]) * 0.65
    extent = max(1.0, sprite.height * length)
    matrix = np.column_stack((across, -direction * extent / max(sprite.height - 1, 1)))
    offset = direction * extent - across * foot_x
    corners = np.array([[0, 0], [sprite.width, 0], [0, sprite.height], [sprite.width, sprite.height]])
    mapped = corners @ matrix.T + offset
    radius = max(0.0, softness * sprite.height)
    pad = math.ceil(radius * 3 + 2)
    low = np.floor(mapped.min(axis=0)).astype(int) - pad
    high = np.ceil(mapped.max(axis=0)).astype(int) + pad
    inverse = np.linalg.inv(matrix)
    translation = inverse @ (low - offset)
    coeffs = (inverse[0, 0], inverse[0, 1], translation[0],
              inverse[1, 0], inverse[1, 1], translation[1])
    mask = alpha.transform(tuple(high - low), Image.Transform.AFFINE, coeffs,
                           resample=Image.Resampling.BICUBIC)
    if radius:
        mask = mask.filter(ImageFilter.GaussianBlur(radius))
    mask = mask.point(lambda v: round(v * opacity))
    result = Image.new("RGBA", mask.size, (44, 39, 49, 0))
    result.putalpha(mask)
    return result, (round(low[0] + foot_x), int(low[1] + (sprite.height - 1)))


def render_layout(layout: CrowdLayout, sprites: Sequence[Image.Image], *,
                  output_size: int = 2000, dimensions: Optional[tuple[int, int]] = None,
                  background_color=(245, 240, 230), paper_grain: float = 0.003,
                  color_match_strength: float = 0.0, figure_palette: str = "natural", shadow_enabled: bool = True,
                  sun_angle: float = 225, sun_elevation: float = 45,
                  shadow_opacity: float = 0.23, shadow_softness: float = 0.035,
                  progress_callback: Optional[Callable[[float], None]] = None) -> Image.Image:
    if dimensions is None:
        dimensions = (max(1, round(layout.width * output_size)), max(1, round(layout.height * output_size)))
    width, height = dimensions
    if min(width, height) < 1 or width * height > MAX_OUTPUT_PIXELS:
        raise ValueError("Export must contain between 1 and 160 million pixels")
    if not 1 <= sun_elevation <= 89:
        raise ValueError("Sun elevation must be between 1 and 89 degrees")
    if not 0 <= shadow_opacity <= 1 or not 0 <= color_match_strength <= 1:
        raise ValueError("Opacity and color matching must be between 0 and 1")
    if figure_palette not in {"natural", "ink"}:
        raise ValueError("Unknown figure palette")
    canvas = Image.new("RGB", dimensions, background_color)
    # Uniform contain transform: exact physical page size without distortion.
    scale = min(width / layout.width, height / layout.height)
    offset_x = (width - layout.width * scale) / 2
    offset_y = (height - layout.height * scale) / 2
    if paper_grain:
        rng = np.random.default_rng(layout.seed)
        for y in range(0, height, 128):
            rows = min(128, height - y)
            arr = np.broadcast_to(np.array(background_color, dtype=np.float32), (rows, width, 3)).copy()
            arr += rng.normal(0, paper_grain * 255, (rows, width, 1)).astype(np.float32)
            canvas.paste(Image.fromarray(np.clip(arr, 0, 255).astype("uint8")), (0, y))
    prepared = []
    cache = {}
    max_upscale = 0.0
    length = 0.75 / math.tan(math.radians(sun_elevation))
    for f in layout.figures:
        source = sprites[f.sprite]
        h = max(2, round(f.height * scale)); w = max(1, round(h * source.width / source.height))
        key = (f.sprite, f.mirrored, w, h)
        if key not in cache:
            resized = source.resize((w, h), Image.Resampling.LANCZOS)
            if f.mirrored:
                resized = ImageOps.mirror(resized)
            if figure_palette == "ink" and f.sprite % 5 != 0:
                pixels = np.asarray(resized).copy()
                gray = pixels[..., :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])
                pixels[..., :3] = np.clip(np.array([20, 24, 28]) + gray[..., None] * 0.19, 0, 255).astype("uint8")
                resized = Image.fromarray(pixels)
            shadow = _cast_shadow(resized, sun_angle, length, shadow_opacity, shadow_softness) if shadow_enabled else None
            cache[key] = resized, shadow
        motif, shadow = cache[key]
        px = round(offset_x + f.x * scale - w / 2)
        py = round(offset_y + f.y * scale - h / 2)
        prepared.append((f, motif, shadow, px, py))
        max_upscale = max(max_upscale, h / source.height)
    # One ground-shadow pass, then figures in depth order. A shadow cannot paint
    # over another person's body; each shadow is attached to its own feet.
    for _, _, shadow, px, py in prepared:
        if shadow:
            shade, (dx, dy) = shadow
            canvas.paste(shade, (px + dx, py + dy), shade)
    for index, (f, motif, _, px, py) in enumerate(prepared):
        if color_match_strength:
            arr = np.asarray(motif).copy()
            rgb = arr[..., :3].astype(np.float32)
            target = np.array(f.rgb, dtype=np.float32)
            # Keep dark outlines and skin; tint chromatic clothing without
            # flattening every pixel to the sampled input color.
            chroma = rgb.max(axis=2) - rgb.min(axis=2)
            skin = (rgb[..., 0] > rgb[..., 1] * 1.12) & (rgb[..., 1] > rgb[..., 2] * 1.12)
            amount = np.where((chroma > 25) & ~skin, color_match_strength, 0)[..., None]
            luminance = rgb.mean(axis=2, keepdims=True) / 255
            tint = target[None, None, :] * (0.25 + 0.75 * luminance)
            arr[..., :3] = np.clip(rgb * (1 - amount) + tint * amount, 0, 255).astype("uint8")
            motif = Image.fromarray(arr)
        canvas.paste(motif, (px, py), motif)
        if progress_callback and index % max(1, len(prepared) // 30) == 0:
            progress_callback(0.65 + 0.35 * index / max(1, len(prepared)))
    canvas.info.update(figures_placed=len(layout.figures), requested_figures=layout.requested + layout.requested_scatter,
                       seed=layout.seed, max_sprite_upscale=max_upscale)
    if progress_callback:
        progress_callback(1.0)
    return canvas
