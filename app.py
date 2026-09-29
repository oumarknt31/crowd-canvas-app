"""CrowdCanvas — Streamlit app: turn an image into a Craig-Alan-style crowd portrait.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import io
import hashlib
import json
import secrets
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, ImageCms

# Compatibility shim for streamlit-drawable-canvas: it calls a private helper
# `streamlit.elements.image.image_to_url` that was removed in Streamlit ≥ 1.39.
# Provide a drop-in replacement that returns a base64 data URL.
try:
    import streamlit.elements.image as _st_image  # type: ignore[attr-defined]
    if not hasattr(_st_image, "image_to_url"):
        from io import BytesIO as _BytesIO

        def _image_to_url(image, width, clamp, channels, output_format, image_id):
            buf = _BytesIO()
            fmt = output_format.upper() if output_format and output_format != "auto" else "PNG"
            image.save(buf, format=fmt)
            return (
                f"data:image/{fmt.lower()};base64,"
                + base64.b64encode(buf.getvalue()).decode("ascii")
            )

        _st_image.image_to_url = _image_to_url  # type: ignore[attr-defined]
except Exception:
    pass

from streamlit_drawable_canvas import st_canvas  # noqa: E402

from crowdcanvas import (
    detect_face_bbox,
    extract_subject_mask,
    hex_to_rgb,
    load_sprites,
)

from crowd_layout import create_layout, render_layout, print_dimensions

HERE = Path(__file__).parent
SPRITES_DIR = HERE / "Artwork people"

st.set_page_config(page_title="CrowdCanvas", page_icon="🧑‍🎨", layout="wide")
st.title("CrowdCanvas")
st.caption(
    "Upload an image — the app rebuilds it as a crowd of tiny figures, in the "
    "spirit of Craig Alan's *Populus* portraits. Works best with high-contrast "
    "subjects (faces, silhouettes, logos)."
)


# Exact physical frames. Exports contain the artwork without stretching it.
PRINT_PRESETS = [
    ("A4", 21.0, 29.7, 300),
    ("A3", 29.7, 42.0, 300),
    ("30 × 40 cm", 30.0, 40.0, 300),
    ("50 × 70 cm", 50.0, 70.0, 240),
    ("60 × 90 cm", 60.0, 90.0, 200),
    ("70 × 100 cm", 70.0, 100.0, 200),
]

# ------- Pan/zoom HTML viewer (self-contained, no JS deps) -------------------
ZOOM_HTML_TEMPLATE = """
<style>
  .cc-wrap { position: relative; width: 100%; height: __HEIGHT__px;
             overflow: hidden; cursor: grab; background: #f5f0e6;
             border: 1px solid #ddd; border-radius: 4px;
             font-family: system-ui, -apple-system, sans-serif; }
  .cc-wrap.cc-grabbing { cursor: grabbing; }
  .cc-wrap img { position: absolute; top: 0; left: 0;
                 transform-origin: 0 0;
                 user-select: none; -webkit-user-drag: none;
                 pointer-events: none;
                 image-rendering: -webkit-optimize-contrast; }
  .cc-ctrls { position: absolute; top: 8px; right: 8px; display: flex;
              gap: 4px; z-index: 10;
              background: rgba(255,255,255,0.92); padding: 4px;
              border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .cc-ctrls button { padding: 4px 10px; font-size: 13px; cursor: pointer;
                     border: 1px solid #aaa; background: #fff;
                     border-radius: 3px; min-width: 36px; }
  .cc-ctrls button:hover { background: #f0f0f0; }
  .cc-info { position: absolute; bottom: 8px; left: 8px;
             background: rgba(255,255,255,0.92); padding: 4px 8px;
             font-size: 12px; border-radius: 3px;
             color: #333; }
</style>
<div id="cc-wrap" class="cc-wrap">
  <img id="cc-img" src="data:image/jpeg;base64,__IMAGE__" alt="canvas"/>
  <div class="cc-ctrls">
    <button id="cc-in" title="Zoom in">＋</button>
    <button id="cc-out" title="Zoom out">－</button>
    <button id="cc-fit" title="Fit to view">Fit</button>
    <button id="cc-100" title="100% scale">1:1</button>
  </div>
  <div class="cc-info">
    Zoom <span id="cc-z">100%</span> · drag to pan · scroll to zoom
  </div>
</div>
<script>
(function () {
  var img = document.getElementById('cc-img');
  var wrap = document.getElementById('cc-wrap');
  var lbl = document.getElementById('cc-z');
  var scale = 1, tx = 0, ty = 0;

  function apply() {
    img.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
    lbl.textContent = Math.round(scale * 100) + '%';
  }
  function fit() {
    var w = wrap.clientWidth, h = wrap.clientHeight;
    var iw = img.naturalWidth, ih = img.naturalHeight;
    if (!iw || !ih) return;
    scale = Math.min(w / iw, h / ih, 1);
    tx = (w - iw * scale) / 2;
    ty = (h - ih * scale) / 2;
    apply();
  }
  function zoomBy(factor, cx, cy) {
    if (cx === undefined) { cx = wrap.clientWidth / 2; cy = wrap.clientHeight / 2; }
    var ix = (cx - tx) / scale, iy = (cy - ty) / scale;
    scale = Math.max(0.05, Math.min(20, scale * factor));
    tx = cx - ix * scale;
    ty = cy - iy * scale;
    apply();
  }
  function zoom100() {
    var cx = wrap.clientWidth / 2, cy = wrap.clientHeight / 2;
    zoomBy(1 / scale, cx, cy);
  }

  document.getElementById('cc-in').addEventListener('click', function () { zoomBy(1.4); });
  document.getElementById('cc-out').addEventListener('click', function () { zoomBy(0.71); });
  document.getElementById('cc-fit').addEventListener('click', fit);
  document.getElementById('cc-100').addEventListener('click', zoom100);

  if (img.complete && img.naturalWidth) fit();
  img.addEventListener('load', fit);

  wrap.addEventListener('wheel', function (e) {
    e.preventDefault();
    var r = wrap.getBoundingClientRect();
    zoomBy(e.deltaY < 0 ? 1.15 : 0.87, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });

  var drag = false, sx = 0, sy = 0;
  wrap.addEventListener('mousedown', function (e) {
    drag = true; sx = e.clientX - tx; sy = e.clientY - ty;
    wrap.classList.add('cc-grabbing');
  });
  window.addEventListener('mousemove', function (e) {
    if (!drag) return;
    tx = e.clientX - sx; ty = e.clientY - sy; apply();
  });
  window.addEventListener('mouseup', function () {
    drag = false; wrap.classList.remove('cc-grabbing');
  });

  // Touch / pinch
  var pinchDist = 0;
  wrap.addEventListener('touchstart', function (e) {
    if (e.touches.length === 1) {
      drag = true;
      sx = e.touches[0].clientX - tx; sy = e.touches[0].clientY - ty;
    } else if (e.touches.length === 2) {
      var dx = e.touches[0].clientX - e.touches[1].clientX;
      var dy = e.touches[0].clientY - e.touches[1].clientY;
      pinchDist = Math.sqrt(dx * dx + dy * dy);
    }
  });
  wrap.addEventListener('touchmove', function (e) {
    e.preventDefault();
    if (e.touches.length === 1 && drag) {
      tx = e.touches[0].clientX - sx; ty = e.touches[0].clientY - sy;
      apply();
    } else if (e.touches.length === 2) {
      var dx = e.touches[0].clientX - e.touches[1].clientX;
      var dy = e.touches[0].clientY - e.touches[1].clientY;
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (pinchDist > 0) {
        var r = wrap.getBoundingClientRect();
        var cx = (e.touches[0].clientX + e.touches[1].clientX) / 2 - r.left;
        var cy = (e.touches[0].clientY + e.touches[1].clientY) / 2 - r.top;
        zoomBy(dist / pinchDist, cx, cy);
      }
      pinchDist = dist;
    }
  }, { passive: false });
  wrap.addEventListener('touchend', function (e) {
    if (e.touches.length === 0) drag = false;
    if (e.touches.length < 2) pinchDist = 0;
  });
})();
</script>
"""


def _image_to_jpeg_b64(img: Image.Image, max_dim: int = 2400, quality: int = 88) -> str:
    """Encode an image as base64 JPEG, downsampled if larger than ``max_dim``.

    Keeps embedded payload small enough for browser viewer; full-res download
    is served separately via st.download_button.
    """
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize(
            (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_zoom_viewer(img: Image.Image, height_px: int = 720) -> None:
    """Embed an interactive pan/zoom HTML viewer for the given image."""
    b64 = _image_to_jpeg_b64(img)
    html = ZOOM_HTML_TEMPLATE.replace("__IMAGE__", b64).replace("__HEIGHT__", str(height_px))
    if hasattr(st, "iframe"):
        st.iframe(html, height=height_px + 12)
    else:
        components.html(html, height=height_px + 12, scrolling=False)


# ------- Manual mask editor helpers ------------------------------------------
ADD_BRUSH_HEX = "#ff3030"   # red = add to mask
REMOVE_BRUSH_HEX = "#3060ff"  # blue = remove from mask


def _canvas_dims(img: Image.Image, max_side: int = 700) -> tuple:
    iw, ih = img.size
    if iw >= ih:
        cw = min(iw, max_side)
        ch = max(1, int(round(cw * ih / iw)))
    else:
        ch = min(ih, max_side)
        cw = max(1, int(round(ch * iw / ih)))
    return cw, ch


def _apply_strokes(
    auto_mask: Optional[np.ndarray],
    canvas_data: np.ndarray,
    target_hw: tuple,
) -> np.ndarray:
    """Combine the canvas's painted strokes with the auto mask.

    Red strokes set mask = 1; blue strokes set mask = 0; untouched pixels keep
    the auto-mask value (or 0 if no auto mask is supplied).
    """
    target_h, target_w = target_hw
    rgba = np.asarray(canvas_data)
    alpha = rgba[..., 3]
    r = rgba[..., 0].astype(np.int16)
    g = rgba[..., 1].astype(np.int16)
    b = rgba[..., 2].astype(np.int16)
    painted = alpha > 30

    add = (painted & (r > 150) & (b < 120)).astype(np.uint8) * 255
    remove = (painted & (b > 150) & (r < 120)).astype(np.uint8) * 255

    add_img = Image.fromarray(add, mode="L").resize((target_w, target_h), Image.LANCZOS)
    rem_img = Image.fromarray(remove, mode="L").resize((target_w, target_h), Image.LANCZOS)
    add_arr = np.asarray(add_img, dtype=np.float32) / 255.0
    rem_arr = np.asarray(rem_img, dtype=np.float32) / 255.0

    if auto_mask is None:
        base = np.zeros((target_h, target_w), dtype=np.float32)
    elif auto_mask.shape != (target_h, target_w):
        base_img = Image.fromarray(
            (auto_mask * 255).clip(0, 255).astype(np.uint8), mode="L"
        ).resize((target_w, target_h), Image.LANCZOS)
        base = np.asarray(base_img, dtype=np.float32) / 255.0
    else:
        base = auto_mask.copy()

    new_mask = np.maximum(base, add_arr)         # add → force toward 1
    new_mask = new_mask * (1.0 - rem_arr)         # remove → multiply toward 0
    return np.clip(new_mask, 0.0, 1.0)


# ------- Studio controls and reusable composition ---------------------------
_light_compass = components.declare_component("crowd_light_compass", path=str(HERE / "components" / "light_compass"))


@st.cache_resource(show_spinner="Preparing people motifs…")
def _load_sprites_cached(path: str, clean: bool):
    return load_sprites(path, remove_baked_shadows=clean)


def _apply_mode_defaults():
    portrait = st.session_state["input_type"] == "Portrait"
    st.session_state["people_requested"] = 2200 if portrait else 2000
    st.session_state["person_height_percent"] = 0.7 if portrait else 1.0
    st.session_state["tone_contrast"] = 2.8 if portrait else 1.0
    st.session_state["people_palette"] = "Ink with color accents" if portrait else "Original motif colors"


def _shuffle_seed():
    st.session_state["random_seed"] = secrets.randbelow(2**31)
    st.session_state["shuffle_requested"] = True


with st.sidebar:
    for key, value in {"people_requested": 2200, "person_height_percent": 0.7,
                       "tone_contrast": 2.8, "random_seed": 42,
                       "people_palette": "Ink with color accents"}.items():
        st.session_state.setdefault(key, value)
    st.header("Composition")
    mode_label = st.selectbox("Input type", ["Portrait", "Text / logo", "Object / silhouette"], key="input_type", on_change=_apply_mode_defaults)
    mode = {"Portrait": "portrait", "Text / logo": "text", "Object / silhouette": "silhouette"}[mode_label]
    polarity_label = st.selectbox("Reconstruct", ["Dark areas on light background", "Light areas on dark background"])
    polarity = "dark" if polarity_label.startswith("Dark") else "light"
    subject_only = st.toggle("Remove input background", value=True, disabled=mode == "text",
        help="Use for a person or object against a busy background. Letters and logos do not need an AI subject detector.")
    subject_only = subject_only and mode != "text"
    subject_model = st.selectbox("Subject detector", ["u2net_human_seg", "u2net", "isnet-general-use"],
        index=0 if mode == "portrait" else 2, disabled=not subject_only)
    density_count = st.slider("People requested", 200, 20000, value=None, step=200, key="people_requested")
    sprite_height_pct = st.slider("Person height (% of long edge)", 0.3, 3.0, value=None, step=0.1, key="person_height_percent") / 100
    spacing = st.slider("Space between people", 0.0, 1.0, 0.15, 0.05,
        help="Minimum extra gap as a fraction of each person's width and height. If people cannot fit, fewer will be placed.")
    scatter_count = st.slider("People outside the image", 0, 300, 20, 5)
    seed_input = st.number_input("Arrangement seed", 0, 2**31 - 1, value=None, key="random_seed")
    st.button("Shuffle and generate", on_click=_shuffle_seed, width="stretch")
    with st.expander("Tone and detail"):
        gamma = st.slider("Tone contrast", 0.5, 4.0, value=None, step=0.1, key="tone_contrast")
        detail_strength = st.slider("Feature enhancement", 0.0, 1.5, 0.4, 0.1)
        blur = st.slider("Input smoothing", 0.0, 3.0, 0.5, 0.1)
        min_density = st.slider("Ignore faint tones", 0.0, 0.3, 0.035, 0.005)
        face_boost = st.slider("Face emphasis", 1.0, 2.5, 1.0, 0.1, disabled=mode != "portrait")
        scale_jitter = st.slider("Person size variation", 0.0, 0.3, 0.12, 0.02)
        iterations = st.slider("Spacing refinement passes", 0, 12, 6)
    st.header("Light and finish")
    clean_sprites = st.toggle("Separate people from painted shadows", value=True,
        help="Automatic cleanup of the existing motifs. Inspect pale clothing in the motif preview; clean transparent motifs are best for final prints.")
    shadow_enabled = st.toggle("Cast shadows", value=True, disabled=not clean_sprites)
    shadow_enabled = shadow_enabled and clean_sprites
    compass_value = _light_compass(angle=225, default=225, key="sun_compass")
    try:
        sun_angle = float(compass_value) % 360
    except (TypeError, ValueError):
        sun_angle = 225.0
    if not np.isfinite(sun_angle):
        sun_angle = 225.0
    sun_elevation = st.slider("Sun height", 15, 80, 45, help="A lower sun makes longer shadows.")
    shadow_opacity = st.slider("Shadow strength", 0.0, 0.5, 0.23, 0.01)
    shadow_softness = st.slider("Shadow softness", 0.0, 0.12, 0.035, 0.005)
    bg_hex = st.color_picker("Canvas color", "#F5F0E6")
    palette_label = st.selectbox("People palette", ["Ink with color accents", "Original motif colors"], index=None, key="people_palette")
    figure_palette = "ink" if palette_label.startswith("Ink") else "natural"
    color_match_strength = st.slider("Match clothing to input colors", 0.0, 0.9, 0.0, 0.05)
    paper_grain = st.slider("Paper texture", 0.0, 0.02, 0.003, 0.001)
    output_size = st.select_slider("Preview long edge (pixels)", [1000, 1500, 2000, 2500, 3000], value=1500)

sprites = _load_sprites_cached(str(SPRITES_DIR), clean_sprites)
if not sprites:
    st.error("No usable PNG motifs found in Artwork people.")
    st.stop()
st.sidebar.caption(f"{len(sprites)} usable motifs · composition seed {seed_input}")

with st.expander("Inspect the motif library"):
    st.caption("The source files are preserved. Automatic cleanup removes the old cool-gray shadows; check that clothing and accessories remain intact. Replace motifs with shadow-free transparent PNGs for the best print results.")
    cols = st.columns(8)
    for index, sprite in enumerate(sprites):
        with cols[index % 8]:
            st.image(sprite, caption=f"Motif {index + 1}", width="stretch")

uploaded = st.file_uploader("Input image", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
if uploaded is None:
    st.info("Upload a portrait, lettering or an object. Start with a clean background and clearly visible details.")
    st.stop()
raw_bytes = uploaded.getvalue()
input_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes)))
upload_key = hashlib.sha256(raw_bytes).hexdigest()
if st.session_state.get("upload_key") != upload_key:
    st.session_state["upload_key"] = upload_key
    for key in ("layout", "output", "output_meta", "manual_mask", "render_options", "export_buffers"):
        st.session_state.pop(key, None)
    st.session_state["mask_canvas_key"] = 0


@st.cache_data(show_spinner="Extracting subject (first run downloads a model)…")
def _cached_subject_mask(image_bytes: bytes, model: str):
    return extract_subject_mask(ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))), model=model)


@st.cache_data(show_spinner=False)
def _cached_face_bbox(image_bytes: bytes):
    return detect_face_bbox(ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))))


subject_mask = None
if subject_only:
    try:
        subject_mask = _cached_subject_mask(raw_bytes, subject_model)
        if float(subject_mask.max()) < 0.05:
            st.warning("Subject extraction found no subject. Whole-image reconstruction is available.")
            subject_mask = None
    except Exception as exc:
        st.warning(f"Background removal unavailable: {exc}. Whole-image reconstruction is available.")
face_bbox = _cached_face_bbox(raw_bytes) if mode == "portrait" and face_boost > 1 else None
manual_mask = st.session_state.get("manual_mask")
effective_subject_mask = manual_mask if manual_mask is not None else subject_mask

layout_options = dict(density_count=density_count, scatter_count=scatter_count,
    sprite_height_pct=sprite_height_pct, spacing=spacing, iterations=iterations,
    gamma=gamma, blur=blur, detail_strength=detail_strength, min_density=min_density,
    face_boost=face_boost if mode == "portrait" else 1.0, face_bbox=face_bbox,
    scale_jitter=scale_jitter, mode=mode, polarity=polarity, seed=int(seed_input))
render_options = dict(background_color=hex_to_rgb(bg_hex), paper_grain=paper_grain,
    color_match_strength=color_match_strength, figure_palette=figure_palette, shadow_enabled=shadow_enabled,
    sun_angle=sun_angle, sun_elevation=sun_elevation, shadow_opacity=shadow_opacity,
    shadow_softness=shadow_softness)


@st.cache_data(show_spinner=False, max_entries=6)
def _composition_cached(image_bytes: bytes, options_json: str, mask, clean: bool, _sprites):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    return create_layout(image, _sprites, subject_mask=mask, **json.loads(options_json))


def _do_render(target_size: int, label: str, *, dimensions=None, dpi=None,
               rebuild=False, refresh_style=False):
    progress = st.progress(0.0, text="Preparing crowd artwork…")
    try:
        if rebuild or "layout" not in st.session_state:
            layout = _composition_cached(raw_bytes, json.dumps(layout_options, sort_keys=True),
                effective_subject_mask, clean_sprites, sprites)
            st.session_state["layout"] = layout
            st.session_state["layout_clean"] = clean_sprites
            st.session_state["render_options"] = render_options.copy()
        layout = st.session_state["layout"]
        if refresh_style:
            st.session_state["render_options"] = render_options.copy()
        style = st.session_state["render_options"].copy()
        # A saved composition always uses the library it was built with.
        saved_sprites = _load_sprites_cached(str(SPRITES_DIR), st.session_state["layout_clean"])
        style["shadow_enabled"] = style["shadow_enabled"] and st.session_state["layout_clean"]
        image = render_layout(layout, saved_sprites, output_size=target_size,
            dimensions=dimensions, progress_callback=lambda p: progress.progress(p, text="Rendering artwork…"), **style)
    except (ValueError, MemoryError) as exc:
        st.error(str(exc))
        return False
    finally:
        progress.empty()
    st.session_state["output"] = image
    st.session_state["output_meta"] = dict(label=label, size_px=image.size, dpi=dpi,
        seed=layout.seed, figures=len(layout.figures), requested=layout.requested + layout.requested_scatter)
    st.session_state["export_buffers"] = {}
    return True


# Lighting/finish changes reuse the composition and refresh only a preview.
if ("layout" in st.session_state and st.session_state.get("layout_clean") == clean_sprites
        and st.session_state.get("render_options") != render_options):
    _do_render(output_size, "preview", refresh_style=True)

col_in, col_out = st.columns([1, 1])
with col_in:
    st.subheader("Input")
    st.image(input_img, width="stretch")
    if face_bbox:
        st.caption("Face detected; face emphasis is enabled.")
    if max(input_img.size) < 800:
        st.caption("This input is small. A sharper original can improve facial features and fine lettering.")
    with st.expander("Subject mask and manual touch-up"):
        if effective_subject_mask is not None:
            st.image((effective_subject_mask * 255).clip(0, 255).astype("uint8"),
                caption="White areas receive people. Black areas remain empty.", width="stretch")
        edit_on = st.toggle("Edit mask manually", value=False, key="edit_mask_toggle")
        if edit_on:
            brush_mode = st.radio("Brush", ["Add to mask (red)", "Remove from mask (blue)"], horizontal=True)
            brush_size = st.slider("Brush size", 5, 80, 28)
            cw, ch = _canvas_dims(input_img)
            canvas_result = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=brush_size,
                stroke_color=ADD_BRUSH_HEX if brush_mode.startswith("Add") else REMOVE_BRUSH_HEX,
                background_image=input_img.convert("RGB"), update_streamlit=True,
                width=cw, height=ch, drawing_mode="freedraw",
                key=f"mask_canvas_{st.session_state.get('mask_canvas_key', 0)}")
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Apply mask edits"):
                    if canvas_result.image_data is not None and np.asarray(canvas_result.image_data)[..., 3].max() > 0:
                        target = subject_mask.shape if subject_mask is not None else (input_img.height, input_img.width)
                        base = manual_mask if manual_mask is not None else subject_mask
                        st.session_state["manual_mask"] = _apply_strokes(base, canvas_result.image_data, target)
                        st.session_state["mask_canvas_key"] += 1
                        st.rerun()
            with b2:
                if st.button("Clear strokes"):
                    st.session_state["mask_canvas_key"] += 1
                    st.rerun()
            with b3:
                if st.button("Reset mask"):
                    st.session_state.pop("manual_mask", None)
                    st.session_state["mask_canvas_key"] += 1
                    st.rerun()
        st.caption("Generate a new preview after changing the mask or composition controls.")

with col_out:
    st.subheader("Crowd artwork")
    generate = st.button("Generate preview", type="primary", width="stretch")
    if generate or st.session_state.pop("shuffle_requested", False):
        _do_render(output_size, "preview", rebuild=True)
    if "output" in st.session_state:
        image = st.session_state["output"]
        meta = st.session_state["output_meta"]
        st.image(image, width="stretch")
        w, h = meta["size_px"]
        st.caption(f"{meta['label']} · {w:,} × {h:,} px · {meta['figures']:,} people · seed {meta['seed']}")
        if meta["figures"] < meta["requested"]:
            st.caption("Fewer people fit while keeping the chosen gaps. Reduce person size or spacing to fit more.")
        if image.info.get("max_sprite_upscale", 0) > 1:
            st.warning("Some motifs are enlarged beyond their original pixels. Smaller figures or higher-resolution motifs will print more sharply.")
        file_format = st.selectbox("Download format", ["PNG", "TIFF"])
        buffers = st.session_state["export_buffers"]
        if file_format not in buffers:
            buf = io.BytesIO()
            options = {"icc_profile": ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()}
            if meta["dpi"]:
                options["dpi"] = (meta["dpi"], meta["dpi"])
            if file_format == "TIFF":
                options["compression"] = "tiff_lzw"
            image.save(buf, format=file_format, **options)
            buffers[file_format] = buf.getvalue()
        extension = "png" if file_format == "PNG" else "tif"
        st.download_button(f"Download {file_format}", buffers[file_format],
            file_name=f"crowdcanvas_{w}x{h}_seed{meta['seed']}.{extension}",
            mime="image/png" if file_format == "PNG" else "image/tiff", width="stretch")
    else:
        st.caption("Generate a preview to compose the artwork. The same arrangement is reused for every export.")

if "layout" in st.session_state:
    st.divider()
    st.subheader("Print dimensions")
    st.caption("Exports use the displayed arrangement. If the frame has a different shape, the artwork is centered with blank margins and keeps its proportions.")
    orientation = st.radio("Preset orientation", ["Portrait", "Landscape"], horizontal=True)
    cols = st.columns(3)
    for index, (label, wc, hc, dpi) in enumerate(PRINT_PRESETS):
        if orientation == "Landscape":
            wc, hc = hc, wc
        with cols[index % 3]:
            if st.button(f"{label} · {dpi} DPI", width="stretch"):
                dims = print_dimensions(wc, hc, dpi)
                if _do_render(max(dims), f"{wc:g} × {hc:g} cm at {dpi} DPI", dimensions=dims, dpi=dpi):
                    st.rerun()
    with st.expander("Custom size", expanded=True):
        c1, c2, c3 = st.columns(3)
        wc = c1.number_input("Width (cm)", 5.0, 200.0, 50.0, 1.0)
        hc = c2.number_input("Height (cm)", 5.0, 200.0, 70.0, 1.0)
        dpi = c3.selectbox("Print DPI", [150, 200, 240, 300], index=2)
        try:
            dims = print_dimensions(wc, hc, dpi)
            st.caption(f"{dims[0]:,} × {dims[1]:,} pixels · {dims[0] * dims[1] / 1e6:.1f} megapixels")
            if st.button("Render custom print", type="primary"):
                if _do_render(max(dims), f"{wc:g} × {hc:g} cm at {dpi} DPI", dimensions=dims, dpi=dpi):
                    st.rerun()
        except ValueError as exc:
            st.info(str(exc))
    if st.button("Return to preview"):
        _do_render(output_size, "preview")
        st.rerun()
    st.divider()
    st.subheader("Inspect the people")
    render_zoom_viewer(st.session_state["output"], height_px=650)
