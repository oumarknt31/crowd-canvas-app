"""CrowdCanvas — Streamlit app: turn an image into a Craig-Alan-style crowd portrait.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import io
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

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
    generate_crowd,
    hex_to_rgb,
    load_sprites,
)

HERE = Path(__file__).parent
SPRITES_DIR = HERE / "Artwork people"

st.set_page_config(page_title="CrowdCanvas", page_icon="🧑‍🎨", layout="wide")
st.title("CrowdCanvas")
st.caption(
    "Upload an image — the app rebuilds it as a crowd of tiny figures, in the "
    "spirit of Craig Alan's *Populus* portraits. Works best with high-contrast "
    "subjects (faces, silhouettes, logos)."
)


# ------- Print / display presets ---------------------------------------------
# Each preset specifies the longest pixel side. The image's aspect ratio is
# preserved (output_size in generate_crowd is the longest side).
PRINT_PRESETS = [
    {"label": "Preview", "longest_px": 1600,
     "physical": "Quick preview render"},
    {"label": "HD wallpaper", "longest_px": 1920,
     "physical": "Fits a 1920×1080 screen"},
    {"label": "4K wallpaper", "longest_px": 3840,
     "physical": "Fits a 3840×2160 screen"},
    {"label": "A4 print · 300 DPI", "longest_px": 3508,
     "physical": "21 × 29.7 cm"},
    {"label": "A3 print · 300 DPI", "longest_px": 4961,
     "physical": "29.7 × 42 cm"},
    {"label": "60 cm canvas · 250 DPI", "longest_px": 5906,
     "physical": "≈60 cm long side"},
    {"label": "1 m canvas · 200 DPI", "longest_px": 7874,
     "physical": "≈1 m long side"},
    {"label": "1.5 m canvas · 150 DPI", "longest_px": 8858,
     "physical": "≈1.5 m long side"},
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


# ------- Sprite + per-upload caches ------------------------------------------
@st.cache_resource(show_spinner="Loading people sprites…")
def _load_sprites_cached(path: str):
    return load_sprites(path)


sprites = _load_sprites_cached(str(SPRITES_DIR))
if not sprites:
    st.error(f"No PNG sprites found in `{SPRITES_DIR}`.")
    st.stop()
st.sidebar.success(f"Loaded {len(sprites)} people sprites")


with st.sidebar:
    st.header("Settings")
    subject_only = st.toggle(
        "Focus on subject only", value=True,
        help="Auto-removes the background so figures only land on the person/subject. "
             "First run downloads a ~170MB model.",
    )
    subject_model = st.selectbox(
        "Subject detector",
        options=["u2net_human_seg", "u2net", "isnet-general-use"],
        index=0,
        help="Use the human-seg model for portraits. Switch to 'u2net' / 'isnet' for "
             "non-human subjects (objects, animals, paintings).",
        disabled=not subject_only,
    )
    output_size = st.slider(
        "Preview size (px, longest side)", 800, 4000, 2000, 200,
        help="Resolution used for the on-screen preview render. Use the print presets "
             "below to render at higher physical sizes.",
    )
    density_count = st.slider("Crowd density (figures placed)", 200, 10000, 2800, 100)
    scatter_count = st.slider(
        "Background scatter (stragglers)", 0, 500, 0, 5,
        help="Stragglers placed anywhere on the canvas, ignoring the subject mask. "
             "Set to 0 to keep figures strictly on the subject.",
    )

    st.markdown("**Face emphasis**")
    face_boost = st.slider(
        "Face boost", 1.0, 4.0, 2.4, 0.1,
        help="Multiplier on density inside the detected face. 1.0 = body and face equal; "
             "2.4 = face gets ~2.4× the figures of shoulders/torso.",
    )
    detail_strength = st.slider(
        "Detail enhancement", 0.0, 1.5, 0.6, 0.1,
        help="Sharpens edges before density extraction so eyes, lips, glasses, and hair "
             "strands register as denser regions.",
    )
    color_match_strength = st.slider(
        "Match input colors", 0.0, 0.7, 0.25, 0.05,
        help="Tints each figure toward the input image's color at its position. "
             "0 = keep original sprite colors; higher values blend the figure's clothing "
             "toward the underlying subject color (red tie → reddish figures, etc.).",
    )

    st.markdown("**Crowd behavior**")
    selectivity = st.slider(
        "Selectivity", 1.0, 6.0, 3.0, 0.2,
        help="How tightly figures cluster on dark features. Higher = sharper portrait, more empty space.",
    )
    min_density = st.slider(
        "Density floor", 0.00, 0.50, 0.10, 0.02,
        help="Pixels lighter than this never receive figures — keeps the background clean.",
    )
    sprite_height_pct = st.slider(
        "Person size (% of canvas height)", 0.008, 0.040, 0.015, 0.001,
        help="Smaller figures resolve finer features (eyes, mouth) but need more figures total.",
    )
    scale_jitter = st.slider("Size variation", 0.00, 0.50, 0.18, 0.05)
    gamma = st.slider(
        "Contrast (gamma)", 0.5, 3.0, 2.0, 0.1,
        help="Higher = denser dark areas, sharper crowd shapes.",
    )
    blur = st.slider("Smoothing radius", 0.0, 6.0, 1.0, 0.5)
    bg_hex = st.color_picker("Background color", "#F5F0E6")
    paper_grain = st.slider("Paper grain", 0.00, 0.05, 0.015, 0.005)
    seed_input = st.number_input(
        "Random seed (0 = random)", min_value=0, max_value=999_999, value=0, step=1
    )

uploaded = st.file_uploader(
    "Input image", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"]
)

if uploaded is None:
    st.info("Upload an image to begin. Tip: portraits with strong shadows look best.")
    st.stop()

raw_bytes = uploaded.getvalue()
input_img = Image.open(io.BytesIO(raw_bytes))


@st.cache_data(show_spinner="Extracting subject (first run downloads a ~170MB model)…")
def _cached_subject_mask(image_bytes: bytes, model: str) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    return extract_subject_mask(img, model=model)


subject_mask: Optional[np.ndarray] = None
if subject_only:
    try:
        subject_mask = _cached_subject_mask(raw_bytes, subject_model)
        if float(subject_mask.max()) < 0.05:
            st.warning("Subject extraction returned an empty mask — falling back to whole image.")
            subject_mask = None
    except Exception as exc:
        st.warning(f"Subject extraction unavailable ({exc}); falling back to whole image.")
        subject_mask = None


@st.cache_data(show_spinner=False)
def _cached_face_bbox(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    return detect_face_bbox(img)


face_bbox = _cached_face_bbox(raw_bytes) if face_boost > 1.0 else None

# Clear last render when a new image is uploaded.
upload_key = uploaded.name + ":" + str(len(raw_bytes))
if st.session_state.get("upload_key") != upload_key:
    st.session_state["upload_key"] = upload_key
    st.session_state.pop("output", None)
    st.session_state.pop("output_meta", None)
    st.session_state.pop("manual_mask", None)
    st.session_state["mask_canvas_key"] = 0

# Manual mask edits override the auto-detected mask if present.
manual_mask: Optional[np.ndarray] = st.session_state.get("manual_mask")
effective_subject_mask = manual_mask if manual_mask is not None else subject_mask


# ------- Render helper -------------------------------------------------------
def _do_render(target_size: int, label: str) -> None:
    progress = st.progress(0.0, text=f"Rendering {label} ({target_size}px longest side)…")

    def _cb(p: float) -> None:
        progress.progress(min(max(p, 0.0), 1.0),
                          text=f"Rendering {label} ({target_size}px longest side)…")

    output = generate_crowd(
        input_img,
        sprites,
        output_size=target_size,
        density_count=density_count,
        scatter_count=scatter_count,
        sprite_height_pct=sprite_height_pct,
        scale_jitter=scale_jitter,
        gamma=gamma,
        blur=blur,
        selectivity=selectivity,
        min_density=min_density,
        detail_strength=detail_strength,
        face_boost=face_boost,
        face_bbox=face_bbox,
        color_match_strength=color_match_strength,
        subject_only=subject_only,
        subject_mask=effective_subject_mask,
        background_color=hex_to_rgb(bg_hex),
        paper_grain=paper_grain,
        seed=None if seed_input == 0 else int(seed_input),
        progress_callback=_cb,
    )
    progress.empty()
    st.session_state["output"] = output
    st.session_state["output_meta"] = {"label": label, "size_px": output.size}


# ------- Layout: input + generate --------------------------------------------
col_in, col_out = st.columns([1, 1])
with col_in:
    st.subheader("Input")
    if face_bbox is not None:
        st.caption(f"✓ Face detected at {face_bbox} — face boost will apply.")
    elif face_boost > 1.0:
        st.caption("⚠ No face detected — face boost will have no effect.")
    st.image(input_img, use_container_width=True)

    # Subject mask preview + manual editor
    has_mask = effective_subject_mask is not None
    label = "Subject mask & manual editor"
    if manual_mask is not None:
        label += " · manual edits applied"
    with st.expander(label, expanded=False):
        if has_mask:
            st.image(
                (effective_subject_mask * 255).clip(0, 255).astype("uint8"),
                caption=(
                    "White = figures land here, black = ignored. "
                    + ("Showing manually edited mask." if manual_mask is not None
                       else "Showing auto-detected mask.")
                ),
                use_container_width=True,
            )
        else:
            st.info(
                "No mask available. Turn on **Focus on subject only** in the sidebar "
                "to auto-detect a subject, or paint a mask from scratch below."
            )

        edit_on = st.toggle(
            "Edit mask manually", value=False, key="edit_mask_toggle",
            help="Paint to add regions (red brush) or remove regions (blue brush) the "
                 "auto detector got wrong, e.g. missed hair tendrils or stray hands.",
        )
        if edit_on:
            brush_mode = st.radio(
                "Brush", ["Add to mask (red)", "Remove from mask (blue)"],
                horizontal=True, key="brush_mode",
            )
            stroke_color = (ADD_BRUSH_HEX if brush_mode.startswith("Add")
                            else REMOVE_BRUSH_HEX)
            brush_size = st.slider("Brush size", 5, 80, 28, key="brush_size")

            cw, ch = _canvas_dims(input_img, max_side=700)
            canvas_key = f"mask_canvas_{st.session_state.get('mask_canvas_key', 0)}"
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=brush_size,
                stroke_color=stroke_color,
                background_image=input_img.convert("RGB"),
                update_streamlit=True,
                height=ch,
                width=cw,
                drawing_mode="freedraw",
                key=canvas_key,
            )

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Apply edits", use_container_width=True, type="primary"):
                    if (canvas_result.image_data is not None
                            and np.asarray(canvas_result.image_data)[..., 3].max() > 0):
                        target_hw = (
                            subject_mask.shape if subject_mask is not None
                            else (input_img.height, input_img.width)
                        )
                        st.session_state["manual_mask"] = _apply_strokes(
                            subject_mask, canvas_result.image_data, target_hw
                        )
                        st.session_state["mask_canvas_key"] = (
                            st.session_state.get("mask_canvas_key", 0) + 1
                        )
                        st.rerun()
                    else:
                        st.warning("Nothing to apply — paint some strokes first.")
            with b2:
                if st.button("Clear strokes", use_container_width=True):
                    st.session_state["mask_canvas_key"] = (
                        st.session_state.get("mask_canvas_key", 0) + 1
                    )
                    st.rerun()
            with b3:
                if st.button("Reset to auto", use_container_width=True,
                             disabled=manual_mask is None):
                    st.session_state.pop("manual_mask", None)
                    st.session_state["mask_canvas_key"] = (
                        st.session_state.get("mask_canvas_key", 0) + 1
                    )
                    st.rerun()

with col_out:
    st.subheader("Crowd canvas")
    if st.button("Generate preview", type="primary", use_container_width=True):
        _do_render(output_size, "preview")

    out_meta = st.session_state.get("output_meta")
    if out_meta:
        w, h = out_meta["size_px"]
        st.caption(
            f"Showing **{out_meta['label']}** · {w} × {h} px · "
            f"~{w * h / 1_000_000:.1f} megapixels"
        )
        out_img = st.session_state["output"]
        buf = io.BytesIO()
        out_img.save(buf, format="PNG", optimize=True)
        st.download_button(
            "Download full-resolution PNG",
            data=buf.getvalue(),
            file_name=f"crowdcanvas_{out_meta['label'].replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.caption("Press **Generate preview** to render at the sidebar's preview size.")

# ------- Print / display preset row ------------------------------------------
if st.session_state.get("output") is not None:
    st.markdown("---")
    st.subheader("Render at print or display size")
    st.caption(
        "Each preset re-runs the algorithm at the target pixel resolution while "
        "preserving your input's aspect ratio. Larger sizes take longer."
    )
    cols_per_row = 4
    for row_start in range(0, len(PRINT_PRESETS), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, preset in enumerate(PRINT_PRESETS[row_start:row_start + cols_per_row]):
            with cols[i]:
                if st.button(
                    preset["label"],
                    key=f"preset_{row_start + i}",
                    use_container_width=True,
                    help=f"{preset['physical']} · {preset['longest_px']} px longest side",
                ):
                    _do_render(preset["longest_px"], preset["label"])
                    st.rerun()
                st.caption(f"{preset['longest_px']}px · {preset['physical']}")

    # ------- Zoom / pan viewer ----------------------------------------------
    st.markdown("---")
    st.subheader("Zoom in & inspect the figures")
    st.caption(
        "Drag to pan, scroll to zoom (centered on the cursor), or use the buttons. "
        "On touch devices, pinch to zoom."
    )
    render_zoom_viewer(st.session_state["output"], height_px=720)
