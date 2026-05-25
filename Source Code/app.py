"""CrowdCanvas — Streamlit app: turn an image into a Craig-Alan-style crowd portrait.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import hashlib
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

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
    output_size = st.slider("Output size (px, longest side)", 800, 4000, 2000, 200)
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


subject_mask: np.ndarray | None = None
if subject_only:
    try:
        subject_mask = _cached_subject_mask(raw_bytes, subject_model)
        if float(subject_mask.max()) < 0.05:
            st.warning("Subject extraction returned an empty mask — falling back to whole image.")
            subject_mask = None
    except Exception as exc:  # network/model load problem — degrade gracefully
        st.warning(f"Subject extraction unavailable ({exc}); falling back to whole image.")
        subject_mask = None


@st.cache_data(show_spinner=False)
def _cached_face_bbox(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    return detect_face_bbox(img)


face_bbox = _cached_face_bbox(raw_bytes) if face_boost > 1.0 else None

col_in, col_out = st.columns(2)
with col_in:
    st.subheader("Input")
    if face_bbox is not None:
        st.caption(f"✓ Face detected at {face_bbox} — face boost will apply.")
    elif face_boost > 1.0:
        st.caption("⚠ No face detected — face boost will have no effect.")
    st.image(input_img, use_column_width=True)
    if subject_mask is not None:
        with st.expander("Subject mask preview"):
            st.image(
                (subject_mask * 255).astype("uint8"),
                caption="White = where figures will be placed. Black = ignored.",
                use_column_width=True,
            )

with col_out:
    st.subheader("Crowd canvas")
    if st.button("Generate", type="primary", use_container_width=True):
        progress = st.progress(0.0, text="Painting the crowd…")

        def _cb(p: float) -> None:
            progress.progress(min(max(p, 0.0), 1.0), text="Painting the crowd…")

        output = generate_crowd(
            input_img,
            sprites,
            output_size=output_size,
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
            subject_only=subject_only,
            subject_mask=subject_mask,
            background_color=hex_to_rgb(bg_hex),
            paper_grain=paper_grain,
            seed=None if seed_input == 0 else int(seed_input),
            progress_callback=_cb,
        )
        progress.empty()

        st.image(output, use_column_width=True)
        buf = io.BytesIO()
        output.save(buf, format="PNG", optimize=True)
        st.download_button(
            "Download PNG",
            data=buf.getvalue(),
            file_name="crowdcanvas.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.caption("Adjust the sidebar settings, then press **Generate**.")
