"""
AgroScan AI — Agricultural Plant Disease & Visual Saliency Dashboard.

Streamlit application providing:
- Leaf photograph disease diagnosis
- Top-5 confidence ranking breakdown
- Grad-CAM lesion explainability heatmap
- Scientific dataset limitation notes

Launch:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from agroscan.inference.predictor import PlantDiseasePredictor
from agroscan.visualization.visualizer import LeafVisualizer

st.set_page_config(
    page_title="AgroScan AI — Crop Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 0.1rem;
    }
    .brand-sub {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .healthy-card {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        border-radius: 6px;
    }
    .disease-card {
        background: #ffebee;
        border-left: 5px solid #c62828;
        padding: 1rem;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Diagnosis Settings")
    st.markdown("---")
    cam_alpha = st.slider("Grad-CAM Heatmap Opacity", 0.1, 0.9, 0.5, 0.05)
    top_k_select = st.slider("Top Predictions to Show", 3, 8, 5)

    st.markdown("---")
    st.markdown("### ⚠️ Dataset Limitations")
    st.info(
        "Trained on PlantVillage laboratory dataset (uniform background). "
        "Field photographs with complex soil backgrounds, lighting shifts, or multiple overlapping leaves "
        "may induce out-of-distribution artifacts."
    )

st.markdown('<div class="brand-title">🌿 AgroScan AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-sub">Agricultural Computer Vision, Plant Disease Classification & Grad-CAM Visual Explanations</div>',
    unsafe_allow_html=True,
)

@st.cache_resource
def get_predictor():
    return PlantDiseasePredictor()

predictor = get_predictor()

uploaded_file = st.file_uploader(
    "Upload a plant leaf photograph (Tomato, Potato, Corn, Apple, Grape, Pepper, etc.)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    raw_pil = Image.open(uploaded_file).convert("RGB")
    np_img = np.array(raw_pil)

    with st.spinner("Analyzing botanical pathologies..."):
        diag = predictor.predict(np_img, top_k=top_k_select)

    top = diag.top_prediction

    # Diagnosis Banner
    if top.is_healthy:
        st.markdown(
            f'<div class="healthy-card">'
            f'<h3>✅ {top.crop} — Healthy Leaf</h3>'
            f'<p>Confidence: <b>{top.confidence:.1%}</b> | Inference Time: {diag.latency_ms:.1f} ms</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="disease-card">'
            f'<h3>⚠️ {top.crop} — {top.condition}</h3>'
            f'<p>Confidence: <b>{top.confidence:.1%}</b> | Inference Time: {diag.latency_ms:.1f} ms</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Input Leaf & Saliency Explanation")
        # Overlay Grad-CAM
        overlay = LeafVisualizer.overlay_gradcam(np_img, diag.gradcam_heatmap, alpha=cam_alpha)
        st.image(overlay, caption=f"Grad-CAM Heatmap (Highlights regions driving '{top.condition}' prediction)", use_container_width=True)

    with col2:
        st.markdown("#### Top Diagnosis Candidates")
        chart_data = {p.condition: p.confidence for p in diag.top_k}
        st.bar_chart(chart_data)

        st.markdown("#### Detailed Candidates Breakdown")
        table_rows = [
            {
                "Crop": p.crop,
                "Condition": p.condition,
                "Status": "Healthy" if p.is_healthy else "Pathology",
                "Confidence": f"{p.confidence:.2%}",
            }
            for p in diag.top_k
        ]
        st.table(table_rows)

else:
    st.info("Upload a leaf image to run automated crop disease diagnosis.")
    st.markdown("""
    #### Supported Agricultural Species (38 categories)
    - **Solanaceae**: Tomato (Early Blight, Late Blight, Septoria, Spider Mites), Potato, Bell Pepper
    - **Grains**: Corn / Maize (Common Rust, Northern Leaf Blight, Gray Leaf Spot)
    - **Fruit Trees**: Apple (Apple Scab, Black Rot, Cedar Rust), Peach, Cherry, Grape
    - **Berries & Vines**: Strawberry, Blueberry, Raspberry, Squash
    """)
