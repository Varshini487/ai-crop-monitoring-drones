import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd

st.set_page_config(page_title="🚁 AI Crop Monitoring with Drones", layout="wide")
st.title("🚁 AI Crop Monitoring with Drones")
st.markdown("Analyze drone imagery to detect crop stress, disease, and estimate yield")

st.sidebar.header("🛸 Drone Flight Settings")
field_size = st.sidebar.slider("Field Size (hectares)", 5, 100, 25)
disease_intensity = st.sidebar.slider("Disease Presence (0-100%)", 0, 100, 20)
seed = st.sidebar.number_input("Seed", value=42)

np.random.seed(seed)
h, w = 512, 512
field_rgb = np.random.randint(50, 180, (h, w, 3), dtype=np.uint8)
for _ in range(3):
    y, x = np.random.randint(0, h, 2)
    r = np.random.randint(30, 100)
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - y)**2 + (xx - x)**2 <= r**2
    field_rgb[mask] = [100, 50, 50]

nir = np.random.randint(100, 220, (h, w))
red = field_rgb[:, :, 0].astype(float)
ndvi = (nir.astype(float) - red) / (nir.astype(float) + red + 1e-5)
ndvi_normalized = (ndvi + 1) / 2

if st.button("🚀 Analyze Drone Imagery"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Field Area", f"{field_size} ha")
    col2.metric("Ground Resolution", "3 cm/pixel")
    col3.metric("Captured Bands", "RGB + NIR + Red-Edge")

    tab1, tab2, tab3, tab4 = st.tabs(["RGB Image", "NDVI Heatmap", "Disease Map", "Yield Estimate"])

    with tab1:
        st.image(field_rgb, caption="RGB Ortho Mosaic (geo-registered)")

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(ndvi_normalized, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title("NDVI (Vegetation Index) — Red=Stressed, Green=Healthy")
        plt.colorbar(im, ax=ax, label="NDVI")
        st.pyplot(fig)
        
        avg_ndvi = ndvi_normalized.mean()
        st.metric("Average NDVI", f"{avg_ndvi:.2f}", delta="-0.05" if avg_ndvi < 0.6 else "+0.02")
        if avg_ndvi < 0.5:
            st.warning("⚠️ Low NDVI across field — widespread stress detected")

    with tab3:
        disease_map = np.random.rand(h, w) < (disease_intensity / 100)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(field_rgb)
        ax.contourf(disease_map.astype(int), levels=[0.5, 1.5], colors=['red'], alpha=0.4)
        ax.set_title("Disease Detection (Red = Infected Area)")
        st.pyplot(fig)
        
        disease_pct = disease_map.mean() * 100
        st.error(f"🚨 Diseased Area: {disease_pct:.1f}% of field")
        if disease_pct > 15:
            st.warning("Consider targeted spray in affected zones to contain spread")

    with tab4:
        yield_pred = 8.5 + (avg_ndvi - 0.5) * 5 + np.random.normal(0, 0.3)
        st.metric("Predicted Yield", f"{yield_pred:.1f} T/ha", "Reference: 8.0 T/ha")
        st.write(f"Based on NDVI ({avg_ndvi:.2f}) and historical patterns, estimated yield is **{yield_pred:.1f} metric tons per hectare**.")

st.markdown("---")
st.caption("Demo uses simulated imagery. Integrate real drone APIs (DJI, senseFly) for production use.")
