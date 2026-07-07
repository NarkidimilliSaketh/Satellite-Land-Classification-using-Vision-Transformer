import os
import streamlit as st
import pandas as pd
from ui_utils import set_page_config, load_json, render_sidebar_info

set_page_config(page_title="Dataset Analysis | Satellite CV")

st.title("📊 Dataset Analysis")
render_sidebar_info()

st.header("Dataset Overview: EuroSAT")
st.write("EuroSAT is a land use and land cover classification dataset based on Sentinel-2 satellite images.")

summary_path = os.path.join("results", "reports", "dataset_summary.json")
if not os.path.exists(summary_path):
    summary_path = os.path.join("results", "dataset_analysis", "dataset_summary.json")
summary = load_json(summary_path)

if summary:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", summary.get("Total Images", 27000))
    col2.metric("Classes", summary.get("Number of Classes", 10))
    col3.metric("Resolution", "64x64")
    
    st.markdown("---")
    st.header("Data Splits")
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Training (70%)", summary.get("Training Samples", 18900))
    scol2.metric("Validation (15%)", summary.get("Validation Samples", 4050))
    scol3.metric("Testing (15%)", summary.get("Testing Samples", 4050))
    
    st.markdown("---")
    st.header("Class Distribution")
    dist_img = os.path.join("results", "dataset_analysis", "class_distribution.png")
    if os.path.exists(dist_img):
        st.image(dist_img, caption="Distribution of images across 10 land use classes", use_container_width=True)
        
    st.markdown("---")
    st.header("Sample Images")
    grid_img = os.path.join("results", "dataset_analysis", "sample_grid.png")
    if os.path.exists(grid_img):
        st.image(grid_img, caption="Sample images from each class", use_container_width=True)
else:
    st.warning("Dataset summary not found. Please ensure Phase 1 explorer was run.")
