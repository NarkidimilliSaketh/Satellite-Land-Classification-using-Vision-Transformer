import os
import streamlit as st
from ui_utils import set_page_config, load_json, render_sidebar_info

set_page_config(page_title="Model Analysis | Satellite CV")

st.title("🧠 Model Analysis")
render_sidebar_info()

summary_path = os.path.join("results", "model_summary.json")
if not os.path.exists(summary_path):
    # Generate it dynamically
    from model import create_model, generate_model_summary
    model = create_model()
    generate_model_summary(model)

summary = load_json(summary_path)

if summary:
    st.header("Architecture Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Base Architecture", summary.get("Backbone", "ViT-B/16"))
        st.metric("Input Size", summary.get("Input Size", "3x224x224"))
        st.metric("Output Classes", summary.get("Output Classes", 10))
        
    with col2:
        st.metric("Transfer Learning", "Enabled (Stage 1)")
        st.metric("Pretrained Weights", "IMAGENET1K_V1")
        st.metric("Compute Device", summary.get("Device", "CPU"))
        
    st.markdown("---")
    st.header("Parameter Distribution")
    
    total = summary.get("Number of Parameters", 0)
    trainable = summary.get("Trainable Parameters", 0)
    frozen = summary.get("Frozen Parameters", 0)
    
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Total Parameters", f"{total:,}")
    pcol2.metric("Trainable Parameters", f"{trainable:,}", "Classification Head")
    pcol3.metric("Frozen Parameters", f"{frozen:,}", "Transformer Backbone")
    
    st.markdown("---")
    st.header("Vision Transformer Details")
    st.info("""
    **ViT-B/16**: The Vision Transformer splits images into fixed-size patches (16x16), linearly embeds each of them, adds position embeddings, and feeds the resulting sequence of vectors to a standard Transformer encoder.
    
    For this project, the entire Transformer backbone is frozen to preserve ImageNet features, and only the final Multi-Layer Perceptron (MLP) head is trained to classify the 10 land-use categories.
    """)
else:
    st.warning("Model summary not found. Please ensure Phase 3 evaluation was run.")
