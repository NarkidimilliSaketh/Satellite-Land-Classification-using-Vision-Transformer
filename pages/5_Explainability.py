import os
import streamlit as st
from PIL import Image
from ui_utils import set_page_config, load_predictor, render_sidebar_info, display_error

set_page_config(page_title="Explainability | Satellite CV")

st.title("🔍 Explainability (Grad-CAM)")
render_sidebar_info()

st.markdown("""
### What is Grad-CAM?
Gradient-weighted Class Activation Mapping (**Grad-CAM**) produces a coarse localization map highlighting the important regions in the image for predicting the target concept.
For **Vision Transformers (ViT)**, we extract the gradients flowing into the final Layer Normalization of the Transformer Encoder. This allows us to see exactly which patches of the satellite image the model paid the most attention to when making its classification.
""")

st.markdown("---")

predictor = load_predictor()

st.header("Analyze an Image")
uploaded_file = st.file_uploader("Upload an image to generate attention heatmaps", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        temp_path = os.path.join("results", "temp_gradcam.jpg")
        os.makedirs("results", exist_ok=True)
        image.save(temp_path)
        
        with st.spinner("Generating Grad-CAM..."):
            res = predictor.predict_image(temp_path, save_gradcam=True)
            
        if res and 'Grad-CAM Path' in res and os.path.exists(res['Grad-CAM Path']):
            st.success(f"Prediction: **{res['Predicted Class']}** ({res['Confidence Level']} Confidence)")
            
            st.subheader("Attention Visualization")
            # The generate_gradcam function saves a figure with 3 subplots (Original, Heatmap, Overlay)
            st.image(res['Grad-CAM Path'], use_container_width=True)
            
            st.info("""
            **Interpretation Guide:**
            - **Original Image**: The raw satellite input.
            - **Attention Heatmap**: Red/Warm areas indicate high attention. Blue/Cool areas indicate low attention.
            - **Overlay**: The heatmap superimposed on the original image. Look at the red areas to understand what geographic features drove the prediction.
            """)
    except Exception as e:
        display_error(f"Error generating Grad-CAM: {e}")
elif 'last_gradcam' in st.session_state and os.path.exists(st.session_state['last_gradcam']):
    st.info("Showing the Grad-CAM from your latest prediction in the Prediction tab.")
    st.image(st.session_state['last_gradcam'], use_container_width=True)
