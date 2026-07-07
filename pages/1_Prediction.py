import os
import streamlit as st
import time
import pandas as pd
import plotly.express as px
from PIL import Image

from ui_utils import set_page_config, load_predictor, render_sidebar_info, display_error

set_page_config(page_title="Prediction | Satellite CV")

st.title("🛰️ Live Prediction")
render_sidebar_info()

if 'history' not in st.session_state:
    st.session_state.history = []

predictor = load_predictor()

if predictor is None:
    display_error("Model could not be loaded. Please ensure best_model.pth exists.")
    st.stop()

st.header("Upload Satellite Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Save temp image for predict.py to use
        temp_path = os.path.join("results", "temp_upload.jpg")
        os.makedirs("results", exist_ok=True)
        image.save(temp_path)
        
        st.markdown("### Inference Results")
        with st.spinner('Running Vision Transformer...'):
            res = predictor.predict_image(temp_path, save_gradcam=True)
            
        if res:
            # KPI Cards
            cols = st.columns(4)
            cols[0].metric("Predicted Class", res['Predicted Class'])
            cols[1].metric("Confidence", f"{res['Confidence']:.4f}")
            cols[2].metric("Confidence Level", res['Confidence Level'])
            cols[3].metric("Inference Time", f"{res['Inference Time (s)']:.4f}s")
            
            st.markdown("---")
            
            # Display Layout
            col_img, col_chart = st.columns([1, 2])
            
            with col_img:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
            with col_chart:
                st.subheader("Class Probabilities")
                # Prepare data for Plotly
                probs_df = pd.DataFrame(res['Top-5 Predictions'])
                probs_df.rename(columns={"Class": "Land Use Class", "Probability": "Confidence"}, inplace=True)
                
                # Highlight predicted class
                probs_df['Color'] = ['#0A2540' if c == res['Predicted Class'] else '#E2E8F0' for c in probs_df['Land Use Class']]
                
                fig = px.bar(probs_df, x='Confidence', y='Land Use Class', orientation='h', 
                             color='Color', color_discrete_map="identity")
                fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown("---")
            st.subheader("Research Insights")
            st.info(f"The Vision Transformer predicts this image as **{res['Predicted Class']}** with a **{res['Confidence Level']}** confidence level ({res['Confidence']:.2%} probability).")
            
            if len(res['Top-5 Predictions']) > 1:
                second_class = res['Top-5 Predictions'][1]
                st.info(f"The second most probable class is **{second_class['Class']}** at {second_class['Probability']:.2%}.")
                
            if 'Grad-CAM Path' in res and os.path.exists(res['Grad-CAM Path']):
                st.success("Grad-CAM Explainability generated successfully! View it in the Explainability tab.")
                st.session_state['last_gradcam'] = res['Grad-CAM Path']
                
            # Add to history
            st.session_state.history.append({
                "Image": uploaded_file.name,
                "Prediction": res['Predicted Class'],
                "Confidence": f"{res['Confidence']:.4f}",
                "Time (s)": f"{res['Inference Time (s)']:.4f}",
                "Timestamp": time.strftime("%H:%M:%S")
            })
            
    except Exception as e:
        display_error(f"Error processing image: {e}")

st.markdown("---")
st.header("Prediction History")

if st.button("Clear History"):
    st.session_state.history = []
    
if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True)
else:
    st.write("No predictions made in this session.")
