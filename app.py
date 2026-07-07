import streamlit as st
from ui_utils import set_page_config, render_sidebar_info

set_page_config(page_title="Home | Satellite CV")

st.title("Satellite Land Use Classification using Vision Transformer")
st.subheader("Transformer-based Remote Sensing Image Classification")

render_sidebar_info()

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", "ViT-B/16")
    st.metric("Transfer Learning", "Enabled")
with col2:
    st.metric("Dataset", "EuroSAT")
    st.metric("Framework", "PyTorch")
with col3:
    st.metric("Classes", "10")
    st.metric("Device", "CPU / GPU")

st.markdown("---")

st.header("Project Overview")
st.write("""
This project develops a production-ready deep learning application that classifies satellite images into 10 distinct land-use categories. 
Leveraging a pretrained Vision Transformer (ViT), it emphasizes modular architecture, reproducibility, and research-quality evaluation.
""")

st.header("Workflow Diagram")
# We will use the project banner as a stand-in for the workflow diagram
st.image("assets/project_banner.png", use_container_width=True)

st.header("Technology Stack")
cols = st.columns(4)
techs = ["Python 3.11.9", "PyTorch", "Streamlit", "Plotly", "Torchvision", "Scikit-Learn", "Grad-CAM", "Seaborn"]
for i, tech in enumerate(techs):
    cols[i % 4].markdown(f"- **{tech}**")

st.markdown("---")
st.info("👈 Use the Sidebar to navigate through the Research Dashboard.")
