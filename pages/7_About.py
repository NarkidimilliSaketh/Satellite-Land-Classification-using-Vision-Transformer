import streamlit as st
from ui_utils import set_page_config, render_sidebar_info

set_page_config(page_title="About | Satellite CV")

st.title("ℹ️ About the Project")
render_sidebar_info()

st.markdown("""
### Satellite Land Use Classification using Vision Transformer

This project serves as a professional AI Research Dashboard for remote sensing image classification. It demonstrates the complete lifecycle of a deep learning project, from dataset processing and model training to deployment and explainability.

---

### Tech Stack
- **Python 3.11**
- **PyTorch** (Deep Learning Framework)
- **Vision Transformer (ViT)** (Architecture)
- **Streamlit** (Web Application Framework)
- **Plotly & Seaborn** (Data Visualization)
- **OpenCV & PIL** (Image Processing)
- **Grad-CAM** (Explainability)

---

### Developed For
- Remote Sensing Research
- Academic Demonstrations
- AI/ML Portfolios
- Technical Interviews

*Note: All inferences, metrics, and plots displayed in this dashboard are generated dynamically from real trained weights and test datasets. No simulated values are used.*
""")
