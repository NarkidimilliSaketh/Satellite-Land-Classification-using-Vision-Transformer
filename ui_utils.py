import os
import json
import pandas as pd
import streamlit as st
import base64
from PIL import Image

def set_page_config(page_title="AI Research Dashboard"):
    st.set_page_config(
        page_title=page_title,
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for the White/Dark Blue/Light Gray theme
    st.markdown("""
        <style>
        .reportview-container .main .block-container{
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #0A2540;
        }
        .stMetric {
            background-color: #F8F9FA;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            border-left: 5px solid #0A2540;
        }
        div[data-testid="stSidebar"] {
            background-color: #F8F9FA;
            border-right: 1px solid #E2E8F0;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    from predict import Predictor
    try:
        predictor = Predictor(use_gradcam=True)
        return predictor
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def load_csv(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        return pd.read_csv(filepath)
    except:
        return None

def display_error(msg):
    st.error(f"⚠️ {msg}")

def render_sidebar_info():
    st.sidebar.title("Project Information")
    st.sidebar.markdown("**Model:** ViT-B/16")
    st.sidebar.markdown("**Dataset:** EuroSAT")
    st.sidebar.markdown("**Classes:** 10")
    st.sidebar.markdown("**Framework:** PyTorch")
    
def get_image_base64(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
