import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ui_utils import set_page_config, load_json, render_sidebar_info

set_page_config(page_title="Performance Evaluation | Satellite CV")

st.title("📈 Performance Evaluation")
render_sidebar_info()

metrics = load_json(os.path.join("results", "metrics.json"))

if metrics:
    st.header("Test Set Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
    m2.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
    m3.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
    m4.metric("F1-score", f"{metrics.get('F1-score', 0):.4f}")
    
    st.markdown("---")
    m5, m6, m7 = st.columns(3)
    m5.metric("Top-1 Accuracy", f"{metrics.get('Top-1 Accuracy', 0):.4f}")
    m6.metric("Top-3 Accuracy", f"{metrics.get('Top-3 Accuracy', 0):.4f}")
    m7.metric("Avg Inference Time", f"{metrics.get('Average Inference Time (s)', 0):.4f}s")
else:
    st.warning("Metrics JSON not found.")

st.markdown("---")
st.header("Training Curves (Interactive)")
history_path = os.path.join("outputs", "training_history.csv")
if os.path.exists(history_path):
    df = pd.read_csv(history_path)
    
    tab1, tab2, tab3 = st.tabs(["Accuracy", "Loss", "Learning Rate"])
    
    with tab1:
        fig_acc = px.line(df, x="Epoch", y=["Training Accuracy", "Validation Accuracy"], 
                          title="Accuracy over Epochs", markers=True)
        st.plotly_chart(fig_acc, use_container_width=True)
        
    with tab2:
        fig_loss = px.line(df, x="Epoch", y=["Training Loss", "Validation Loss"], 
                           title="Loss over Epochs", markers=True)
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with tab3:
        fig_lr = px.line(df, x="Epoch", y="Learning Rate", 
                         title="Learning Rate Schedule (Cosine Annealing)", log_y=True, markers=True)
        fig_lr.update_traces(line_color='green')
        st.plotly_chart(fig_lr, use_container_width=True)
else:
    st.info("Training history CSV not found.")

st.markdown("---")
st.header("Confusion Matrix")
cm_path = os.path.join("results", "confusion_matrix.png")
if os.path.exists(cm_path):
    st.image(cm_path, use_container_width=True)
else:
    st.info("Confusion matrix image not found.")

st.markdown("---")
st.header("Classification Report")
report = load_json(os.path.join("results", "classification_report.json"))
if report:
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, subset=['f1-score', 'precision', 'recall']), use_container_width=True)
else:
    st.info("Classification report JSON not found.")
