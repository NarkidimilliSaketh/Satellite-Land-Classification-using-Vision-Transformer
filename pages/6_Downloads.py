import os
import streamlit as st
from ui_utils import set_page_config, render_sidebar_info

set_page_config(page_title="Downloads | Satellite CV")

st.title("💾 Downloads & Exports")
render_sidebar_info()

st.markdown("Download the generated artifacts and reports from the Phase 3 Evaluation.")

artifacts = {
    "Evaluation Report (PDF)": os.path.join("results", "evaluation_report.pdf"),
    "Predictions (CSV)": os.path.join("results", "prediction.csv"),
    "Classification Report (JSON)": os.path.join("results", "classification_report.json"),
    "Metrics Summary (JSON)": os.path.join("results", "metrics.json"),
    "Model Summary (JSON)": os.path.join("results", "model_summary.json"),
    "Dataset Summary (JSON)": os.path.join("results", "dataset_analysis", "dataset_summary.json")
}

for name, path in artifacts.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            bytes_data = f.read()
            st.download_button(
                label=f"Download {name}",
                data=bytes_data,
                file_name=os.path.basename(path),
                mime="application/octet-stream"
            )
    else:
        st.button(f"{name} (Not Available)", disabled=True)
