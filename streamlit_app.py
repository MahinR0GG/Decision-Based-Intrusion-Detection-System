import streamlit as st
from multiapp import MultiApp
from pages import data_preprocessing, feature_selection, model_training, model_inference, lccde_framework, results_visualization

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

st.sidebar.title("🔍 Navigation")
st.sidebar.markdown("---")

app = MultiApp()

app.add_app("📊 Data Preprocessing", data_preprocessing.app)
app.add_app("🔬 Feature Selection", feature_selection.app)
app.add_app("📈 Model Training", model_training.app)
app.add_app("🧠 Model Inference", model_inference.app)
app.add_app("⚙️ LCCDE Framework", lccde_framework.app)
app.add_app("📊 Results Visualization", results_visualization.app)

app.run()

st.sidebar.markdown("---")
st.sidebar.info("Developed for Intrusion Detection System")
