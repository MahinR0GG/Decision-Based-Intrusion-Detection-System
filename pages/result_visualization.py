import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def app():
    st.title("📊 Results Visualization")

    # File upload
    uploaded_file = st.file_uploader("📂 Upload CSV file with Predictions", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "Actual Label" in df.columns and "Predicted Label" in df.columns:
            st.write("### 📊 Uploaded Prediction Data")
            st.dataframe(df.head())

            # Compute Confusion Matrix
            cm = confusion_matrix(df["Actual Label"], df["Predicted Label"])

            # Plot Confusion Matrix
            st.write("### 🔹 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

        else:
            st.error("⚠️ Uploaded file must contain 'Actual Label' and 'Predicted Label' columns.")

if __name__ == "__main__":
    app()
