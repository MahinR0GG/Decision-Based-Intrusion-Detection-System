import streamlit as st
import pandas as pd
from utils.model_training import train_models

def app():
    st.title("📈 Model Training")

    # File upload
    uploaded_file = st.file_uploader("📂 Upload Feature-Selected CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X_train = df.drop(columns=['Label'])
        y_train = df['Label']

        st.write("### 📊 Dataset Preview")
        st.dataframe(X_train.head())

        # Train Models Button
        if st.button("🚀 Train Models"):
            st.write("⏳ Training in progress... Please wait.")
            
            try:
                trained_models, f1_scores = train_models(X_train, X_train, y_train, y_train)
                st.success("✅ Model Training Completed!")

                # Display F1 Scores
                st.write("### 🔹 Model Performance (F1 Scores)")
                st.json(f1_scores)

            except Exception as e:
                st.error(f"⚠️ Error in model training: {e}")

if __name__ == "__main__":
    app()
