import streamlit as st
import pandas as pd
from utils.model_inference import load_model, predict

def app():
    st.title("🧠 Model Inference")

    # Select a trained model
    model_choice = st.selectbox("🎯 Select a Model for Prediction", ["CatBoost", "LightGBM", "XGBoost", "KNN", "SVM"])

    # File upload
    uploaded_file = st.file_uploader("📂 Upload CSV file for Prediction", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 📊 Uploaded Data Sample")
        st.dataframe(df.head())

        # Load selected model
        model_path = f"models/{model_choice.lower()}_model.pkl"

        try:
            model = load_model(model_path)
            st.success(f"✅ {model_choice} Model Loaded Successfully!")

            # Make Predictions
            predictions = predict(model, df)
            st.write("### 🔹 Predictions")
            prediction_df = pd.DataFrame({"Predicted Label": predictions})
            st.dataframe(prediction_df)

        except Exception as e:
            st.error(f"⚠️ Error in model inference: {e}")

if __name__ == "__main__":
    app()
