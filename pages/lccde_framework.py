import streamlit as st
import pandas as pd
from utils.LCCDE import determine_leading_models, LCCDE
from utils.model_inference import load_model

def app():
    st.title("⚙️ LCCDE Framework")

    # File upload
    uploaded_file = st.file_uploader("📂 Upload CSV file for LCCDE Prediction", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X_test = df.drop(columns=['Label'])
        y_test = df['Label']

        st.write("### 📊 Uploaded Data Sample")
        st.dataframe(X_test.head())

        # Load trained models
        try:
            models = {
                "LightGBM": load_model("models/lightgbm_model.pkl"),
                "XGBoost": load_model("models/xgboost_model.pkl"),
                "CatBoost": load_model("models/catboost_model.pkl"),
            }
            st.success("✅ Models Loaded Successfully!")

            # Determine leading models
            lgb_f1, xgb_f1, cb_f1 = [0.8], [0.82], [0.85]  # Example F1 scores
            leading_models = determine_leading_models(lgb_f1, xgb_f1, cb_f1, models)

            # Apply LCCDE
            yt, yp = LCCDE(X_test, y_test, models, leading_models)
            st.success("✅ LCCDE Prediction Completed!")

            st.write("### 🔹 Predictions")
            prediction_df = pd.DataFrame({"Actual Label": yt, "Predicted Label": yp})
            st.dataframe(prediction_df)

        except Exception as e:
            st.error(f"⚠️ Error in LCCDE framework: {e}")

if __name__ == "__main__":
    app()
