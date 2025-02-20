import streamlit as st
import pandas as pd
import pickle
from utils.data_processing import load_and_preprocess_data
from utils.feature_selection import select_features
from utils.model_inference import load_model, predict

def main():
    st.title("Intrusion Detection System")

    # Sidebar for navigation
    st.sidebar.header("Options")
    model_choice = st.sidebar.selectbox("Select Model", ["CatBoost", "LightGBM", "XGBoost", "KNN", "SVM"])

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.write(df.head())

        # Preprocess data
        try:
            X_processed, _, _, _ = load_and_preprocess_data(uploaded_file)
            st.write("### Processed Data Preview:")
            st.write(X_processed.head())
        except Exception as e:
            st.error(f"Error in data preprocessing: {e}")
            return

        # Feature Selection
        try:
            X_selected, selected_features = select_features(X_processed, X_processed.columns)
            st.write(f"### Selected {len(selected_features)} Features:")
            st.write(selected_features)
        except Exception as e:
            st.error(f"Error in feature selection: {e}")
            return

        # Load the selected model
        model_path = f"models/{model_choice.lower()}_model.pkl"
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Make predictions
        try:
            predictions = predict(model, X_selected)
            st.write("### Predictions:")
            prediction_df = pd.DataFrame({"Predicted Label": predictions})
            st.dataframe(prediction_df)
        except Exception as e:
            st.error(f"Error in model inference: {e}")

if __name__ == "__main__":
    main()
