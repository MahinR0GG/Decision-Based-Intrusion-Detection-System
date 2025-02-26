import streamlit as st
import pandas as pd
from utils.data_processing import load_and_preprocess_data

def app():
    st.title("📊 Data Preprocessing")

    # File upload
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 📝 Raw Data Sample")
        st.dataframe(df.head())

        # Preprocess Data
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data(uploaded_file)
            st.success("✅ Data Preprocessed Successfully!")
            
            st.write("### 🔹 Preprocessed Data (Train Set Sample)")
            st.dataframe(X_train.head())

            st.write(f"🔹 **Shape of Train Set:** {X_train.shape}")
            st.write(f"🔹 **Shape of Test Set:** {X_test.shape}")
        
        except Exception as e:
            st.error(f"⚠️ Error in preprocessing: {e}")

if __name__ == "__main__":
    app()
