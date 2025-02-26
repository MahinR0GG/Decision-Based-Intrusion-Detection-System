import streamlit as st
import pandas as pd
from utils.feature_selection import select_features

def app():
    st.title("🔬 Feature Selection")

    # File upload
    uploaded_file = st.file_uploader("📂 Upload Preprocessed CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X = df.drop(columns=['Label'])
        y = df['Label']

        # Select K features
        k = st.slider("🔢 Select the number of top features:", min_value=1, max_value=len(X.columns), value=20)

        try:
            X_selected, selected_features = select_features(X, y, k)
            st.success("✅ Feature Selection Completed!")

            st.write(f"### 🔹 Selected Top {k} Features:")
            st.write(selected_features)

            st.write("### 📊 Transformed Feature Set (Sample):")
            st.dataframe(X_selected)

        except Exception as e:
            st.error(f"⚠️ Error in feature selection: {e}")

if __name__ == "__main__":
    app()
