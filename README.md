# 🚀 Intrusion Detection System using LCCDE  

🔹 **Intrusion Detection System using LCCDE** is an advanced machine learning-based IDS that dynamically selects the best model for each attack type using the **Leading Classifier-based Confidence Decision Engine (LCCDE)**.  

🔹 Instead of relying on a **single model**, LCCDE improves accuracy by selecting the **most suitable model dynamically**.  

---

## 📜 **Project Overview**  
### 🚀 **Key Features:**  
✅ **Uses CICIDS2017 Dataset** for real-world network intrusion detection.  
✅ **Trains & evaluates 5 models**:  
   - LightGBM  
   - XGBoost  
   - CatBoost  
   - KNN  
   - SVM  
✅ **LCCDE Framework** dynamically selects the best model for each attack type.  
✅ **Fully interactive UI** built with **Streamlit** for easy dataset upload, training, and visualization.  

---

## 🔧 **Technologies Used**  
- 🐍 **Python**  
- 📊 **Streamlit** (for UI)  
- 🧠 **LightGBM, XGBoost, CatBoost, KNN, SVM** (for ML models)  
- 🔢 **Scikit-learn** (for feature selection, model evaluation)  
- 🔍 **SMOTE** (for data balancing)  
- 📈 **Seaborn & Matplotlib** (for visualizations)  

---

## 📌 **How It Works**  
1️⃣ **Data Preprocessing** - Handles missing values & balances dataset using **SMOTE**.  
2️⃣ **Feature Selection** - Selects best features using **ANOVA F-score**.  
3️⃣ **Model Training** - Trains **LightGBM, XGBoost, CatBoost, KNN, SVM** and saves models.  
4️⃣ **Model Inference** - Predicts network intrusions using trained models.  
5️⃣ **LCCDE Framework** - Selects the **best model dynamically** for each attack type.  
6️⃣ **Results Visualization** - Displays confusion matrices & model performance.  

---
