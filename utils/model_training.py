import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Trains the model, makes predictions, and evaluates performance."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    return f1_score(y_test, y_pred, average=None)

def train_models(X_train, X_test, y_train, y_test):
    """Trains all models and returns their F1 scores."""
    models = {
        "LightGBM": lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, learning_rate=0.05, random_state=42),
        "XGBoost": xgb.XGBClassifier(objective="multi:softmax", eval_metric="mlogloss", use_label_encoder=False, random_state=42),
        "CatBoost": cbt.CatBoostClassifier(verbose=0, depth=6, learning_rate=0.03, iterations=500, random_seed=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    }

    f1_scores = {}
    trained_models = {}

    for name, model in models.items():
        f1_scores[name] = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name)
        trained_models[name] = model

    return trained_models, f1_scores
