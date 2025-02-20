import pickle
import pandas as pd
import numpy as np

def load_model(model_path):
    """Loads a trained model from a file."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def prepare_features(X, model):
    """Ensures that X has the same features as the trained model."""
    expected_features = model.feature_names_ if hasattr(model, 'feature_names_') else X.columns
    X = X.reindex(columns=expected_features, fill_value=0)  # Fill missing features with 0
    return X

def predict(model, X, batch_size=1000):
    """Makes predictions using the loaded model in batches."""
    X = prepare_features(X, model)
    
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i+batch_size]
        batch_pred = model.predict(batch)
        predictions.extend(batch_pred)
    
    return np.array(predictions)

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("dataset.csv")
    X = df.drop(columns=['Label'])  # Assuming 'Label' is the target column
    
    model = load_model("models/catboost_model.pkl")  # Example model
    predictions = predict(model, X)
    
    print("Predictions:", predictions)
