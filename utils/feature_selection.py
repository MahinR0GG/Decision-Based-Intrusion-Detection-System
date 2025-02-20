import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=20):
    """Selects the top K best features based on ANOVA F-score."""
    X = X.dropna()  # Ensure no NaN values before feature selection
    
    # Ensure k is not greater than the number of features
    k = min(k, X.shape[1])

    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    return pd.DataFrame(X_new, columns=selected_features), selected_features

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("dataset.csv")
    X = df.drop(columns=['Label'])  # Assuming 'Label' is the target column
    y = df['Label']
    
    X_selected, selected_features = select_features(X, y)
    print("Selected Features:", list(selected_features))
