import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove leading spaces in column names

    X = df.drop(['Label'], axis=1)
    y = df['Label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Handle missing values
    X_train = X_train.dropna().reset_index(drop=True)
    y_train = y_train.loc[X_train.index].reset_index(drop=True)  # Ensure y_train matches X_train

    # Apply SMOTE to balance dataset (auto handles all minority classes)
    smote = SMOTE(sampling_strategy='auto', random_state=0)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test
