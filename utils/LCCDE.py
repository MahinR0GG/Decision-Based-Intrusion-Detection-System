import numpy as np
from river import stream
from collections import Counter

def determine_leading_models(lgb_f1, xgb_f1, cb_f1, models):
    """Determines the best model for each attack type based on F1 scores."""
    leading_models = []
    for i in range(len(lgb_f1)):
        best_model = max((lgb_f1[i], models["LightGBM"]), 
                         (xgb_f1[i], models["XGBoost"]), 
                         (cb_f1[i], models["CatBoost"]),
                         key=lambda x: x[0])[1]
        leading_models.append(best_model)
    return leading_models

def LCCDE(X_test, y_test, models, leading_models):
    """Implements the LCCDE framework for adaptive model selection."""
    yt, yp = [], []

    for xi, yi in stream.iter_pandas(X_test, y_test):
        xi_array = np.array(list(xi.values())).reshape(1, -1)

        # Get predictions from all models
        y_pred = {name: int(model.predict(xi_array)[0]) for name, model in models.items()}
        proba = {name: np.max(model.predict_proba(xi_array)) if hasattr(model, "predict_proba") else 0 for name, model in models.items()}

        # Decision logic for LCCDE
        unique_preds = list(set(y_pred.values()))

        if len(unique_preds) == 1:
            final_pred = unique_preds[0]
        elif len(unique_preds) == 3:
            leader_preds = {name: pred for name, pred in y_pred.items() if models[name] == leading_models[y_pred[name]]}
            final_pred = list(leader_preds.values())[0] if leader_preds else max(proba, key=proba.get)
        else:
            final_pred = max(y_pred.values(), key=list(y_pred.values()).count)  # Avoids `mode()` issues

        yt.append(yi)
        yp.append(final_pred)

    return yt, yp
