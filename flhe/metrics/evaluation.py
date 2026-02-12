import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

def evaluate_binary(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["auc"] = None
    else:
        out["auc"] = None
    return out

def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p[:,1]
    return None

def flatten_cm(cm_list):
    cm = np.array(cm_list, dtype=int)
    return cm
