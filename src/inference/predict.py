import os
import json
import joblib
import pandas as pd

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_names.json")

# ---------------- CACHE ----------------
_model = None
_FEATURES = None


def load_assets():
    global _model, _FEATURES

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    if _FEATURES is None:
        with open(FEATURE_PATH, "r") as f:
            _FEATURES = json.load(f)


def predict_with_details(user_input: dict):
    load_assets()

    # Build full feature vector
    data = {feature: 0 for feature in _FEATURES}
    for k, v in user_input.items():
        if k in data:
            data[k] = v

    df = pd.DataFrame([data])

    prediction = int(_model.predict(df)[0])

    probability = None
    if hasattr(_model, "predict_proba"):
        probability = float(_model.predict_proba(df)[0][1])

    return prediction, probability


def predict(user_input: dict):
    pred, _ = predict_with_details(user_input)
    return pred
