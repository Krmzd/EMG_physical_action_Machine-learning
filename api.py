"""
FastAPI app for the XGBoost EMG physical-action classifier.

Endpoints:
  GET  /health       - is the API running?
  POST /predict      - 32 features in, predicted action out
  GET  /ping         - health check required by AWS SageMaker
  POST /invocations  - prediction endpoint required by AWS SageMaker
"""
from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier
import os

MODEL_PATH = os.getenv("MODEL_PATH", "xgboost_emg_classifier.json")
ENCODER_PATH = "label_encoder.pkl"

MUSCLE_COLUMNS = [
    "r_bicep", "r_tricep", "l_bicep", "l_tricep",
    "r_thigh", "r_hamstring", "l_thigh", "l_hamstring",
]
STATS = ["RMS", "MAV", "VAR", "SD"]

# Same order as build_feature_table() in model.py: each muscle, then its 4 stats
FEATURE_COLUMNS = [f"{muscle}_{stat}" for muscle in MUSCLE_COLUMNS for stat in STATS]

ACTIVITY_NAMES = {
    0: "Bowing", 1: "Clapping", 2: "Handshaking", 3: "Hugging", 4: "Kneeing",
    5: "Elbowing", 6: "Frontkicking", 7: "Hammering", 8: "Headering", 9: "Jumping",
}

# Load the model and label encoder once, when the API starts
model = XGBClassifier()
model.load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

app = FastAPI(title="EMG Physical Action Classification API")


class EMGFeatures(BaseModel):
    """The 32 windowed EMG features, keyed by name, e.g. 'r_bicep_RMS'."""
    features: Dict[str, float]


def classify(features: Dict[str, float]) -> dict:
    """Check the input, run the model, and return a readable answer."""
    missing = [col for col in FEATURE_COLUMNS if col not in features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    row = np.array([[features[col] for col in FEATURE_COLUMNS]])
    probabilities = model.predict_proba(row)[0]
    best = int(np.argmax(probabilities))
    activity_id = int(label_encoder.inverse_transform([best])[0])

    return {
        "activity_id": activity_id,
        "activity": ACTIVITY_NAMES.get(activity_id, "unknown"),
        "class": "Normal" if activity_id <= 4 else "Aggressive",
        "confidence": round(float(probabilities[best]), 4),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: EMGFeatures):
    return classify(data.features)


# --- Endpoints required by AWS SageMaker ---

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/invocations")
def invocations(data: EMGFeatures):
    return classify(data.features)
