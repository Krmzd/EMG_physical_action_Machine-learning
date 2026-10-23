"""Train a second XGBoost model (variant B) for the A/B test,
and save the test windows that both variants will be compared on."""
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from model import filter_emg, build_feature_table, muscle_columns

data = pd.read_csv("combined_emg_data.csv")
features = build_feature_table(filter_emg(data, muscle_columns), muscle_columns, fs=2000)

X = features.drop(columns=["activity_id"])
label_encoder = joblib.load("label_encoder.pkl")   # same encoder as variant A
y = label_encoder.transform(features["activity_id"])

# Same split as model.py (same random_state), so these test windows
# were never used to train either variant
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Variant B: deeper trees, more trees, faster learning rate than variant A
model_b = XGBClassifier(
    random_state=42, eval_metric="mlogloss",
    n_estimators=300, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
)
model_b.fit(X_train, y_train)
accuracy = float((model_b.predict(X_test) == y_test).mean())
print("Variant B offline test accuracy:", round(accuracy, 4))
model_b.save_model("xgboost_emg_classifier_b.json")

# Save the test windows (features + true activity) for the live A/B test
true_ids = label_encoder.inverse_transform(y_test)
samples = [
    {"features": {k: float(v) for k, v in row.items()}, "true": int(t)}
    for row, t in zip(X_test.to_dict(orient="records"), true_ids)
]
with open("ab_payloads.json", "w") as f:
    json.dump(samples, f)
print(f"Saved {len(samples)} test windows to ab_payloads.json")
