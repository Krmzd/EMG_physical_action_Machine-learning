"""Save the 32 features of one real 250 ms window as payload.json,
to test the SageMaker endpoint.

Usage: python make_payload.py [start_row]
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS = 2000
WINDOW = 500
MUSCLES = [
    "r_bicep", "r_tricep", "l_bicep", "l_tricep",
    "r_thigh", "r_hamstring", "l_thigh", "l_hamstring",
]

df = pd.read_csv("combined_emg_data.csv")

b, a = butter(4, [20 / (FS / 2), 450 / (FS / 2)], btype="band")
for m in MUSCLES:
    df[m] = filtfilt(b, a, df[m] - df[m].mean())

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
window = df.iloc[start:start + WINDOW]

features = {}
for m in MUSCLES:
    x = window[m].values
    features[f"{m}_RMS"] = float(np.sqrt(np.mean(x ** 2)))
    features[f"{m}_MAV"] = float(np.mean(np.abs(x)))
    features[f"{m}_VAR"] = float(np.var(x))
    features[f"{m}_SD"] = float(np.std(x))

with open("payload.json", "w") as f:
    json.dump({"features": features}, f)

print("Saved payload.json. True activity_id:", int(window["activity_id"].iloc[0]))
