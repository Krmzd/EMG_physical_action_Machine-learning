"""Send one real 250 ms window from the dataset to the API and print the prediction.

Usage: python try_api.py [start_row]
"""
import json
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS = 2000      # sampling rate (Hz)
WINDOW = 500   # 250 ms at 2000 Hz
MUSCLES = [
    "r_bicep", "r_tricep", "l_bicep", "l_tricep",
    "r_thigh", "r_hamstring", "l_thigh", "l_hamstring",
]

df = pd.read_csv("combined_emg_data.csv")

# Same filtering as model.py: remove DC offset, then 20-450 Hz band-pass
b, a = butter(4, [20 / (FS / 2), 450 / (FS / 2)], btype="band")
for m in MUSCLES:
    df[m] = filtfilt(b, a, df[m] - df[m].mean())

# Pick one window (choose the start row on the command line)
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
window = df.iloc[start:start + WINDOW]

# Same 4 features per muscle as build_feature_table() in model.py
features = {}
for m in MUSCLES:
    x = window[m].values
    features[f"{m}_RMS"] = float(np.sqrt(np.mean(x ** 2)))
    features[f"{m}_MAV"] = float(np.mean(np.abs(x)))
    features[f"{m}_VAR"] = float(np.var(x))
    features[f"{m}_SD"] = float(np.std(x))

request = urllib.request.Request(
    "http://127.0.0.1:8080/predict",
    data=json.dumps({"features": features}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request) as response:
    result = json.loads(response.read())

print("True activity_id:", int(window["activity_id"].iloc[0]))
print("API prediction:  ", result)
