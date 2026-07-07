import pandas as pd
import os 

files = {
    "normal": {
        "Bowing.csv": 0,
        "Clapping.csv": 1,
        "Handshaking.csv": 2,
        "Hugging.csv": 3,
        "Kneeing.csv": 4,
    },
    "aggressive": {
        "Elbowing.csv": 5,
        "Frontkicking.csv": 6,
        "Hamering.csv": 7,
        "Headering.csv": 8,
        "Jumping.csv": 9,
    }
}

muscle_headers = [
    'r_bicep', 'r_tricep', 'l_bicep', 'l_tricep', 
    'r_thigh', 'r_hamstring', 'l_thigh', 'l_hamstring', 'is_aggressive'
]

all_files =[]

for i in files:
    for filename, activity_id in files[i].items():
        df = pd.read_csv(filename, header=0, names=muscle_headers)
        df['activity_id'] = activity_id
        df['is_aggressive'] = 1 if i == 'aggressive' else 0

        all_files.append(df)

combined_df = pd.concat(all_files, ignore_index=True)
combined_df.to_csv("combined_emg_data.csv", index=False)