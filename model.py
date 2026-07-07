import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Import plotting functions from visualization.py
from visualization import (
    plot_raw_waveforms,
    plot_amplitude_distribution,
    plot_feature_importance,
    plot_loss_curve,
    plot_confusion_matrix,
    plot_pca_projection
)

# Filtering constants
fs = 2000.0                      
nyq = 0.5 * fs                   
low = 20.0 / nyq                  
high = 450.0 / nyq  

# Muscle columns
muscle_columns = [
    'r_bicep', 'r_tricep', 'l_bicep', 'l_tricep', 
    'r_thigh', 'r_hamstring', 'l_thigh', 'l_hamstring'
]

# Band-pass filter
def filter_emg(data, muscle_columns, fs=2000):
    filtered_data = data.copy()
    nyq = 0.5 * fs
    low = 20 / nyq
    high = 450 / nyq
    b, a = butter(4, [low, high], btype="band")

    for col in muscle_columns:
        # Remove DC offset
        signal = filtered_data[col] - filtered_data[col].mean()
        # Band-pass filter
        signal = filtfilt(b, a, signal)
        filtered_data[col] = signal

    return filtered_data

# Feature table
def build_feature_table(emg_data, muscle_columns, fs=2000, window_ms=250, overlap=0.5):
    window_size = int(window_ms * fs / 1000)
    step = int(window_size * (1 - overlap))
    all_features = []

    for start in range(0, len(emg_data) - window_size + 1, step):
        row = {}
        for muscle in muscle_columns:
            window = emg_data[muscle].iloc[start:start + window_size].values
            row[f"{muscle}_RMS"] = np.sqrt(np.mean(window**2))
            row[f"{muscle}_MAV"] = np.mean(np.abs(window))
            row[f"{muscle}_VAR"] = np.var(window)
            row[f"{muscle}_SD"] = np.std(window)

        row["activity_id"] = emg_data["activity_id"].iloc[start]
        all_features.append(row)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    print("--- SCRIPT STARTED ---")
    
    # Loading data 
    print("Loading raw dataset...")
    emg_data = pd.read_csv('combined_emg_data.csv')

    # Filter the EMG signals
    print("Filtering signals...")
    emg_data_filtered = filter_emg(emg_data, muscle_columns)

    # Build the feature table
    print("Building statistical windowed features...")
    features = build_feature_table(emg_data_filtered, muscle_columns, fs=fs)
    print(f"Feature table shape: {features.shape}")

    # Separate Features and Targets
    X = features.drop(columns=["activity_id"])
    y = features["activity_id"]

    # Target Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)

    #  Split the data 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Initialize XGBoost model
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        random_state=42,
        eval_metric="mlogloss",
        n_estimators=150,
        early_stopping_rounds=15,   # Automatically stop if validation stops improving
        learning_rate=0.05,         # Safer learning step size
        max_depth=4,                # Shallower trees prevent overfitting
        subsample=0.8,              # Random row selection
        colsample_bytree=0.8,       # Random feature selection
        reg_lambda=1.5              # Regularization
    )
    
    #  Train the model
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    #  Predictions
    y_pred_encoded = model.predict(X_test)

    #  Decode predictions for evaluation
    y_pred_decoded = le.inverse_transform(y_pred_encoded)
    y_test_decoded = le.inverse_transform(y_test)

    # Print Accuracy and Reports
    print("\nAccuracy Score:", accuracy_score(y_test_decoded, y_pred_decoded))
    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))

    # Print Feature Importance Table
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 20 Most Important Features:")
    print(importance.head(20))

    # Raw EMG Waveforms (first 5 sec)
    print("\nGenerating Raw Waveforms Plot...")
    sample_limit = int(5 * fs)
    subset_data = emg_data_filtered.iloc[:sample_limit]
    time_vector = np.arange(len(subset_data)) / fs
    plot_raw_waveforms(time_vector, subset_data[muscle_columns].values, channel_names=muscle_columns)

    # Amplitude Density Distributions
    print("Generating Amplitude Distribution Plot...")
    plot_amplitude_distribution(emg_data_filtered[muscle_columns].values[:100000], channel_names=muscle_columns)

    # PCA Feature Space Projection
    print("Generating PCA Projection Plot...")
    plot_pca_projection(X, y, n_components=2)

    # Confusion Matrix Heatmap
    print("Generating Confusion Matrix Plot...")
    plot_confusion_matrix(y_test_decoded, y_pred_decoded, classes=classes)

    # Feature Importance Chart
    print("Generating and saving Feature Importance Plot...")
    plot_feature_importance(model, X.columns)

    # Training Loss Curve
    print("Generating and saving Loss Curve Plot...")
    plot_loss_curve(model)
