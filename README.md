# Multi-Channel EMG Physical Action Recognition
### End-to-End Database Design, SQL Analytics, and Machine Learning Classification

This repository contains an end-to-end data engineering, SQL analytics, and machine learning pipeline designed to decode human movement intent. Using multi-channel surface electromyography (sEMG) signals, this project cleans high-frequency data, structures it inside a relational database, audits dataset quality with SQL, and trains a tuned XGBoost model to classify physical behavior categories (Normal vs. Aggressive).

---

## Technical Pipeline Architecture

```text
  [ 10 Raw CSV Files ] 
            │ (Python ETL: merge_emg_data.py)
            ▼
  [ combined_emg_data.csv ]
            │ (MySQL Import Wizard)
            ▼
  [ MySQL Database: 'emg_action' ] (Quality Auditing & Feature Profiling)
            │ (Data Load & Processing: model.py)
            ▼
  [ Remove DC Offset & Butterworth Band-Pass Filter (20-450 Hz) ]
            │
            ▼
  [ Extract Sliding Window Features (250ms Windows, 50% Overlap) ]
  (Features extracted per muscle: RMS, MAV, Variance, Standard Deviation)
            │ (Data Split)
            ▼
  [ Train-Test Split (80/20) with Target Stratification ]
            │ (Modeling)
            ▼
  [ Train Tuned XGBoost Classifier with Early Stopping & Regularization ]
            │ (Performance Evaluation)
            ▼
  [ Visualization Modules (visualization.py) ] 
---
## Dataset and Hardware Specifications

The dataset monitors 8 muscle channels across the upper and lower limbs:
1. r_bicep (Right Bicep)
2. r_tricep (Right Tricep)
3. l_bicep (Left Bicep)
4. l_tricep (Left Tricep)
5. r_thigh (Right Thigh)
6. r_hamstring (Right Hamstring)
7. l_thigh (Left Thigh)
8. l_hamstring (Left Hamstring)

Subjects performed 10 distinct physical actions grouped into two behavioral classes:
*   Normal (0): Bowing, Clapping, Handshaking, Hugging, Kneeing (mapped to activity IDs 0 to 4)
*   Aggressive (1): Elbowing, Frontkicking, Hammering, Headering, Jumping (mapped to activity IDs 5 to 9)

---

## Step 1 - Data Engineering and ETL Pipeline

Raw time-series files are originally stored across separate experimental logs with overlapping indices. The script merge_emg_data.py performs an ETL pipeline to map and compile these tables. 

*   **Header Mapping:** Cleans the file read by skipping the original trapped headers (1.0, 2.0, ... label).
*   **Target Generation:** Dynamically assigns the target label is_aggressive (0 or 1) based on the dictionary grouping directory.
*   **Activity Mapping:** Appends the numeric activity_id (0 to 9) to distinguish between specific movements.
*   **Data Export:** Consolidates all tables vertically and exports them to a unified file named combined_emg_data.csv containing 10 distinct columns.

---

## Step 2 - Relational Database Design and SQL Analytics

### Database Schema Blueprints
```sql
CREATE DATABASE emg_action;
USE emg_action;

CREATE TABLE emg (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    r_bicep FLOAT,
    r_tricep FLOAT, 
    l_bicep FLOAT, 
    l_tricep FLOAT, 
    r_thigh FLOAT, 
    r_hamstring FLOAT, 
    l_thigh FLOAT, 
    l_hamstring FLOAT, 
    is_aggressive INT,
    activity_id INT
);
```

### Exploratory Data Auditing and Analytics

Before initiating model training, analytical SQL queries were executed to audit data health and profile physiological baselines.

#### 1. Macro-Level Class Balance Check
```sql
SELECT is_aggressive, COUNT(*)
FROM emg
GROUP BY is_aggressive;
```
*   **Insight:** Normal rows total 48,401 (approx. 50.3%) and Aggressive rows total 47,831 (approx. 49.7%). The distribution is balanced, protecting the model from training bias.

#### 2. Micro-Level Action Distribution Check
```sql
SELECT activity_id, COUNT(*) 
FROM emg
GROUP BY activity_id;
```
*   **Insight:** Each of the 10 activities contains approximately 9,600 rows of data, confirming that the experimental trials were executed with equal duration and sampling control.

#### 3. Neuromuscular Average Amplitude Profiling
```sql
SELECT 
    is_aggressive,
    ROUND(AVG(ABS(r_bicep)), 4) AS avg_r_bicep,
    ROUND(AVG(ABS(r_tricep)), 4) AS avg_r_tricep,
    ROUND(AVG(ABS(l_bicep)), 4) AS avg_l_bicep,
    ROUND(AVG(ABS(l_tricep)), 4) AS avg_l_tricep,
    ROUND(AVG(ABS(r_thigh)), 4) AS avg_r_thigh,
    ROUND(AVG(ABS(r_hamstring)), 4) AS avg_r_hamstring,
    ROUND(AVG(ABS(l_thigh)), 4) AS avg_l_thigh,
    ROUND(AVG(ABS(l_hamstring)), 4) AS avg_l_hamstring
FROM emg
GROUP BY is_aggressive;
```
*   **Insight:** Taking the absolute averages (Mean Absolute Value / MAV) [1.2.6] reveals that aggressive trials (value 1) exhibit substantially higher average amplitudes across upper-body and lower-body sensors than normal trials (value 0), validating that muscle activation intensity is highly discriminative of behavior.

---

## Step 3 - Biosignal Processing and Machine Learning

The machine learning script model.py applies standard biosignal pre-processing techniques to the extracted data before initiating training.

### 1. Preprocessing and Signal Conditioning
*   **DC Offset Removal
*   **Butterworth Band-Pass Filter (20 Hz to 450 Hz)
*   **Feature Extraction (Sliding Windows):** The continuous signal is segmented into 250ms windows with 50% overlap. For each window, four time-domain features are extracted per muscle to construct a 32-dimensional feature space for classification:
    *   **RMS (Root Mean Square)
    *   **MAV (Mean Absolute Value)
    *   **VAR (Variance)
    *   **SD (Standard Deviation)

### 2. Classifier Configuration
The model uses an optimized XGBoost Classifier with parameters tuned to restrict overfitting:
*   max_depth=4: Restricts tree complexity.
*   learning_rate=0.05: Keeps learning increments stable.
*   reg_lambda=1.5: Applies L2 regularization to weight changes.
*   early_stopping_rounds=15: Halts training as soon as testing loss (mlogloss) fails to improve.

---

## Project Directory Structure

```text
├── .gitignore               # Excludes massive raw data files from git tracking
├── README.md                # Project documentation and summary
├── merge_emg_data.py        # Python ETL script used to consolidate separate files
├── schema.sql               # MySQL script containing database and table schemas
├── queries.sql              # Auditing and analytical SQL queries
├── train_model.py           # Main ML script (Data loading, filtering, feature extraction, training)
└── visualization.py         # Plotting file containing modular visualization functions
```
