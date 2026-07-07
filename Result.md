# EMG Physical Action Classification - Model Results

This document summarizes the signal characteristics, training dynamics, and classification performance of the XGBoost model trained on 8-channel raw EMG data.

---

## 1. Raw EMG Waveforms
This plot shows the raw, band-pass filtered electrical activity for all 8 muscle channels simultaneously over a 5-second interval.

![Raw EMG Waveforms](images/raw_waveforms.png)

---

## 2. EMG Amplitude Density Distribution
This probability density plot maps the distribution of raw voltage values for each muscle channel over the entire duration of the trials.

![EMG Amplitude Density Distribution](images/amplitude_distribution.png)

* **Comparative Activity:** Channel distributions with tall, narrow peaks (such as the triceps in orange and red) indicate muscles that remained largely inactive or resting during the gestures. Channels with short, highly spread-out curves (such as the thighs and biceps in purple and blue) represent muscles that actively fired and sustained high-voltage outputs.

---

## 3. PCA Feature Space Projection (2D)
This plot uses Principal Component Analysis (PCA) to project the high-dimensional muscle data into a simplified 2D space.

![PCA Feature Space Projection](images/pca_projection.png)

### Explanation:
* **91.6% Information Capture:** Principal Component 1 (PC1) and Principal Component 2 (PC2) combine to explain 91.6% of the total variance in the data, showing that a 2D projection is highly representative of the overall feature space.
* **PC1 (82.3% Variance - Overall Muscle Intensity):** Separates actions horizontally based on contraction power. Low-energy or resting movements (Classes 0, 1, 2, 3, 8) form tight, dense clusters on the far left. High-energy, forceful movements (Classes 4, 5, 6, 7, 9) are pushed far to the right and exhibit wider trial-to-trial variance.
* **PC2 (9.3% Variance - Muscle Coordination):** Groups active movements vertically. For example, PC2 separates the upper-right neighborhood (Classes 5 and 7) from the lower-right neighborhood (Classes 4, 6, and 9) based on which specific muscles are contracted.

---

## 4. Action Classification Confusion Matrix Heatmap
This grid evaluates model predictions on unseen test data, mapping True Actions (Rows) against Predicted Actions (Columns).

![Action Classification Confusion Matrix](images/confusion_matrix.png)

### Explanation:
* **High Accuracy Diagonal:** The strong dark blue diagonal line represents correct classifications, yielding an overall classification accuracy of approximately 89.6% (69 out of 77 correct predictions).

---

## 5. XGBoost Muscle Feature Importance
This chart ranks all 32 extracted features based on their Gini Importance (how heavily the XGBoost model relied on them to make split decisions).

![XGBoost Muscle Feature Importance](images/feature_importance.png)

### Explanation:
* **Left Bicep Dominance (`l_bicep_SD`):** The standard deviation of the left bicep is the most critical feature (accounting for ~13% of all decision power). It acts as the model's primary classification "switch."
* **Postural Bracing Roles:** Right leg features (`r_thigh` and `r_hamstring`) make up three of the top five features. This indicates that whole-body posture adjustments and leg bracing during forcefully executed actions are highly correlated with the gestures.
* **Fluctuations over Averages:** Features measuring signal power and fluctuations (Standard Deviation, Variance, and RMS) heavily dominate the top rankings, whereas simple average contraction levels (MAV) sit much lower.

---

## 6. XGBoost Training Dynamics (Loss Curve)
This learning curve monitors the Multi-Class Log Loss (`mlogloss`) of both the training set and the validation set across 150 boosting iterations.

![XGBoost Training Loss Curve](images/loss_curve.png)

### Explanation:
* **Prevention of Overfitting:** By lowering the learning rate to `0.05` and adding `early_stopping_rounds=15`, the training loss (blue) and validation loss (orange) decreased.
* **No Upward Drift:** The validation curve continuously decreases and stabilizes without ever curling back upward, proving that the model generalizes robustly and is not memorizing noise.
* **Convergence:** The flattening of the curves around iteration 130–150 indicates that the model has reached near-optimal convergence, and further training would yield diminishing returns.