import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


# Raw Signal
def plot_raw_waveforms(time, emg_data, channel_names=None):
    
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(8)]
        
    # Create 8 subplots sharing the X-axis
    fig, axes = plt.subplots(8, 1, figsize=(12, 10), sharex=True)
    
    for i in range(8):
        axes[i].plot(time, emg_data[:, i], color='tab:blue', linewidth=0.8)
        axes[i].set_ylabel("Voltage (V)", fontsize=9)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
        # Label each channel inside its subplot
        axes[i].text(0.01, 0.85, channel_names[i], transform=axes[i].transAxes, 
                     fontsize=10, weight='bold', verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Keep y-limits symmetrical around 0 for easier visual comparison
        max_val = np.max(np.abs(emg_data[:, i]))
        axes[i].set_ylim(-max_val * 1.1, max_val * 1.1)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.suptitle("Raw EMG Waveform (8 Channels)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

# Density
def plot_amplitude_distribution(emg_data, channel_names=None):
    
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(8)]
        
    plt.figure(figsize=(10, 6))
    
    # Plot smooth Kernel Density Estimate (KDE) for each channel
    for i in range(8):
        sns.kdeplot(emg_data[:, i], label=channel_names[i], alpha=0.7, linewidth=1.5)
        
    plt.axvline(0, color='black', linestyle=':', alpha=0.5, label="Zero Voltage")
    plt.title("EMG Amplitude Density Distribution", fontsize=14)
    plt.xlabel("Voltage (μV)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Feature importance
def plot_feature_importance(model, feature_names, output_filename="feature_importance.png"):
    """
    Extracts feature importances from a trained XGBoost model 
    and saves a ranked bar chart.
    """
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_names = np.array(feature_names)[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_indices], y=sorted_names, palette="viridis", hue=sorted_names, legend=False)
    plt.title("XGBoost - Muscle Feature Importance", fontsize=14, pad=15)
    plt.xlabel("Importance Score (Gini Importance)", fontsize=12)
    plt.ylabel("Muscle Channel", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved: {output_filename}")

# Loss curve
def plot_loss_curve(model, output_filename="loss_curve.png"):
    """
    Extracts the log loss history recorded during model training
    for both train and test sets, and saves a learning curve plot.
    """
    # Retrieve the recorded loss metrics
    evals_result = model.evals_result()
    
    # Extract the training loss and testing loss lists
    metric_key = 'mlogloss' if 'mlogloss' in evals_result['validation_0'] else 'logloss'
    train_loss = evals_result['validation_0'][metric_key]
    test_loss = evals_result['validation_1'][metric_key]
    epochs = len(train_loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss, label="Training Loss", color="royalblue", lw=2)
    plt.plot(range(1, epochs + 1), test_loss, label="Validation Loss", color="darkorange", lw=2, linestyle="--")
    
    plt.xlabel("Boosting Round (Iteration)", fontsize=12)
    plt.ylabel("Log Loss (Cross-Entropy)", fontsize=12)
    plt.title("XGBoost Training Dynamics - Loss Curve", fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved: {output_filename}")

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    # Compute the raw confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Compute the normalized matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) # Handle division by zero if a class has no samples
    
    # Create text labels showing both percentage and raw count: "85.0% (17)"
    labels = [
        f"{pct:.1%}\n({count})" 
        for pct, count in zip(cm_normalized.flatten(), cm.flatten())
    ]
    labels = np.asarray(labels).reshape(cm.shape)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        xticklabels=classes, 
        yticklabels=classes,
        cbar_kws={'label': 'Proportion of Correct Predictions'}
    )
    
    plt.title("Action Classification Confusion Matrix", fontsize=14, pad=15)
    plt.ylabel("True Actions (Ground Truth)", fontsize=11)
    plt.xlabel("Predicted Actions", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# PCA projection
def plot_pca_projection(X, y, n_components=2):
    
    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3.")
        
    # Fit and transform the data using PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Get the proportion of variance explained by each principal component
    explained_variance = pca.explained_variance_ratio_ * 100
    
    fig = plt.figure(figsize=(9, 7))
    unique_labels = np.unique(y)
    
    if n_components == 2:
        ax = fig.add_subplot(111)
        for label in unique_labels:
            mask = (y == label)
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1], 
                label=label, alpha=0.7, edgecolors='w', s=60
            )
        ax.set_xlabel(f"Muscle Activation Intensity ({explained_variance[0]:.1f}% Variance)")
        ax.set_ylabel(f"Muscle Coordination Pattern ({explained_variance[1]:.1f}% Variance)")
        
    elif n_components == 3:
        # Import 3D plotting axes
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        for label in unique_labels:
            mask = (y == label)
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                label=label, alpha=0.7, edgecolors='w', s=60
            )
        ax.set_xlabel(f"PC 1 ({explained_variance[0]:.1f}%)")
        ax.set_ylabel(f"PC 2 ({explained_variance[1]:.1f}%)")
        ax.set_zlabel(f"PC 3 ({explained_variance[2]:.1f}%)")
        
    plt.title(f"PCA Feature Space Projection ({n_components}D)", fontsize=14, pad=15)
    plt.legend(title="Action Types", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()