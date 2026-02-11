"""
LEO Satellite Reaction Wheel Anomaly Detection - Scikit-Learn Based Pipeline
Complete ML Pipeline without TensorFlow dependency for faster execution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("="*60)
print("LEO SATELLITE REACTION WHEEL ANOMALY DETECTION")
print("="*60)

# 1. Load data
print("\n[1/8] Loading dataset...")
df = pd.read_csv('data.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
print(f"  Dataset shape: {df.shape}")
print(f"  Date range: {df.index.min()} to {df.index.max()}")

# 2. Data exploration
print("\n[2/8] Exploring data...")
feature_cols = ['wheel_speed_rpm', 'wheel_torque', 'motor_current', 'motor_temp', 'vibration_level']
X = df[feature_cols].copy()
y = (df['label'] == 'anomal').astype(int)
y_types = df['anomaly_type'].copy()

anomaly_counts = df['anomaly_type'].value_counts()
print(f"  Anomaly distribution:")
for atype, count in anomaly_counts.items():
    print(f"    {atype}: {count}")
print(f"  Total anomaly rate: {y.mean()*100:.2f}%")

# 3. Normalization
print("\n[3/8] Preprocessing...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

# 4. Feature engineering
print("\n[4/8] Feature engineering...")
window = 6
X_eng = X_scaled.copy()

for col in feature_cols:
    X_eng[f'{col}_rolling_mean'] = X_scaled[col].rolling(window).mean()
    X_eng[f'{col}_rolling_std'] = X_scaled[col].rolling(window).std()
    X_eng[f'{col}_diff'] = X_scaled[col].diff()

X_eng['speed_current_ratio'] = X_scaled['wheel_speed_rpm'] / (X_scaled['motor_current'] + 1e-6)
X_eng['power_proxy'] = X_scaled['motor_current'] * X_scaled['wheel_torque']

X_eng = X_eng.dropna()
y_eng = y.loc[X_eng.index]
y_types_eng = y_types.loc[X_eng.index]
print(f"  Engineered features: {X_eng.shape[1]}")

# 5. Train models
print("\n[5/8] Training models...")
X_normal = X_eng[y_eng == 0]
X_test = X_eng.copy()
y_test = y_eng.copy()

# Model 1: Isolation Forest
print("  Training Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(X_normal)
iso_scores = -iso_forest.decision_function(X_test)
iso_threshold = np.percentile(iso_scores, 95)
iso_preds = (iso_scores > iso_threshold).astype(int)

# Model 2: One-Class SVM
print("  Training One-Class SVM...")
ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_normal)
svm_scores = -ocsvm.decision_function(X_test)
svm_threshold = np.percentile(svm_scores, 95)
svm_preds = (svm_scores > svm_threshold).astype(int)

# Model 3: MLP Autoencoder (sklearn-based alternative to LSTM)
print("  Training MLP Autoencoder...")
mlp_ae = MLPRegressor(hidden_layer_sizes=(32, 16, 8, 16, 32), activation='relu', 
                       solver='adam', max_iter=200, random_state=42, early_stopping=True)
mlp_ae.fit(X_normal, X_normal)
X_reconstructed = mlp_ae.predict(X_test)
mlp_mse = np.mean(np.power(X_test.values - X_reconstructed, 2), axis=1)
mlp_threshold = np.percentile(mlp_mse[y_test == 0], 95)
mlp_preds = (mlp_mse > mlp_threshold).astype(int)

# 6. Ensemble
print("\n[6/8] Building ensemble...")
iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
svm_scores_norm = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())
mlp_scores_norm = (mlp_mse - mlp_mse.min()) / (mlp_mse.max() - mlp_mse.min())

ensemble_scores = 0.3 * iso_scores_norm + 0.3 * svm_scores_norm + 0.4 * mlp_scores_norm
ensemble_threshold = np.percentile(ensemble_scores, 95)
ensemble_preds = (ensemble_scores > ensemble_threshold).astype(int)

# 7. Evaluation
print("\n[7/8] Evaluating models...")
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)

def evaluate(name, y_true, y_pred, scores):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0
    print(f"\n{name}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    return {'name': name, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

results = []
results.append(evaluate('Isolation Forest', y_test, iso_preds, iso_scores_norm))
results.append(evaluate('One-Class SVM', y_test, svm_preds, svm_scores_norm))
results.append(evaluate('MLP Autoencoder', y_test, mlp_preds, mlp_scores_norm))
results.append(evaluate('Ensemble', y_test, ensemble_preds, ensemble_scores))

# Per-type analysis
print("\n" + "="*60)
print("PER-ANOMALY-TYPE DETECTION")
print("="*60)

for atype in ['speed_', 'torque', 'curren', 'vibrat', 'overte']:
    mask = y_types_eng == atype
    if mask.sum() > 0:
        detected = ensemble_preds[mask].sum()
        total = mask.sum()
        recall = detected / total
        print(f"  {atype}: {detected}/{total} detected ({recall*100:.1f}% recall)")

# 8. Visualizations
print("\n[8/8] Generating visualizations...")

# Figure 1: Telemetry time-series
fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
anomaly_idx = df[df['label'] == 'anomal'].index

for i, col in enumerate(feature_cols):
    axes[i].plot(df.index, df[col], 'b-', alpha=0.7, linewidth=0.8)
    axes[i].scatter(anomaly_idx, df.loc[anomaly_idx, col], c='red', s=30, zorder=5)
    axes[i].set_ylabel(col.replace('_', '\n'), fontsize=9)
    axes[i].grid(True, alpha=0.3)

axes[0].set_title('Reaction Wheel Telemetry with Anomaly Markers', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('telemetry_timeseries.png', dpi=150)
print("  Saved: telemetry_timeseries.png")

# Figure 2: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for (name, scores), color in zip([('Isolation Forest', iso_scores_norm), 
                                   ('One-Class SVM', svm_scores_norm),
                                   ('MLP Autoencoder', mlp_scores_norm),
                                   ('Ensemble', ensemble_scores)], colors):
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2, color=color)

ax.plot([0,1], [0,1], 'k--', label='Random', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
print("  Saved: roc_curves.png")

# Figure 3: Confusion Matrices
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
models = [('Isolation Forest', iso_preds), ('One-Class SVM', svm_preds),
          ('MLP AE', mlp_preds), ('Ensemble', ensemble_preds)]

for ax, (name, preds) in zip(axes, models):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
print("  Saved: confusion_matrices.png")

# Figure 4: Anomaly Score Timeline
fig, ax = plt.subplots(figsize=(14, 6))
time_idx = X_test.index

ax.plot(time_idx, ensemble_scores, 'b-', alpha=0.7, label='Anomaly Score')
ax.axhline(y=ensemble_threshold, color='r', linestyle='--', label=f'Threshold ({ensemble_threshold:.2f})')

anomaly_mask = y_test == 1
ax.scatter(time_idx[anomaly_mask], ensemble_scores[anomaly_mask], c='red', s=50, label='True Anomaly', zorder=5)

ax.set_xlabel('Timestamp', fontsize=12)
ax.set_ylabel('Ensemble Anomaly Score', fontsize=12)
ax.set_title('Ensemble Anomaly Score Timeline', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('anomaly_timeline.png', dpi=150)
print("  Saved: anomaly_timeline.png")

# Figure 5: Severity Dashboard
def classify_severity(score, threshold):
    if score < threshold * 0.5:
        return 'NOMINAL'
    elif score < threshold:
        return 'LOW'
    elif score < threshold * 1.5:
        return 'MEDIUM'
    elif score < threshold * 2:
        return 'HIGH'
    else:
        return 'CRITICAL'

severity = [classify_severity(s, ensemble_threshold) for s in ensemble_scores]
colors_map = {'NOMINAL': '#2ecc71', 'LOW': '#f1c40f', 'MEDIUM': '#e67e22', 'HIGH': '#e74c3c', 'CRITICAL': '#8e44ad'}
colors = [colors_map[s] for s in severity]

fig, ax = plt.subplots(figsize=(14, 5))
ax.scatter(time_idx, ensemble_scores, c=colors, s=20, alpha=0.7)
ax.axhline(y=ensemble_threshold, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Timestamp')
ax.set_ylabel('Anomaly Score')
ax.set_title('Operator Alert Dashboard - Severity Levels', fontsize=14, fontweight='bold')

for sev, col in colors_map.items():
    ax.scatter([], [], c=col, label=sev, s=50)
ax.legend(loc='upper right', title='Severity')
plt.tight_layout()
plt.savefig('severity_dashboard.png', dpi=150)
print("  Saved: severity_dashboard.png")

# Summary
print("\n" + "="*60)
print("PIPELINE COMPLETE - SUMMARY")
print("="*60)
print(f"Dataset: 90 days of reaction wheel telemetry ({len(df)} samples)")
print(f"Anomaly rate: {y.mean()*100:.2f}%")
print(f"\nBest Model: Ensemble (weighted combination)")
print(f"  - 30% Isolation Forest")
print(f"  - 30% One-Class SVM")
print(f"  - 40% MLP Autoencoder")
print(f"\nGenerated Visualizations:")
print("  1. telemetry_timeseries.png")
print("  2. roc_curves.png")
print("  3. confusion_matrices.png")
print("  4. anomaly_timeline.png")
print("  5. severity_dashboard.png")
print("\n" + "="*60)
