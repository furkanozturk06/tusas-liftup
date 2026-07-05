
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix)

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

class ModelEvaluator:
    def __init__(self, models_dir: str = '../models/'):
        self.models_dir = models_dir
        self.models = {}
        self.metrics = {}
        self.inference_times = {}
        self.model_sizes = {}
        self.predictions = {}
        self.probabilities = {}
        
    def load_models(self, supervised_list: List[str], unsupervised_list: List[str]):
        print("Modeller yükleniyor...")
        
        for name in supervised_list:
            path_joblib = os.path.join(self.models_dir, f"{name.lower()}_model.joblib")
            path_keras = os.path.join(self.models_dir, f"{name.lower()}_model.keras")
            
            if os.path.exists(path_joblib):
                self.models[name] = joblib.load(path_joblib)
                self.model_sizes[name] = os.path.getsize(path_joblib) / (1024 * 1024)
            elif os.path.exists(path_keras) and load_model:
                self.models[name] = load_model(path_keras)
                self.model_sizes[name] = os.path.getsize(path_keras) / (1024 * 1024)
                
        unsup_dir = os.path.join(self.models_dir, 'unsupervised')
        for name in unsupervised_list:
            path_joblib = os.path.join(unsup_dir, f"{name.lower()}_model.joblib")
            path_keras = os.path.join(unsup_dir, f"{name.lower()}_model.keras")
            
            if os.path.exists(path_joblib):
                self.models[name] = joblib.load(path_joblib)
                self.model_sizes[name] = os.path.getsize(path_joblib) / (1024 * 1024)
            elif os.path.exists(path_keras) and load_model:
                self.models[name] = load_model(path_keras)
                self.model_sizes[name] = os.path.getsize(path_keras) / (1024 * 1024)

        thresh_path = os.path.join(unsup_dir, 'unsupervised_thresholds.json')
        if os.path.exists(thresh_path):
            with open(thresh_path, 'r', encoding='utf-8') as f:
                self.unsup_thresholds = json.load(f)
        else:
            self.unsup_thresholds = {}
            
        print(f"Yüklenen Modeller: {list(self.models.keys())}")

    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Test seti üzerinde metrikler ve çıkarım (inference) hızları hesaplanıyor...")
        
        for name, model in self.models.items():
            start_time = time.time()
            
            if name == 'MLP' or name == 'Autoencoder':
                preds_raw = model.predict(X_test, verbose=0)
                if name == 'MLP':
                    prob = preds_raw.flatten()
                    pred = (prob >= 0.5).astype(int)
                else:
                    prob = np.mean(np.power(X_test - preds_raw, 2), axis=1)
                    thresh = self.unsup_thresholds.get('Autoencoder', np.mean(prob))
                    pred = (prob > thresh).astype(int)
            elif name in ['IsolationForest', 'LOF']:
                prob = -model.score_samples(X_test)
                thresh = self.unsup_thresholds.get(name, np.mean(prob))
                pred = (prob > thresh).astype(int)
            elif name == 'OneClassSVM':
                prob = -model.decision_function(X_test)
                thresh = self.unsup_thresholds.get(name, np.mean(prob))
                pred = (prob > thresh).astype(int)
            elif name == 'KMeans':
                dist = model.transform(X_test)
                prob = np.min(dist, axis=1)
                thresh = self.unsup_thresholds.get(name, np.mean(prob))
                pred = (prob > thresh).astype(int)
            else:
                pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_test)[:, 1]
                else:
                    prob = pred
            
            inf_time = (time.time() - start_time) * 1000
            
            self.predictions[name] = pred
            self.probabilities[name] = prob
            self.inference_times[name] = inf_time / len(X_test)
            
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, prob)
            except:
                auc = 0.5
                
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            self.metrics[name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'AUC-ROC': auc,
                'FAR': far,
                'FNR': fnr,
                'Inf.Time(ms)': self.inference_times[name],
                'Model Size(MB)': self.model_sizes.get(name, 0)
            }

    def generate_comparison_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.metrics).T
        return df

    def plot_roc_curves(self, y_test: np.ndarray, save_path: str = None):
        plt.figure(figsize=(12, 10))
        for name, prob in self.probabilities.items():
            try:
                fpr, tpr, _ = roc_curve(y_test, prob)
                auc_val = roc_auc_score(y_test, prob)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
            except:
                continue
                
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Yanlış Alarm)')
        plt.ylabel('True Positive Rate (Yakalanan Anomali)')
        plt.title('Tüm Modellerin ROC Eğrileri Karşılaştırması')
        plt.legend(loc="lower right")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_pr_curves(self, y_test: np.ndarray, save_path: str = None):
        plt.figure(figsize=(12, 10))
        for name, prob in self.probabilities.items():
            try:
                prec, rec, _ = precision_recall_curve(y_test, prob)
                plt.plot(rec, prec, lw=2, label=name)
            except:
                continue
                
        plt.xlabel('Recall (Yakalanan Anomali)')
        plt.ylabel('Precision (Tahmin Doğruluğu)')
        plt.title('Tüm Modellerin Precision-Recall Eğrileri')
        plt.legend(loc="lower left")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_anomaly_timeline(self, df_raw: pd.DataFrame, y_true: np.ndarray, sample_size: int = 500):
        df_sub = df_raw.head(sample_size).copy()
        y_sub = y_true[:sample_size]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            subplot_titles=('Telemetri Sinyali (Orijinal)', 'Modellerin Anomali Skorları'))

        if 'value' in df_sub.columns:
            fig.add_trace(go.Scatter(y=df_sub['value'], mode='lines', name='Ham Sinyal', line=dict(color='white')), row=1, col=1)
            
        anomaly_indices = np.where(y_sub == 1)[0]
        for idx in anomaly_indices:
            fig.add_vline(x=idx, line_width=1, line_dash="dot", line_color="red", opacity=0.3, row=1, col=1)

        best_models = ['MLP', 'XGBoost', 'Autoencoder']
        colors = ['cyan', 'orange', 'lime']
        
        for idx, m in enumerate(best_models):
            if m in self.probabilities:
                prob_sub = self.probabilities[m][:sample_size]
                prob_sub = (prob_sub - np.min(prob_sub)) / (np.max(prob_sub) - np.min(prob_sub) + 1e-10)
                fig.add_trace(go.Scatter(y=prob_sub, mode='lines', name=f'{m} Skoru', line=dict(color=colors[idx])), row=2, col=1)

        fig.update_layout(height=700, title_text="Gerçek Zamanlı Anomali Tespiti Operatör Paneli", template="plotly_dark")
        fig.show()

    def export_metrics(self, path_csv: str, path_json: str):
        os.makedirs(os.path.dirname(path_csv), exist_ok=True)
        df = self.generate_comparison_table()
        df.to_csv(path_csv)
        
        with open(path_json, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrikler kaydedildi: {path_csv}")
