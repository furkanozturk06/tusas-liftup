
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix,
)

BENCHMARK_METRICS = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC_ROC", "AUC_PR"]

PRIMARY_SORT_METRIC = "AUC_PR"


def compute_metrics(y_true, y_pred, y_score=None, inf_time_ms=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    m = {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        y_score = np.asarray(y_score, dtype=float)
        try:
            m["AUC_ROC"] = round(roc_auc_score(y_true, y_score), 4)
        except Exception:
            m["AUC_ROC"] = float("nan")
        try:
            m["AUC_PR"] = round(average_precision_score(y_true, y_score), 4)
        except Exception:
            m["AUC_PR"] = float("nan")
    else:
        m["AUC_ROC"] = float("nan")
        m["AUC_PR"] = float("nan")

    m["FAR"] = round(far, 4)
    m["FNR"] = round(fnr, 4)
    if inf_time_ms is not None:
        m["Inf.Time(ms)"] = round(float(inf_time_ms), 4)

    return m


def metrics_table(all_metrics, sort_by=PRIMARY_SORT_METRIC, ascending=False):
    import pandas as pd
    df = pd.DataFrame(all_metrics).T
    extra = [c for c in df.columns if c not in BENCHMARK_METRICS]
    ordered = [c for c in BENCHMARK_METRICS if c in df.columns] + extra
    df = df[ordered]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")
    return df


def format_metrics_line(name, m):
    def g(k):
        v = m.get(k, float("nan"))
        return f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else "  nan"
    return (f"  {name:28s}  AUC_PR={g('AUC_PR')}  F1={g('F1')}  "
            f"MCC={g('MCC')}  AUC_ROC={g('AUC_ROC')}")
