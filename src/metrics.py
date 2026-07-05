"""
OPS-SAT Benchmark Metrik Modulu
================================

Ruszczak et al. (2024) "The OPS-SAT benchmark for detecting anomalies in
satellite telemetry" makalesinde tanimlanan 7 ZORUNLU kalite metrigini
hesaplar. Tum anomali tespit sonuclari bu metriklerle raporlanmali ve
karsilastirma tablolari **AUC_PR**'a gore siralanmalidir.

Zorunlu 7 metrik (hepsi maksimize edilir):
    Accuracy, Precision, Recall, F1, MCC, AUC_ROC, AUC_PR
    - MCC (Matthews Correlation Coefficient): dengesiz siniflandirmada
      tercih edilen olcut (Chicco & Jurman 2020). Aralik [-1, 1].
    - AUC_PR (Area under Precision-Recall): makalenin birincil siralama
      olcutu; sklearn'de average_precision_score'a karsilik gelir.
    - Diger metrikler [0, 1] araligindadir.

Ek (operasyonel) metrikler de saglanir: FAR (yanlis alarm orani),
FNR (kacirma orani), Inf.Time. Bunlar benchmark'in 7 zorunlu metrigine
DAHIL DEGILDIR ancak on-board operasyon tartismasi icin faydalidir.

Kullanim:
    from metrics import compute_metrics, metrics_table, BENCHMARK_METRICS

    m = compute_metrics(y_true, y_pred, y_score)   # tek model
    df = metrics_table(all_metrics)                # AUC_PR sirali tablo
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix,
)

BENCHMARK_METRICS = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC_ROC", "AUC_PR"]

PRIMARY_SORT_METRIC = "AUC_PR"


def compute_metrics(y_true, y_pred, y_score=None, inf_time_ms=None):
    """Tek bir modelin tahminleri icin 7 zorunlu metrik + ek metrikleri hesaplar.

    Args:
        y_true: Gercek etiketler (0=nominal, 1=anomali).
        y_pred: Ikili tahminler (0/1).
        y_score: Anomali olasiligi / karar skoru (AUC_ROC ve AUC_PR icin).
                 None ise AUC metrikleri NaN olur.
        inf_time_ms: Tek ornek icin cikarim suresi (ms), opsiyonel.

    Returns:
        dict — 7 zorunlu metrik + FAR, FNR (+ varsa Inf.Time). AUC metrikleri
               hesaplanamazsa NaN doner.
    """
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
    """Cok modelli metrik sozlugunu siralanmis bir DataFrame'e cevirir.

    Args:
        all_metrics: {model_adi: metrik_dict} sozlugu.
        sort_by: Siralama olcutu (varsayilan AUC_PR — makaleyle ayni).
        ascending: Artan siralama (varsayilan False = en iyi ustte).

    Returns:
        pandas.DataFrame — satirlar model, sutunlar metrikler, sort_by'a gore sirali.
    """
    import pandas as pd
    df = pd.DataFrame(all_metrics).T
    extra = [c for c in df.columns if c not in BENCHMARK_METRICS]
    ordered = [c for c in BENCHMARK_METRICS if c in df.columns] + extra
    df = df[ordered]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")
    return df


def format_metrics_line(name, m):
    """Tek satirlik ozet (konsol ciktisi icin)."""
    def g(k):
        v = m.get(k, float("nan"))
        return f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else "  nan"
    return (f"  {name:28s}  AUC_PR={g('AUC_PR')}  F1={g('F1')}  "
            f"MCC={g('MCC')}  AUC_ROC={g('AUC_ROC')}")
