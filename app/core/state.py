"""Uygulama çalışma-zamanı durumu (module-level Singleton service).

Eğitilmiş modeller, ölçekleyici, resmi test verisi, kanonik metrikler ve
canlı/SHAP verisi tek noktadan, bir kez yüklenir. Tüm sayfa modülleri bu durumu
buradan tüketir; böylece app.py'de dağınık duran global state ortadan kalkar.
"""
import os
import sys

# TensorFlow ortam ayarları — model_loader (ve dolayısıyla TF) import edilmeden ÖNCE.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# app/ dizinini path'e ekle (utils.* ve core.* import'lari icin) — bagimsiz import'ta da calissin.
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import pandas as pd
import joblib

from utils.model_loader import load_all, load_metrics, predict  # noqa: F401  (predict re-export)
from core.constants import DEMO_PATH, LIVE_DATA_PATH, SHAP_PKL, PRIMARY_METRIC

# ── Modeller, ölçekleyici, resmi Ψ test verisi, eşikler ──
MODELS, THRESHOLDS, SCALER, TEST_DATA = load_all()

# ── Kanonik metrikler (reports/metrics/final_comparison.json) ──
ALL_METRICS = load_metrics()

# ── Kanonik 18 özellik adı (test_data.joblib'den) ──
FEATURE_COLS = TEST_DATA.get("feature_cols", None) if TEST_DATA else None

# ── Canlı izleme için ham telemetri akışı ──
try:
    LIVE_DATA = pd.read_csv(LIVE_DATA_PATH)
except Exception:
    LIVE_DATA = pd.DataFrame()

# ── SHAP analiz verisi (varsa) ──
try:
    SHAP_DATA = joblib.load(SHAP_PKL) if os.path.exists(SHAP_PKL) else None
except Exception:
    SHAP_DATA = None


_SHAP_EXPLAINERS = {}
def get_tree_explainer(model):
    """TreeExplainer'ı model başına bir kez kurup önbelleğe alır (her tıklamada
    yeniden kurmamak için — detay sayfasında belirgin hızlanma sağlar)."""
    key = id(model)
    if key not in _SHAP_EXPLAINERS:
        import shap
        _SHAP_EXPLAINERS[key] = shap.TreeExplainer(model)
    return _SHAP_EXPLAINERS[key]


def best_model(metric=PRIMARY_METRIC, among=None):
    """ALL_METRICS içinde verilen metriğe göre en iyi modeli (ad, metrik_dict) döndürür."""
    pool = {n: v for n, v in ALL_METRICS.items() if among is None or n in among}
    if not pool:
        return None, {}
    name = max(pool, key=lambda n: pool[n].get(metric, 0))
    return name, pool[name]
