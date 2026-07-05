import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import pandas as pd
import joblib

from utils.model_loader import load_all, load_metrics, predict
from core.constants import DEMO_PATH, LIVE_DATA_PATH, SHAP_PKL, PRIMARY_METRIC

MODELS, THRESHOLDS, SCALER, TEST_DATA = load_all()

ALL_METRICS = load_metrics()

FEATURE_COLS = TEST_DATA.get("feature_cols", None) if TEST_DATA else None

try:
    LIVE_DATA = pd.read_csv(LIVE_DATA_PATH)
except Exception:
    LIVE_DATA = pd.DataFrame()

try:
    SHAP_DATA = joblib.load(SHAP_PKL) if os.path.exists(SHAP_PKL) else None
except Exception:
    SHAP_DATA = None


_SHAP_EXPLAINERS = {}
def get_tree_explainer(model):
    key = id(model)
    if key not in _SHAP_EXPLAINERS:
        import shap
        _SHAP_EXPLAINERS[key] = shap.TreeExplainer(model)
    return _SHAP_EXPLAINERS[key]


def best_model(metric=PRIMARY_METRIC, among=None):
    pool = {n: v for n, v in ALL_METRICS.items() if among is None or n in among}
    if not pool:
        return None, {}
    name = max(pool, key=lambda n: pool[n].get(metric, 0))
    return name, pool[name]
