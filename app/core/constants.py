import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO_PATH = os.path.join(ROOT, "data", "features", "segment_features.parquet")
LIVE_DATA_PATH = os.path.join(ROOT, "data", "raw", "segments.csv")
SHAP_PKL = os.path.join(ROOT, "models", "shap_values.pkl")

BENCHMARK_METRICS = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC_ROC", "AUC_PR"]
PRIMARY_METRIC = "AUC_PR"

DROP_COLS = ["segment", "anomaly", "train", "channel"]

SUP_MODEL_NAMES = ["RandomForest", "XGBoost", "SVM", "MLP", "LightGBM", "CatBoost", "Stacking Ensemble",
                   "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "AdaBoost", "KNN",
                   "LogisticRegression", "DecisionTree", "NaiveBayes", "Voting Ensemble",
                   "LDA", "QDA", "Bagging", "Ridge", "SGD", "LSVC", "XGBOD"]
UNSUP_MODEL_NAMES = ["IsolationForest", "OneClassSVM", "KMeans", "LOF",
                     "GMM", "EllipticEnvelope", "PCA", "DBSCAN",
                     "ECOD", "COPOD", "HBOS", "CBLOF",
                     "ABOD", "COF", "SOD", "SOS", "LODA", "INNE", "LMDD"]

DEEP_SEQ_MODELS = ["CNN1D", "TCN"]

ESA_ADB_BASELINES = [
    {"name": "Telemanom-ESA", "type": "Gözetimsiz (LSTM + dinamik eşikleme)",
     "source": "Hundman et al., 2018; ESA-ADB: Kotowski et al., 2024"},
    {"name": "DC-VAE-ESA", "type": "Gözetimsiz (genişletilmiş 1D-CNN + VAE)",
     "source": "García González et al., 2022; ESA-ADB: Kotowski et al., 2024"},
]

CANONICAL_MODEL_COUNT = (len(SUP_MODEL_NAMES) + len(UNSUP_MODEL_NAMES)
                         + len(DEEP_SEQ_MODELS) + len(ESA_ADB_BASELINES))


def model_category(name):
    if name in SUP_MODEL_NAMES:
        return "Gözetimli"
    if name in UNSUP_MODEL_NAMES:
        return "Gözetimsiz"
    if name in DEEP_SEQ_MODELS:
        return "Derin Sıralı"
    if name in {b["name"] for b in ESA_ADB_BASELINES}:
        return "ESA-ADB"
    return "Diğer"

ANALYSIS_PRESETS = {
    "hizli": {"title": "Hızlı Tarama", "icon": "mdi:flash",
              "desc": "Tek hafif model ile düşük maliyetli ön tarama.",
              "sup": ["HistGradientBoosting"], "unsup": [], "thr": 1.0},
    "dogru": {"title": "Yüksek Doğruluk", "icon": "mdi:bullseye-arrow",
              "desc": "En iyi modellerin topluluğu, en güvenilir tespit.",
              "sup": ["ExtraTrees", "Voting Ensemble", "MLP"], "unsup": [], "thr": 1.0},
    "dusuk_alarm": {"title": "Düşük Yanlış Alarm", "icon": "mdi:shield-check-outline",
                    "desc": "Yüksek kesinlikli modeller ve sıkı eşik; yanlış alarmı en aza indirir.",
                    "sup": ["Stacking Ensemble", "Voting Ensemble"], "unsup": [], "thr": 1.15},
}

CHANNEL_NAMES = {
    "CADC0872": "Manyetometre X", "CADC0873": "Manyetometre Y", "CADC0874": "Manyetometre Z",
    "CADC0884": "Foto Diyot 1", "CADC0886": "Foto Diyot 2", "CADC0888": "Foto Diyot 3",
    "CADC0890": "Foto Diyot 4", "CADC0892": "Foto Diyot 5", "CADC0894": "Foto Diyot 6",
}


def channel_label(code, with_code=True):
    name = CHANNEL_NAMES.get(str(code))
    if not name:
        return str(code)
    return f"{name} ({code})" if with_code else name
