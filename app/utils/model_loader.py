"""Model loading and prediction utilities."""
import os, json, joblib, warnings, logging
import numpy as np
import pandas as pd

log = logging.getLogger("dashboard.model_loader")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(ROOT, "models")
UNSUP_DIR = os.path.join(MODEL_DIR, "unsupervised")

SEQUENCE_MODELS = {"LSTM", "BiLSTM", "GRU", "BiGRU", "CNN1D", "CNN_LSTM", "CNN_BiLSTM",
                   "CNN_GRU", "Transformer", "TCN", "Attention_BiLSTM",
                   "FCN", "ResNet1D", "InceptionTime", "LSTM_FCN"}
PYOD_MODELS = {"ECOD", "COPOD", "HBOS", "CBLOF",
               "ABOD", "COF", "SOD", "SOS", "LODA", "INNE", "LMDD",
               "SO_GAAL", "MO_GAAL", "DeepSVDD", "LUNAR", "DIF", "XGBOD"}


def _safe_load(path):
    try:
        return joblib.load(path) if os.path.exists(path) else None
    except Exception as e:
        log.warning("Model yüklenemedi (%s): %s", os.path.basename(path), e)
        return None


def _kload(path):
    """Keras modelini güvenli şekilde yükler (VAE'nin Lambda katmanı için safe_mode=False)."""
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False, safe_mode=False)


def _keras_feature_dim(model):
    """Keras modelinin beklediği düz özellik sayısını döndürür.
    Dense/MLP -> (None, F) => F;  sıralı -> (None, T, 1) => T*1 = T. Bilinmezse None."""
    try:
        shp = model.input_shape
        if isinstance(shp, list):
            shp = shp[0]
        dims = [d for d in shp if d is not None]
        flat = 1
        for d in dims:
            flat *= d
        return flat
    except Exception:
        return None


def load_all():
    """Return (models_dict, thresholds_dict, scaler, test_data)."""
    models = {}
    for name, fname in [("RandomForest", "rf_model.joblib"), ("XGBoost", "xgb_model.joblib"),
                        ("SVM", "svm_model.joblib"), ("LightGBM", "lightgbm_model.joblib"),
                        ("CatBoost", "catboost_model.joblib"), ("Stacking Ensemble", "stacking_ensemble_model.joblib"),
                        ("ExtraTrees", "extratrees_model.joblib"),
                        ("GradientBoosting", "gradientboosting_model.joblib"),
                        ("HistGradientBoosting", "histgradientboosting_model.joblib"),
                        ("AdaBoost", "adaboost_model.joblib"), ("KNN", "knn_model.joblib"),
                        ("LogisticRegression", "logisticregression_model.joblib"),
                        ("DecisionTree", "decisiontree_model.joblib"),
                        ("NaiveBayes", "naivebayes_model.joblib"),
                        ("Voting Ensemble", "voting_ensemble_model.joblib"),
                        ("LDA", "lda_model.joblib"),
                        ("QDA", "qda_model.joblib"),
                        ("Bagging", "bagging_model.joblib"),
                        ("Ridge", "ridge_model.joblib"),
                        ("SGD", "sgd_model.joblib"),
                        ("LSVC", "lsvc_model.joblib"),
                        ("XGBOD", "xgbod_model.joblib"),
                        ("MLP", "mlp_sklearn_model.joblib")]:
        m = _safe_load(os.path.join(MODEL_DIR, fname))
        if m: models[name] = m

    for name, fname in [("IsolationForest", "isolationforest_model.joblib"),
                        ("OneClassSVM", "oneclasssvm_model.joblib"),
                        ("KMeans", "kmeans_model.joblib"), ("LOF", "lof_model.joblib"),
                        ("GMM", "gmm_model.joblib"), ("EllipticEnvelope", "ellipticenvelope_model.joblib"),
                        ("PCA", "pca_model.joblib"), ("DBSCAN", "dbscan_model.joblib"),
                        ("ECOD", "ecod_model.joblib"), ("COPOD", "copod_model.joblib"),
                        ("HBOS", "hbos_model.joblib"), ("CBLOF", "cblof_model.joblib"),
                        ("ABOD", "abod_model.joblib"), ("COF", "cof_model.joblib"),
                        ("SOD", "sod_model.joblib"), ("SOS", "sos_model.joblib"),
                        ("LODA", "loda_model.joblib"), ("INNE", "inne_model.joblib"),
                        ("LMDD", "lmdd_model.joblib"),
                        ("SO_GAAL", "so_gaal_model.joblib"), ("MO_GAAL", "mo_gaal_model.joblib"),
                        ("DeepSVDD", "deepsvdd_model.joblib"),
                        ("LUNAR", "lunar_model.joblib"), ("DIF", "dif_model.joblib")]:
        m = _safe_load(os.path.join(UNSUP_DIR, fname))
        if m is not None: models[name] = m

    scaler = _safe_load(os.path.join(MODEL_DIR, "scaler.joblib"))
    test_data = _safe_load(os.path.join(MODEL_DIR, "test_data.joblib"))
    n_features = len(test_data["feature_cols"]) if test_data and test_data.get("feature_cols") else None

    def _load_keras_checked(name, path):
        if not os.path.exists(path):
            return
        try:
            m = _kload(path)
        except Exception as e:
            log.warning("Keras modeli yüklenemedi (%s): %s", name, e)
            return
        fdim = _keras_feature_dim(m)
        if n_features is not None and fdim is not None and fdim != n_features:
            log.warning("'%s' atlandı: %s özellik bekliyor, kanonik %s (uyumsuz, muhtemelen eski model)",
                        name, fdim, n_features)
            return
        models[name] = m

    for name, fname in [("MLP", "mlp_model.keras"), ("LSTM", "lstm_model.keras"),
                        ("BiLSTM", "bilstm_model.keras"), ("GRU", "gru_model.keras"),
                        ("BiGRU", "bigru_model.keras"), ("CNN1D", "cnn1d_model.keras"),
                        ("CNN_LSTM", "cnn_lstm_model.keras"),
                        ("CNN_BiLSTM", "cnn_bilstm_model.keras"), ("CNN_GRU", "cnn_gru_model.keras"),
                        ("Transformer", "transformer_model.keras"), ("TCN", "tcn_model.keras"),
                        ("Attention_BiLSTM", "attention_bilstm_model.keras"),
                        ("FCN", "fcn_model.keras"), ("ResNet1D", "resnet1d_model.keras"),
                        ("InceptionTime", "inceptiontime_model.keras"), ("LSTM_FCN", "lstm_fcn_model.keras")]:
        _load_keras_checked(name, os.path.join(MODEL_DIR, fname))
    for name, fname in [("Autoencoder", "autoencoder_model.keras"), ("VAE", "vae_model.keras"),
                        ("AnoGAN", "anogan_model.keras"), ("ALAD", "alad_model.keras")]:
        _load_keras_checked(name, os.path.join(UNSUP_DIR, fname))

    thresholds = {}
    tp = os.path.join(UNSUP_DIR, "unsupervised_thresholds.json")
    if os.path.exists(tp):
        with open(tp) as f:
            thresholds = json.load(f)

    return models, thresholds, scaler, test_data


def predict(model, name, X, thresholds, threshold_mult=1.0):
    """Return (predictions, scores) for a single model."""
    if name in SEQUENCE_MODELS:
        X_seq = np.asarray(X, dtype="float32").reshape((X.shape[0], X.shape[1], 1))
        sc = model.predict(X_seq, verbose=0).flatten()
        pr = (sc >= 0.5).astype(int)
    elif name == "MLP":
        if hasattr(model, "predict_proba"):
            pr = model.predict(X)
            sc = model.predict_proba(X)[:, 1]
        else:
            sc = model.predict(X, verbose=0).flatten()
            pr = (sc >= 0.5).astype(int)
    elif name in PYOD_MODELS:
        sc = model.decision_function(X)
        pr = model.predict(X)
    elif name in ("Autoencoder", "VAE", "AnoGAN", "ALAD"):
        recon = model.predict(X, verbose=0)
        sc = np.mean(np.power(X - recon, 2), axis=1)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name == "PCA":
        recon = model.inverse_transform(model.transform(X))
        sc = np.mean(np.power(X - recon, 2), axis=1)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name in ("IsolationForest", "LOF", "GMM", "EllipticEnvelope"):
        sc = -model.score_samples(X)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name == "OneClassSVM":
        sc = -model.decision_function(X)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name == "KMeans":
        sc = np.min(model.transform(X), axis=1)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name == "DBSCAN":
        sc = model.kneighbors(X)[0].min(axis=1)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif hasattr(model, "predict_proba"):
        pr = model.predict(X)
        sc = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        pr = model.predict(X)
        sc = model.decision_function(X)
    else:
        pr = model.predict(X)
        sc = pr.astype(float)
    return pr, sc


def load_metrics():
    """Kanonik metrikleri yükler (7 metrik: Accuracy/Precision/Recall/F1/MCC/
    AUC_ROC/AUC_PR + FAR/FNR). İki kaynak birleştirilir:
      - final_comparison.json        : 42 tabular model (18 ESA özelliği)
      - deep_sequence_comparison.json: 2 derin sıralı model (CNN1D, TCN; ham sinyal)
    Toplam Ψ-ölçümlü 44 model. ESA-ADB baseline'ları (2) literatür olduğu için
    metrik içermez; ayrı listelenir.

    Not: Eski şemalı adv_metrics.json (tireli 'AUC-ROC', MCC/AUC_PR yok) bilinçli
    olarak BİRLEŞTİRİLMEZ — kanonik anahtarları bozar ve karışık şema yaratırdı.
    """
    metrics = {}
    p1 = os.path.join(ROOT, "reports", "metrics", "final_comparison.json")
    if os.path.exists(p1):
        with open(p1) as f:
            metrics.update(json.load(f))
    else:
        log.warning("final_comparison.json bulunamadı: %s", p1)
    # Ham-sinyal derin sıralı modeller (NB04 Bölüm 6 çıktısı; varsa birleştir)
    p2 = os.path.join(ROOT, "reports", "metrics", "deep_sequence_comparison.json")
    if os.path.exists(p2):
        try:
            with open(p2) as f:
                metrics.update(json.load(f))
        except Exception as e:
            log.warning("deep_sequence_comparison.json okunamadı: %s", e)
    return metrics
