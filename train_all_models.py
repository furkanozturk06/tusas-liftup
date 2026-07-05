
import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from metrics import compute_metrics, metrics_table, PRIMARY_SORT_METRIC
MODEL_DIR = os.path.join(ROOT, "models")
UNSUP_DIR = os.path.join(MODEL_DIR, "unsupervised")
METRICS_DIR = os.path.join(ROOT, "reports", "metrics")
FEATURES_DIR = os.path.join(ROOT, "data", "features")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UNSUP_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

print("=" * 60)
print("  UYDU TELEMETRI - MODEL EGITIM PIPELINE")
print("=" * 60)

df = pd.read_csv(os.path.join(ROOT, "data", "raw", "dataset.csv"))
print(f"\nDataset: {df.shape[0]} segment, {df.shape[1]} sutun")

ESA_18_FEATURES = [
    "mean", "var", "std", "kurtosis", "skew", "n_peaks",
    "duration", "len", "gaps_squared", "len_weighted",
    "var_div_duration", "var_div_len",
    "smooth10_n_peaks", "smooth20_n_peaks",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
]
FEATURE_COLS = [c for c in ESA_18_FEATURES if c in df.columns]
assert len(FEATURE_COLS) == 18, f"18 ESA ozelligi beklenirken {len(FEATURE_COLS)} bulundu: {FEATURE_COLS}"
print(f"Kanonik ozellik sayisi: {len(FEATURE_COLS)} (resmi ESA 18)")
print(f"Features: {FEATURE_COLS}")

train_df = df[df["train"] == 1].copy()
test_df = df[df["train"] == 0].copy()
print(f"Train: {len(train_df)} ({train_df['anomaly'].sum()} anomali)")
print(f"Test:  {len(test_df)} ({test_df['anomaly'].sum()} anomali)")

X_train = train_df[FEATURE_COLS].fillna(0).values
y_train = train_df["anomaly"].values
X_test = test_df[FEATURE_COLS].fillna(0).values
y_test = test_df["anomaly"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump({
    "X_test": X_test_s,
    "y_test": y_test,
    "feature_cols": FEATURE_COLS,
}, os.path.join(MODEL_DIR, "test_data.joblib"))

df[FEATURE_COLS + ["segment", "anomaly", "train", "channel"]].to_parquet(
    os.path.join(FEATURES_DIR, "segment_features.parquet"), index=False)
print("Scaler, test_data, segment_features.parquet kaydedildi.\n")

ALL_METRICS = {}

def calc_metrics(name, y_true, y_pred, y_score=None, inf_time_ms=None):
    m = compute_metrics(y_true, y_pred, y_score, inf_time_ms=inf_time_ms)
    ALL_METRICS[name] = m
    return m

def print_metrics(name, m):
    ap = m.get("AUC_PR", float("nan"))
    print(f"  {name:30s}  AUC_PR={ap:.3f}  F1={m['F1']:.3f}  "
          f"MCC={m['MCC']:.3f}  AUC_ROC={m['AUC_ROC']:.3f}")


print("-" * 60)
print("  GOZETIMLI MODELLER")
print("-" * 60)

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

SUP_MODELS = {}

def train_sup(name, model, fname):
    t0 = time.time()
    try:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_s)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test_s)
        else:
            y_score = y_pred.astype(float)
        m = calc_metrics(name, y_test, y_pred, y_score)
        joblib.dump(model, os.path.join(MODEL_DIR, fname))
        SUP_MODELS[name] = model
        elapsed = time.time() - t0
        print_metrics(name, m)
    except Exception as e:
        print(f"  {name:30s}  HATA: {e}")

train_sup("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42), "logisticregression_model.joblib")
train_sup("Ridge", CalibratedClassifierCV(RidgeClassifier(random_state=42), cv=3), "ridge_model.joblib")
train_sup("SGD", CalibratedClassifierCV(SGDClassifier(random_state=42, max_iter=1000), cv=3), "sgd_model.joblib")
train_sup("NaiveBayes", GaussianNB(), "naivebayes_model.joblib")
train_sup("LDA", LinearDiscriminantAnalysis(), "lda_model.joblib")
train_sup("QDA", QuadraticDiscriminantAnalysis(), "qda_model.joblib")
train_sup("DecisionTree", DecisionTreeClassifier(random_state=42, max_depth=10), "decisiontree_model.joblib")
train_sup("KNN", KNeighborsClassifier(n_neighbors=5), "knn_model.joblib")
train_sup("LSVC", CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42), cv=3), "lsvc_model.joblib")

train_sup("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), "rf_model.joblib")
train_sup("ExtraTrees", ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1), "extratrees_model.joblib")
train_sup("GradientBoosting", GradientBoostingClassifier(n_estimators=200, random_state=42), "gradientboosting_model.joblib")
train_sup("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=200, random_state=42), "histgradientboosting_model.joblib")
train_sup("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=42), "adaboost_model.joblib")
train_sup("Bagging", BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1), "bagging_model.joblib")

train_sup("SVM", SVC(probability=True, random_state=42), "svm_model.joblib")

try:
    import xgboost as xgb
    train_sup("XGBoost", xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0), "xgb_model.joblib")
except ImportError:
    print("  XGBoost                        ATLANACAK (paket yok)")

try:
    import lightgbm as lgb
    train_sup("LightGBM", lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1), "lightgbm_model.joblib")
except ImportError:
    print("  LightGBM                       ATLANACAK (paket yok)")

try:
    from catboost import CatBoostClassifier
    train_sup("CatBoost", CatBoostClassifier(iterations=200, random_state=42, verbose=0), "catboost_model.joblib")
except ImportError:
    print("  CatBoost                       ATLANACAK (paket yok)")

try:
    from pyod.models.xgbod import XGBOD
    t0 = time.time()
    xgbod = XGBOD(random_state=42)
    xgbod.fit(X_train_s, y_train)
    y_pred = xgbod.predict(X_test_s)
    y_score = xgbod.decision_function(X_test_s)
    m = calc_metrics("XGBOD", y_test, y_pred, y_score)
    joblib.dump(xgbod, os.path.join(MODEL_DIR, "xgbod_model.joblib"))
    print_metrics("XGBOD", m)
except Exception as e:
    print(f"  XGBOD                          HATA: {e}")

train_sup("MLP", MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True), "mlp_sklearn_model.joblib")

try:
    base_models = []
    for n in ["RandomForest", "ExtraTrees", "LogisticRegression"]:
        if n in SUP_MODELS:
            base_models.append((n, SUP_MODELS[n]))
    if len(base_models) >= 2:
        voting = VotingClassifier(estimators=base_models, voting="soft", n_jobs=-1)
        train_sup("Voting Ensemble", voting, "voting_ensemble_model.joblib")
except Exception as e:
    print(f"  Voting Ensemble                HATA: {e}")

try:
    stack_estimators = []
    for n in ["RandomForest", "LogisticRegression", "KNN"]:
        if n in SUP_MODELS:
            stack_estimators.append((n, SUP_MODELS[n]))
    if len(stack_estimators) >= 2:
        stacking = StackingClassifier(estimators=stack_estimators,
                                       final_estimator=LogisticRegression(max_iter=500),
                                       cv=3, n_jobs=-1)
        train_sup("Stacking Ensemble", stacking, "stacking_ensemble_model.joblib")
except Exception as e:
    print(f"  Stacking Ensemble              HATA: {e}")


print("\n" + "─" * 60)
print("  GOVETIMSIZ MODELLER")
print("-" * 60)

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA as PCA_Model

THRESHOLDS = {}

def train_unsup_sklearn(name, model, fname, score_fn):
    t0 = time.time()
    try:
        model.fit(X_train_s)
        scores = score_fn(model, X_test_s)
        threshold = float(np.percentile(scores, 80))
        preds = (scores > threshold).astype(int)
        m = calc_metrics(name, y_test, preds, scores)
        joblib.dump(model, os.path.join(UNSUP_DIR, fname))
        THRESHOLDS[name] = threshold
        print_metrics(name, m)
    except Exception as e:
        print(f"  {name:30s}  HATA: {e}")

train_unsup_sklearn("IsolationForest",
    IsolationForest(n_estimators=200, contamination=0.2, random_state=42),
    "isolationforest_model.joblib",
    lambda m, X: -m.score_samples(X))

train_unsup_sklearn("OneClassSVM",
    OneClassSVM(kernel="rbf", nu=0.2),
    "oneclasssvm_model.joblib",
    lambda m, X: -m.decision_function(X))

train_unsup_sklearn("KMeans",
    KMeans(n_clusters=2, random_state=42, n_init=10),
    "kmeans_model.joblib",
    lambda m, X: np.min(m.transform(X), axis=1))

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.2, novelty=True)
train_unsup_sklearn("LOF", lof, "lof_model.joblib",
    lambda m, X: -m.score_samples(X))

gmm = GaussianMixture(n_components=2, random_state=42)
train_unsup_sklearn("GMM", gmm, "gmm_model.joblib",
    lambda m, X: -m.score_samples(X))

train_unsup_sklearn("EllipticEnvelope",
    EllipticEnvelope(contamination=0.2, random_state=42),
    "ellipticenvelope_model.joblib",
    lambda m, X: -m.score_samples(X))

pca_model = PCA_Model(n_components=min(10, X_train_s.shape[1]))
train_unsup_sklearn("PCA", pca_model, "pca_model.joblib",
    lambda m, X: np.mean(np.power(X - m.inverse_transform(m.transform(X)), 2), axis=1))

try:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_train_s)
    scores_db = nn.kneighbors(X_test_s)[0].mean(axis=1)
    threshold_db = float(np.percentile(scores_db, 80))
    preds_db = (scores_db > threshold_db).astype(int)
    m = calc_metrics("DBSCAN", y_test, preds_db, scores_db)
    joblib.dump(nn, os.path.join(UNSUP_DIR, "dbscan_model.joblib"))
    THRESHOLDS["DBSCAN"] = threshold_db
    print_metrics("DBSCAN", m)
except Exception as e:
    print(f"  DBSCAN                         HATA: {e}")


print("\n  -- PyOD Modelleri --")

def train_pyod(name, model_class, fname, **kwargs):
    t0 = time.time()
    try:
        model = model_class(**kwargs)
        model.fit(X_train_s)
        scores = model.decision_function(X_test_s)
        preds = model.predict(X_test_s)
        m = calc_metrics(name, y_test, preds, scores)
        joblib.dump(model, os.path.join(UNSUP_DIR, fname))
        THRESHOLDS[name] = float(model.threshold_)
        print_metrics(name, m)
    except Exception as e:
        print(f"  {name:30s}  HATA: {e}")

try:
    from pyod.models.ecod import ECOD
    train_pyod("ECOD", ECOD, "ecod_model.joblib", contamination=0.2)
except ImportError:
    print("  ECOD                           ATLANACAK")

try:
    from pyod.models.copod import COPOD
    train_pyod("COPOD", COPOD, "copod_model.joblib", contamination=0.2)
except ImportError:
    print("  COPOD                          ATLANACAK")

try:
    from pyod.models.hbos import HBOS
    train_pyod("HBOS", HBOS, "hbos_model.joblib", contamination=0.2)
except ImportError:
    print("  HBOS                           ATLANACAK")

try:
    from pyod.models.cblof import CBLOF
    train_pyod("CBLOF", CBLOF, "cblof_model.joblib", contamination=0.2, n_clusters=8)
except ImportError:
    print("  CBLOF                          ATLANACAK")

try:
    from pyod.models.abod import ABOD
    train_pyod("ABOD", ABOD, "abod_model.joblib", contamination=0.2, method="fast")
except ImportError:
    print("  ABOD                           ATLANACAK")

try:
    from pyod.models.cof import COF
    train_pyod("COF", COF, "cof_model.joblib", contamination=0.2)
except ImportError:
    print("  COF                            ATLANACAK")

try:
    from pyod.models.sod import SOD
    train_pyod("SOD", SOD, "sod_model.joblib", contamination=0.2)
except ImportError:
    print("  SOD                            ATLANACAK")

try:
    from pyod.models.sos import SOS
    train_pyod("SOS", SOS, "sos_model.joblib", contamination=0.2)
except ImportError:
    print("  SOS                            ATLANACAK")

try:
    from pyod.models.loda import LODA
    train_pyod("LODA", LODA, "loda_model.joblib", contamination=0.2)
except ImportError:
    print("  LODA                           ATLANACAK")

try:
    from pyod.models.inne import INNE
    train_pyod("INNE", INNE, "inne_model.joblib", contamination=0.2)
except ImportError:
    print("  INNE                           ATLANACAK")

try:
    from pyod.models.lmdd import LMDD
    train_pyod("LMDD", LMDD, "lmdd_model.joblib", contamination=0.2)
except ImportError:
    print("  LMDD                           ATLANACAK")


if "MLP" in SUP_MODELS:
    joblib.dump(SUP_MODELS["MLP"], os.path.join(MODEL_DIR, "mlp_sklearn_model.joblib"))




print("\n" + "─" * 60)
print("  KAYIT")
print("-" * 60)

with open(os.path.join(UNSUP_DIR, "unsupervised_thresholds.json"), "w") as f:
    json.dump(THRESHOLDS, f, indent=2)
print(f"Thresholds kaydedildi: {len(THRESHOLDS)} model")

with open(os.path.join(METRICS_DIR, "final_comparison.json"), "w") as f:
    json.dump(ALL_METRICS, f, indent=2)
print(f"Metrikler kaydedildi: {len(ALL_METRICS)} model")

print("\n" + "=" * 60)
print("  OZET")
print("=" * 60)

df_sorted = metrics_table(ALL_METRICS, sort_by=PRIMARY_SORT_METRIC)
print(f"\n{'Model':30s}  {'AUC_PR':>7s}  {'F1':>7s}  {'MCC':>7s}  {'AUC_ROC':>7s}  {'Acc':>7s}")
print("-" * 80)
for name, row in df_sorted.head(15).iterrows():
    print(f"{name:30s}  {row['AUC_PR']:7.4f}  {row['F1']:7.4f}  {row['MCC']:7.4f}  "
          f"{row['AUC_ROC']:7.4f}  {row['Accuracy']:7.4f}")
if len(df_sorted) > 15:
    print(f"  ... ve {len(df_sorted) - 15} model daha")

print(f"\nToplam {len(ALL_METRICS)} model egitildi ve kaydedildi.")
print(f"Model dosyalari: {MODEL_DIR}")
print(f"Metrikler: {METRICS_DIR}/final_comparison.json")
