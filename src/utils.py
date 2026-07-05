
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


def save_model(model, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model kaydedildi: {filepath}")


def load_model(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
    model = joblib.load(filepath)
    print(f"Model yüklendi: {filepath}")
    return model


def save_metrics(metrics: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    metrics["kaydedilme_zamani"] = datetime.now().isoformat()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrikler kaydedildi: {filepath}")


def load_metrics(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def create_directory_structure(base_path: str):
    dirs = [
        "data/raw", "data/processed", "data/features",
        "models", "reports/figures", "reports/metrics",
        "app/components", "app/assets", "tests", "notebooks",
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)
    print("Dizin yapısı oluşturuldu.")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    print(f"Rastgele tohum ayarlandı: {seed}")
