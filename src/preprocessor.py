"""
Veri Ön İşleme Modülü (Time Series Preprocessing)
==================================================

Zaman serisi telemetri verilerinin temizlenmesi, gürültüden arındırılması,
aykırı değerlerin tespiti, eksik verilerin doldurulması ve model eğitimi
için ölçeklendirilmesi işlemlerini gerçekleştirir.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer


class TelemetriPreprocessor:
    """
    Zaman serisi uydu telemetri verileri için özel ön işleme sınıfı.

    Özellikler:
        - Eksik veri politikası (varsayılan: 'none' — OPS-SAT metodolojisinde
          boşluklar doldurulmaz, gaps_squared/len/duration ile korunur)
        - Gürültü temizleme (Savitzky-Golay, Butterworth, Median)
        - Aykırı değer (Outlier) tespiti ve işleme (IQR, Z-score, Modified Z-score)
        - Normalizasyon (RobustScaler, StandardScaler, MinMaxScaler)
        - Parametre ve metrik takibi (Metadata export)
    """

    def __init__(self,
                 impute_method: str = "none",
                 filter_method: Optional[str] = "savgol",
                 outlier_method: str = "iqr",
                 scaling_method: str = "robust",
                 window_length: int = 51,
                 polyorder: int = 3,
                 outlier_threshold: float = 3.5):
        """
        Args:
            impute_method (str): Eksik veri politikası. Varsayılan 'none' — OPS-SAT
                metodolojisinde (Ruszczak et al. 2024) boşluklar DOLDURULMAZ; ham
                sinyaldeki eksik noktalar gaps_squared/len/duration özellikleriyle
                korunur ve hatta bir anomali türüdür. Diğer seçenekler (keşif amaçlı):
                'ffill', 'linear', 'spline', 'knn'.
            filter_method (str): Gürültü filtresi ('savgol', 'butterworth', 'median', None).
            outlier_method (str): Aykırı değer tespit yöntemi ('iqr', 'zscore', 'mod_zscore').
            scaling_method (str): Ölçeklendirme ('robust', 'standard', 'minmax').
            window_length (int): Filtreler için pencere boyutu (tek sayı olmalı).
            polyorder (int): Savitzky-Golay filtresi polinom derecesi.
            outlier_threshold (float): Z-score ve modified Z-score eşik değeri.
        """
        self.impute_method = impute_method
        self.filter_method = filter_method
        self.outlier_method = outlier_method
        self.scaling_method = scaling_method
        self.window_length = window_length if window_length % 2 != 0 else window_length + 1
        self.polyorder = polyorder
        self.outlier_threshold = outlier_threshold

        self.scaler = None
        self.is_fitted = False
        self.numeric_columns = []
        self.metadata: Dict[str, Any] = {
            "impute_method": impute_method,
            "filter_method": filter_method,
            "outlier_method": outlier_method,
            "scaling_method": scaling_method,
            "outliers_detected": {},
            "missing_filled": 0
        }

        self._init_scaler()

    def _init_scaler(self):
        """Seçilen ölçeklendirme yöntemini başlatır."""
        scalers = {
            "robust": RobustScaler(),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
        }
        self.scaler = scalers.get(self.scaling_method)
        if self.scaler is None:
            raise ValueError(f"Geçersiz ölçeklendirme yöntemi: {self.scaling_method}")

    def fit(self, data: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> 'TelemetriPreprocessor':
        """
        Ölçeklendiriciyi eğitir.

        Args:
            data (pd.DataFrame): Eğitim verisi (Eksik veri ve outlier işlendikten sonra çağrılmalı).
            numeric_columns (list): Sayısal sütun adları.

        Returns:
            self
        """
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['anomaly', 'label', 'segment', 'train']
            self.numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
        else:
            self.numeric_columns = numeric_columns

        self.scaler.fit(data[self.numeric_columns])
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Eksik veri doldurma, filtreleme, aykırı değer kırpma ve ölçeklendirme
        adımlarını veriye uygular.

        Args:
            data (pd.DataFrame): Dönüştürülecek veri.

        Returns:
            pd.DataFrame: İşlenmiş veri.
        """
        df = data.copy()

        df = self._impute_missing(df)

        if self.filter_method is not None:
            df = self._apply_filter(df)

        df = self._handle_outliers(df)

        if not self.is_fitted:
            raise RuntimeError("Ölçeklendirme için önce fit() çağrılmalı!")
        
        df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        
        return df

    def fit_transform(self, data: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit ve transform adımlarını sırasıyla uygular."""
        df = data.copy()
        
        if numeric_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['anomaly', 'label', 'segment', 'train']
            self.numeric_columns = [col for col in numeric_cols if col not in exclude_cols]
        else:
            self.numeric_columns = numeric_columns

        df = self._impute_missing(df)
        if self.filter_method is not None:
            df = self._apply_filter(df)
        df = self._handle_outliers(df)

        self.fit(df, self.numeric_columns)
        
        df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Eksik veri politikasını uygular.

        Varsayılan 'none': OPS-SAT metodolojisi gereği boşluklar interpolasyonla
        DOLDURULMAZ (ham sinyaldeki boşluklar gaps_squared/len/duration ile zaten
        korunur). Yalnızca özellik matrisindeki dejenere NaN'ler (ör. tek-noktalı
        segmentte kurtosis) sayısal güvenlik için nötr 0 ile doldurulur; bu bir
        sinyal-doldurma değildir.
        """
        missing_count = df[self.numeric_columns].isnull().sum().sum()
        self.metadata["missing_filled"] += int(missing_count)

        if missing_count == 0:
            return df

        if self.impute_method == 'none':
            df[self.numeric_columns] = df[self.numeric_columns].fillna(0)
        elif self.impute_method == 'ffill':
            df[self.numeric_columns] = df[self.numeric_columns].ffill().bfill()
        elif self.impute_method == 'linear':
            df[self.numeric_columns] = df[self.numeric_columns].interpolate(method='linear').bfill()
        elif self.impute_method == 'spline':
            df[self.numeric_columns] = df[self.numeric_columns].interpolate(method='spline', order=3).bfill()
        elif self.impute_method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df[self.numeric_columns] = imputer.fit_transform(df[self.numeric_columns])
        else:
            raise ValueError(f"Geçersiz doldurma yöntemi: {self.impute_method}")
            
        return df

    def _apply_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zaman serisine sinyal filtresi uygular (sadece sürekli sayısal verilere)."""
        for col in self.numeric_columns:
            if self.filter_method == 'savgol':
                df[col] = signal.savgol_filter(df[col], self.window_length, self.polyorder)
            elif self.filter_method == 'median':
                df[col] = signal.medfilt(df[col], kernel_size=self.window_length)
            elif self.filter_method == 'butterworth':
                b, a = signal.butter(4, 0.2, 'lowpass')
                df[col] = signal.filtfilt(b, a, df[col])
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aykırı değerleri tespit eder ve üst/alt sınırlarla kırpar (clip)."""
        for col in self.numeric_columns:
            if self.outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
            elif self.outlier_method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - self.outlier_threshold * std
                upper_bound = mean + self.outlier_threshold * std
                
            elif self.outlier_method == 'mod_zscore':
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                if mad == 0: mad = 1e-6
                margin = (self.outlier_threshold * mad) / 0.6745
                lower_bound = median - margin
                upper_bound = median + margin

            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            self.metadata["outliers_detected"][col] = int(outliers_count)
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        return df

    def save_scaler(self, filepath: str):
        """Eğitilmiş scaler objesini kaydeder."""
        if not self.is_fitted:
            raise RuntimeError("Kaydedilecek eğitilmiş bir scaler yok!")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        print(f"Scaler {filepath} konumuna kaydedildi.")

    def load_scaler(self, filepath: str):
        """Daha önceden kaydedilmiş scaler objesini yükler."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        print(f"Scaler {filepath} konumundan yüklendi.")

    def generate_report(self, filepath: Optional[str] = None) -> Dict:
        """Uygulanan ön işleme adımlarının özetini JSON formatında döndürür ve kaydeder."""
        report = {
            "class": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "parameters": {
                "impute_method": self.impute_method,
                "filter_method": self.filter_method,
                "outlier_method": self.outlier_method,
                "scaling_method": self.scaling_method,
                "window_length": self.window_length,
                "polyorder": self.polyorder,
                "outlier_threshold": self.outlier_threshold
            },
            "stats": self.metadata
        }
        
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            print(f"Ön işleme raporu kaydedildi: {filepath}")
            
        return report
