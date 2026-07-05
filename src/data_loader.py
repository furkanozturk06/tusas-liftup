
import os
import pandas as pd
import numpy as np
from typing import Optional, Union, List


class TelemetryDataLoader:

    SUPPORTED_FORMATS = ['.csv', '.parquet', '.json', '.h5', '.hdf5', '.xlsx']

    def __init__(self, data_path: str = "data/raw/"):
        self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None

    def load_data(
        self,
        filename: Optional[str] = None,
        file_format: str = "csv",
        **kwargs
    ) -> pd.DataFrame:
        if filename is None:
            filename = self._find_first_file(file_format)

        filepath = os.path.join(self.data_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.csv':
            self.data = pd.read_csv(filepath, **kwargs)
        elif ext == '.parquet':
            self.data = pd.read_parquet(filepath, **kwargs)
        elif ext == '.json':
            self.data = pd.read_json(filepath, **kwargs)
        elif ext in ['.h5', '.hdf5']:
            self.data = pd.read_hdf(filepath, **kwargs)
        elif ext == '.xlsx':
            self.data = pd.read_excel(filepath, **kwargs)
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")

        print(f"Veri başarıyla yüklendi: {filepath}")
        print(f"   Boyut: {self.data.shape[0]} satır × {self.data.shape[1]} sütun")

        return self.data

    def _find_first_file(self, file_format: str) -> str:
        ext = f".{file_format}" if not file_format.startswith('.') else file_format

        for fname in os.listdir(self.data_path):
            if fname.endswith(ext):
                return fname

        raise FileNotFoundError(
            f"'{self.data_path}' dizininde '{ext}' uzantılı dosya bulunamadı."
        )

    def get_summary(self) -> dict:
        if self.data is None:
            raise ValueError("Henüz veri yüklenmedi. Önce load_data() çağırın.")

        summary = {
            "satir_sayisi": self.data.shape[0],
            "sutun_sayisi": self.data.shape[1],
            "veri_tipleri": self.data.dtypes.value_counts().to_dict(),
            "eksik_deger_sayisi": self.data.isnull().sum().sum(),
            "eksik_deger_orani": (
                self.data.isnull().sum().sum()
                / (self.data.shape[0] * self.data.shape[1])
                * 100
            ),
            "bellek_kullanimi_mb": self.data.memory_usage(deep=True).sum() / 1e6,
            "sutunlar": list(self.data.columns),
        }

        return summary

    def validate_data(self, required_columns: Optional[List[str]] = None) -> bool:
        if self.data is None:
            raise ValueError("Henüz veri yüklenmedi.")

        is_valid = True

        if self.data.empty:
            print("Veri seti boş!")
            is_valid = False

        if required_columns:
            missing = set(required_columns) - set(self.data.columns)
            if missing:
                print(f"Eksik sütunlar: {missing}")
                is_valid = False

        empty_cols = self.data.columns[self.data.isnull().all()].tolist()
        if empty_cols:
            print(f"Tamamen boş sütunlar: {empty_cols}")

        if is_valid:
            print("Veri doğrulama başarılı.")

        return is_valid
