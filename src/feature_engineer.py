"""
Özellik Mühendisliği Modülü (Feature Engineering)
=================================================

Telemetri verilerinden (ESA OPS-SAT) anomali tespiti için
zaman alanı, frekans alanı, fiziksel, çok değişkenli ve gecikmeli (lag)
özellikleri (features) çıkaran gelişmiş modül.
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Tuple, Dict, Any
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

class TelemetryFeatureEngineer:
    """
    Uydu telemetrisi (ESA OPS-SAT) için özellik çıkarım (Feature Engineering)
    sınıfı: zaman / frekans / fiziksel / çok-değişkenli / gecikme özellikleri.
    """
    
    def __init__(self, 
                 rolling_windows: List[int] = [30, 60, 120],
                 lags: List[int] = [1, 5, 10, 30, 60],
                 n_pca_components: int = 5,
                 corr_threshold: float = 0.95):
        """
        Args:
            rolling_windows (List[int]): Pencereli özellikler için pencere boyutları (saniye).
            lags (List[int]): Gecikmeli özellikler için adım sayıları.
            n_pca_components (int): PCA için kullanılacak bileşen sayısı.
            corr_threshold (float): Korelasyon analizi ile elenecek özellikler için eşik.
        """
        self.rolling_windows = rolling_windows
        self.lags = lags
        self.n_pca_components = n_pca_components
        self.corr_threshold = corr_threshold
        
        self.pca = None
        self.variance_selector = None
        self.selected_features = []
        self.feature_metadata: Dict[str, Any] = {}

    def extract_time_domain_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Zaman alanı özelliklerini (istatistiksel, enerji, değişim, trend vb.) hesaplar.
        
        Args:
            df (pd.DataFrame): Ham veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak telemetri sütunları.
            
        Returns:
            pd.DataFrame: Zaman alanı özelliklerinin eklendiği DataFrame.
        """
        result = df.copy()
        
        for col in columns:
            for w in self.rolling_windows:
                rolled = result[col].rolling(window=w, min_periods=1)
                
                result[f'{col}_roll_mean_{w}'] = rolled.mean()
                result[f'{col}_roll_std_{w}'] = rolled.std()
                result[f'{col}_roll_max_{w}'] = rolled.max()
                result[f'{col}_roll_min_{w}'] = rolled.min()
                result[f'{col}_roll_var_{w}'] = rolled.var()
                result[f'{col}_roll_skew_{w}'] = rolled.skew()
                result[f'{col}_roll_kurt_{w}'] = rolled.kurt()
                result[f'{col}_roll_iqr_{w}'] = rolled.quantile(0.75) - rolled.quantile(0.25)
                
                result[f'{col}_rms_{w}'] = np.sqrt((result[col]**2).rolling(window=w, min_periods=1).mean())
                result[f'{col}_p2p_{w}'] = result[f'{col}_roll_max_{w}'] - result[f'{col}_roll_min_{w}']
                result[f'{col}_crest_{w}'] = result[f'{col}_roll_max_{w}'] / (result[f'{col}_rms_{w}'] + 1e-6)
                
            result[f'{col}_roc'] = result[col].diff()
            result[f'{col}_jerk'] = result[f'{col}_roc'].diff()
            
        result.bfill(inplace=True)
        return result

    def extract_frequency_domain_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Frekans alanı özelliklerini (FFT, Spectral Entropy, PSD) hesaplar.
        
        Args:
            df (pd.DataFrame): Veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak telemetri sütunları.
            
        Returns:
            pd.DataFrame: Frekans alanı özelliklerinin eklendiği DataFrame.
        """
        result = df.copy()
        
        for col in columns:
            for w in self.rolling_windows:
                mean_roll = result[col].rolling(window=w, min_periods=1).mean()
                centered = result[col] - mean_roll
                crossings = (np.sign(centered).diff().fillna(0) != 0).astype(float)
                result[f'{col}_zcr_{w}'] = crossings.rolling(window=w, min_periods=1).mean()
                
        result.bfill(inplace=True)
        return result

    def extract_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ESA OPS-SAT sensörlerine (manyetometre CADC0872-0874, fotodiyot
        CADC0884-0894) özel fiziksel proxy özellikleri hesaplar.

        Args:
            df (pd.DataFrame): Veriyi içeren DataFrame.

        Returns:
            pd.DataFrame: Fiziksel özelliklerin eklendiği DataFrame.
        """
        result = df.copy()
        cols = result.columns.tolist()

        mag_cols = ['CADC0872', 'CADC0873', 'CADC0874']
        if all(c in cols for c in mag_cols):
            result['MAG_Magnitude'] = np.sqrt(result['CADC0872']**2 + result['CADC0873']**2 + result['CADC0874']**2)
            
            result['MAG_Sync_Diff'] = result['CADC0872'].abs() - result['CADC0873'].abs()
            
        pd_cols = [c for c in cols if c in ['CADC0884', 'CADC0886', 'CADC0888', 'CADC0890', 'CADC0892', 'CADC0894']]
        if len(pd_cols) > 0:
            result['PD_Total_Sum'] = result[pd_cols].sum(axis=1)
            result['PD_Max_Variance'] = result[pd_cols].var(axis=1)
            
        return result

    def extract_multivariate_features(self, df: pd.DataFrame, columns: List[str], fit_pca: bool = True) -> pd.DataFrame:
        """
        Çok değişkenli özellikleri (PCA, Mahalanobis, Cross-correlation) çıkarır.
        
        Args:
            df (pd.DataFrame): Veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak telemetri sütunları.
            fit_pca (bool): PCA'in baştan fit edilip edilmeyeceği (Train=True, Test=False).
            
        Returns:
            pd.DataFrame: PCA ve Multivariate özelliklerin eklendiği DataFrame.
        """
        result = df.copy()
        
        if len(columns) < 2:
            return result
            
        if fit_pca:
            self.pca = PCA(n_components=min(self.n_pca_components, len(columns)))
            pca_features = self.pca.fit_transform(result[columns].fillna(0))
        else:
            if self.pca is None:
                raise ValueError("PCA henüz fit edilmemiş. Önce train datasıyla fit edin.")
            pca_features = self.pca.transform(result[columns].fillna(0))
            
        for i in range(pca_features.shape[1]):
            result[f'PCA_Component_{i+1}'] = pca_features[:, i]
            
        centroid = result[columns].mean().values
        cov_inv = np.linalg.pinv(result[columns].cov().values)
        
        def mahalanobis(row):
            diff = row.values - centroid
            return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
            
        result['Mahalanobis_Dist'] = result[columns].apply(mahalanobis, axis=1)
        
        return result

    def extract_lag_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Belirlenen gecikme (lag) adımlarına göre özellikleri oluşturur.
        
        Args:
            df (pd.DataFrame): Veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak telemetri sütunları.
            
        Returns:
            pd.DataFrame: Gecikmeli özelliklerin eklendiği DataFrame.
        """
        result = df.copy()
        
        for col in columns:
            for lag in self.lags:
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
                
        result.bfill(inplace=True)
        return result

    def select_features(self, df: pd.DataFrame, protected_cols: List[str] = None, target_col: Optional[str] = None, fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Gereksiz (düşük varyanslı veya aşırı korele) özellikleri eler.
        
        Args:
            df (pd.DataFrame): Özellikleri çıkarılmış DataFrame.
            protected_cols (List[str]): Silinmemesi gereken orijinal sütunlar.
            target_col (str, optional): Hedef değişken sütunu (çıkarılacak).
            fit (bool): Seçicilerin fit edilip edilmeyeceği.
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Seçilen özellikleri içeren DataFrame ve sütun listesi.
        """
        work_df = df.copy()
        if protected_cols is None: protected_cols = []
        
        plot_cols = ['value', 'value_roll_mean_60', 'value_roll_mean_30', 'value_rms_60', 'value_rms_30', 
                     'value_roc', 'value_jerk', 'value_lag_1', 'value_lag_5', 'value_lag_10', 'value_lag_30', 'value_lag_60']
        protected_cols.extend(plot_cols)
        
        meta_cols = [c for c in ['timestamp', 'segment', 'label', 'train', 'channel', target_col] + protected_cols if c in work_df.columns]
        meta_cols = list(dict.fromkeys(meta_cols))
        feature_cols = [c for c in work_df.columns if c not in meta_cols]
        
        if fit:
            self.variance_selector = VarianceThreshold(threshold=1e-5)
            self.variance_selector.fit(work_df[feature_cols])
            
        if self.variance_selector is not None:
            mask = self.variance_selector.get_support()
            feature_cols = [c for c, m in zip(feature_cols, mask) if m]
            
        if fit:
            corr_matrix = work_df[feature_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
            feature_cols = [c for c in feature_cols if c not in to_drop]
            self.selected_features = feature_cols
            self.feature_metadata['dropped_correlated'] = to_drop
        else:
            feature_cols = self.selected_features
            
        final_cols = meta_cols + feature_cols
        
        final_cols = [c for c in final_cols if c in work_df.columns]
        
        return work_df[final_cols], feature_cols

    def extract_segment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zaman serisini segment (olay) bazlı gruplayarak her olay için
        tek satırlık özet özellikler (RMS, Peak-to-Peak vb.) çıkarır.
        """
        print("Segment bazlı istatistikler ve sinyal özellikleri çıkarılıyor...")

        def calculate_segment_stats(group):
            val = group['value'].values

            rms = np.sqrt(np.mean(val**2)) if len(val) > 0 else 0
            p2p = np.ptp(val) if len(val) > 0 else 0
            crest_factor = (np.max(np.abs(val)) / rms) if rms > 0 else 0

            zcr = np.sum(np.diff(np.sign(val)) != 0) / len(val) if len(val) > 1 else 0

            return pd.Series({
                'custom_rms': rms,
                'custom_p2p': p2p,
                'custom_crest_factor': crest_factor,
                'custom_zcr': zcr,
                'anomaly': group['anomaly'].iloc[0],
                'channel': group['channel'].iloc[0]
            })

        segment_features = df.groupby('segment').apply(calculate_segment_stats).reset_index()
        return segment_features


from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy import stats as sp_stats


_DEFAULT_LEN_PROFILE = {"len_mean": 120, "len_std": 90, "len_min": 14, "len_max": 700}


def segment_raw_telemetry(raw_df: pd.DataFrame,
                          profiles: Optional[Dict[str, dict]] = None,
                          train_ratio: float = 0.70,
                          gap_factor: float = 3.0,
                          min_gap_seconds: float = 150.0,
                          min_anomaly_overlap: float = 0.10,
                          seed: int = 42) -> pd.DataFrame:
    """Surekli ham telemetriyi segments.csv formatina boler (HIBRIT).

    Iki asamali, gercek veriye dayali segmentasyon:
      1. **Bosluk-bolme:** Ardisik ornekler arasi Δt, kosu siniri esigini asarsa
         yeni bir "kosu" (kampanya) baslar. Esik = `max(gap_factor * sampling,
         min_gap_seconds)`. `min_gap_seconds` tabani, onboard mikro-bosluklarinin
         (2x sampling) ve buyuk-bosluk artefaktlarinin (<=130s) yanlislikla kosu
         bolmesini onler — bunlar segment ICINDE kalmalidir (gaps_squared'a katki).
      2. **Uzunluk-penceresi:** Her kosu, kanalin gercek segment uzunluk
         dagilimina (`len_mean`/`len_std`, [`len_min`, `len_max`] ile kirpilmis)
         uyan pencerelere bolunur.

    Anomali etiketi ATANMAZ, TURETILIR: bir segment, icinde ground-truth anomali
    ornegi (`_anomaly_truth`) tasiyorsa "anomaly" olarak etiketlenir. Bu sutun
    yoksa tum segmentler nominal kabul edilir.

    Args:
        raw_df: Surekli ham akis (channel, timestamp, value, sampling
                [, _anomaly_truth]).
        profiles: Kanal -> uzunluk profili sozlugu. None ise
                  synthetic_generator.CHANNEL_PROFILES denenir.
        train_ratio: Train olarak isaretlenecek segment orani.
        gap_factor: Kosu siniri esigi (Δt > gap_factor * sampling).
        seed: Pencere uzunlugu ve train atamasi icin rastgelelik tohumu.

    Returns:
        DataFrame — segments.csv formati (channel, timestamp, value, label,
                     sampling, anomaly, segment, train).
    """
    if profiles is None:
        try:
            from synthetic_generator import CHANNEL_PROFILES as profiles
        except ImportError:
            try:
                from src.synthetic_generator import CHANNEL_PROFILES as profiles
            except ImportError:
                profiles = {}

    rng = np.random.default_rng(seed)
    out = []
    seg_id = 0

    for ch, g in raw_df.groupby("channel", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        ts = pd.to_datetime(g["timestamp"])
        dt = ts.diff().dt.total_seconds().fillna(0.0).values
        samp = int(g["sampling"].mode().values[0]) if "sampling" in g.columns else 1
        prof = profiles.get(ch, _DEFAULT_LEN_PROFILE)
        lm, ls = prof["len_mean"], prof["len_std"]
        lmin, lmax = prof["len_min"], prof["len_max"]

        gap_threshold = max(gap_factor * samp, min_gap_seconds)
        run_id = (dt > gap_threshold).cumsum()

        for _rid, run in g.groupby(run_id, sort=False):
            run = run.reset_index(drop=True)
            n = len(run)
            pos = 0
            while pos < n:
                wlen = int(np.clip(round(rng.normal(lm, ls)), lmin, lmax))
                end = min(pos + wlen, n)
                if 0 < n - end < lmin:
                    end = n
                seg_id += 1
                sl = run.iloc[pos:end]
                if "_anomaly_truth" in sl.columns:
                    n_anom = int(sl["_anomaly_truth"].sum())
                    thresh = max(3, int(np.ceil(min_anomaly_overlap * len(sl))))
                    anom = int(n_anom >= thresh)
                else:
                    anom = 0
                for _, r in sl.iterrows():
                    out.append({
                        "channel": ch,
                        "timestamp": r["timestamp"],
                        "value": r["value"],
                        "label": "anomaly" if anom else "nominal",
                        "sampling": samp,
                        "anomaly": anom,
                        "segment": seg_id,
                        "train": 0,
                    })
                pos = end

    df = pd.DataFrame(out)
    if len(df) == 0:
        return df

    seg_ids = df["segment"].unique()
    n_train = int(len(seg_ids) * train_ratio)
    train_ids = set(rng.choice(seg_ids, size=n_train, replace=False))
    df["train"] = df["segment"].isin(train_ids).astype(int)

    n_seg = len(seg_ids)
    n_anom = df.groupby("segment")["anomaly"].first().sum()
    print(f"Segmentasyon tamamlandi: {n_seg} segment ({n_anom} anomali, "
          f"%{n_anom / n_seg * 100:.1f}), {len(df):,} satir.")
    return df


def augment_segments_iccs(segments_df: pd.DataFrame,
                          modes=("omega1", "omega2", "omega3"),
                          nominal_only: bool = True,
                          seed: int = 42) -> pd.DataFrame:
    """ICCS sinyal-seviyesi veri augmentasyonu (Ruszczak et al. 2023).

    Ham segmentlere (segments.csv formati) sinyal-uzayi donusumleri uygular ve
    yeni (sentetik-olmayan, gercek sinyalden tureyen) segmentler uretir. Makale,
    uzman-onayli anomalileri korumak icin augmentasyonu YALNIZ nominal segmentlere
    uygular (nominal_only=True).

    Donusumler (makaledeki terminolojinin yorumu acikca belirtilmistir):
      - omega1 (OX ekseni etrafinda ayna): dikey yansima, v -> 2*median - v.
        Carpiklik (skew) isaretini ters cevirir; ortalama/varyans korunur.
      - omega2 (zaman tersine cevirme): v -> v[::-1]. Sirali/turev tabanli
        ozellikleri etkiler (dagilimsal ozellikler korunur).
      - omega3 (kaydirma): dairesel kaydirma (segment uzunlugunun %15-25'i).

    Yeni segmentler nominal (anomaly=0), train=1 olarak isaretlenir ve yeni
    benzersiz segment kimlikleri alir (orijinal kimliklerle cakismaz).

    Args:
        segments_df: Ham segments.csv (channel, timestamp, value, sampling,
                     anomaly, segment[, train]).
        modes: Uygulanacak donusumler.
        nominal_only: True ise yalniz nominal (anomaly==0) segmentler augmente edilir.
        seed: omega3 kaydirma miktari icin rastgelelik tohumu.

    Returns:
        DataFrame — augmente edilmis YENI segmentler (orijinaller dahil DEGIL),
                    segments.csv ile ayni sutunlar.
    """
    rng = np.random.default_rng(seed)
    src = segments_df
    if nominal_only and "anomaly" in src.columns:
        src = src[src["anomaly"] == 0]

    base_max = int(segments_df["segment"].max())
    out = []
    new_seg = base_max
    for mode in modes:
        for seg_id, grp in src.groupby("segment", sort=True):
            g = grp.sort_values("timestamp") if "timestamp" in grp.columns else grp
            v = g["value"].values.astype(float)
            n = len(v)
            if n < 3:
                continue
            if mode == "omega1":
                med = np.median(v)
                vv = 2.0 * med - v
            elif mode == "omega2":
                vv = v[::-1].copy()
            elif mode == "omega3":
                shift = int(n * rng.uniform(0.15, 0.25))
                vv = np.roll(v, shift if shift > 0 else 1)
            else:
                continue

            new_seg += 1
            ch = g["channel"].iloc[0] if "channel" in g.columns else "AUG"
            samp = int(g["sampling"].iloc[0]) if "sampling" in g.columns else 1
            ts = g["timestamp"].values if "timestamp" in g.columns else range(n)
            for i in range(n):
                out.append({
                    "channel": ch,
                    "timestamp": ts[i],
                    "value": vv[i],
                    "label": "nominal",
                    "sampling": samp,
                    "anomaly": 0,
                    "segment": new_seg,
                    "train": 1,
                })

    aug = pd.DataFrame(out)
    n_new = aug["segment"].nunique() if len(aug) else 0
    print(f"ICCS augmentasyon: {modes} -> {n_new} yeni nominal segment "
          f"(kaynak: {src['segment'].nunique()} nominal segment).")
    return aug


def extract_esa_features(segments_df: pd.DataFrame,
                         prominence_ratio: float = 0.10) -> pd.DataFrame:
    """Ham telemetri segmentlerinden ESA OPSSAT-AD 18 handcrafted feature'i cikarir.

    Her segment icin tek satirlik ozet ozellikler hesaplanir.

    Beklenen sutunlar (segments.csv formati):
        segment, channel, timestamp, value, sampling, anomaly, train[, label]

    Cikti sutunlari (dataset.csv formati):
        segment, anomaly, train, channel, sampling,
        duration, len, mean, var, std, kurtosis, skew,
        n_peaks, smooth10_n_peaks, smooth20_n_peaks,
        diff_peaks, diff2_peaks, diff_var, diff2_var,
        gaps_squared, len_weighted, var_div_duration, var_div_len

    Args:
        segments_df: Ham segments verisi (her satir bir zaman noktasi).
        prominence_ratio: Tepe tespiti icin minimum belirginlik orani
                          (sinyal genliginin yuzde kaci). ESA orijinali = 0.10.

    Returns:
        DataFrame — segment basina 18 ozellik + 5 meta sutun.
    """
    required = {'segment', 'value', 'timestamp'}
    missing = required - set(segments_df.columns)
    if missing:
        raise ValueError(f"Eksik sutunlar: {missing}")

    records = []
    grouped = segments_df.groupby('segment', sort=True)
    total = len(grouped)

    for i, (seg_id, grp) in enumerate(grouped, 1):
        if i % 200 == 0 or i == total:
            print(f"  [{i}/{total}] segment isleniyor...")

        val = grp['value'].values
        n = len(val)
        if n == 0:
            continue

        channel = grp['channel'].iloc[0] if 'channel' in grp.columns else 'UNKNOWN'
        anomaly = int(grp['anomaly'].iloc[0]) if 'anomaly' in grp.columns else 0
        train = int(grp['train'].iloc[0]) if 'train' in grp.columns else 0
        sampling = int(grp['sampling'].iloc[0]) if 'sampling' in grp.columns else 1

        ts = pd.to_datetime(grp['timestamp'])
        if n > 1:
            td = ts.diff().dt.total_seconds().dropna().values
            duration = int(ts.iloc[-1].timestamp() - ts.iloc[0].timestamp())
        else:
            td = np.array([1.0])
            duration = 1


        mean_val = np.mean(val)
        var_val = np.var(val)
        std_val = np.std(val)
        kurt_val = sp_stats.kurtosis(val) if n > 3 else 0.0
        skew_val = sp_stats.skew(val) if n > 2 else 0.0

        p2p = np.ptp(val)
        prom = prominence_ratio * p2p if p2p > 0 else None
        peaks, _ = find_peaks(val, prominence=prom)
        n_peaks = len(peaks)

        seg_len = n
        gaps_squared = int(np.sum(td ** 2))
        len_weighted = seg_len * sampling
        var_div_duration = var_val / duration if duration > 0 else 0.0
        var_div_len = var_val / seg_len if seg_len > 0 else 0.0

        s10 = np.convolve(val, np.ones(10) / 10, mode='same')
        s20 = np.convolve(val, np.ones(20) / 20, mode='same')
        prom10 = prominence_ratio * np.ptp(s10) if np.ptp(s10) > 0 else None
        prom20 = prominence_ratio * np.ptp(s20) if np.ptp(s20) > 0 else None
        pk10, _ = find_peaks(s10, prominence=prom10)
        pk20, _ = find_peaks(s20, prominence=prom20)

        diff1 = np.diff(val)
        diff2 = np.diff(diff1)

        prom_d1 = prominence_ratio * np.ptp(diff1) if len(diff1) > 0 and np.ptp(diff1) > 0 else None
        prom_d2 = prominence_ratio * np.ptp(diff2) if len(diff2) > 0 and np.ptp(diff2) > 0 else None
        dpk1, _ = find_peaks(diff1, prominence=prom_d1) if len(diff1) > 0 else (np.array([]), {})
        dpk2, _ = find_peaks(diff2, prominence=prom_d2) if len(diff2) > 0 else (np.array([]), {})

        diff_var = np.var(diff1) if len(diff1) > 0 else 0.0
        diff2_var = np.var(diff2) if len(diff2) > 0 else 0.0

        records.append({
            'segment': seg_id,
            'anomaly': anomaly,
            'train': train,
            'channel': channel,
            'sampling': sampling,
            'duration': duration,
            'len': seg_len,
            'mean': mean_val,
            'var': var_val,
            'std': std_val,
            'kurtosis': kurt_val,
            'skew': skew_val,
            'n_peaks': n_peaks,
            'smooth10_n_peaks': len(pk10),
            'smooth20_n_peaks': len(pk20),
            'diff_peaks': len(dpk1),
            'diff2_peaks': len(dpk2),
            'diff_var': diff_var,
            'diff2_var': diff2_var,
            'gaps_squared': gaps_squared,
            'len_weighted': len_weighted,
            'var_div_duration': var_div_duration,
            'var_div_len': var_div_len,
        })

    df_out = pd.DataFrame(records)
    print(f"Tamamlandi: {len(df_out)} segment, 18 ozellik cikarildi.")
    return df_out

    def transform(self, df: pd.DataFrame, columns: List[str], target_col: Optional[str] = 'anomaly', fit: bool = True) -> pd.DataFrame:
        """
        Tüm özellik mühendisliği aşamalarını (Time, Freq, Physical, Multi, Lag, Select)
        sırasıyla çalıştırarak nihai özellik matrisini oluşturur.
        
        Args:
            df (pd.DataFrame): Ham veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak temel telemetri sütunları.
            target_col (str, optional): Hedef değişkenin adı.
            fit (bool): PCA ve Feature Selection nesnelerinin eğitilip eğitilmeyeceği.
            
        Returns:
            pd.DataFrame: Zenginleştirilmiş ve filtrelenmiş özellik DataFrame'i.
        """
        print("1. Time Domain özellikleri hesaplanıyor...")
        df = self.extract_time_domain_features(df, columns)
        
        print("2. Frequency Domain özellikleri hesaplanıyor...")
        df = self.extract_frequency_domain_features(df, columns)
        
        print("3. Physical özellikleri hesaplanıyor...")
        df = self.extract_physical_features(df)
        
        print("4. Multivariate özellikleri hesaplanıyor...")
        df = self.extract_multivariate_features(df, columns, fit_pca=fit)
        
        print("5. Lag özellikleri hesaplanıyor...")
        df = self.extract_lag_features(df, columns)
        
        print("6. Feature Selection uygulanıyor...")
        df, final_features = self.select_features(df, protected_cols=columns, target_col=target_col, fit=fit)
        
        self.feature_metadata['total_features_generated'] = len(final_features)
        print(f"İşlem tamam! Toplam {len(final_features)} adet özellik üretildi ve seçildi.")
        
        return df
