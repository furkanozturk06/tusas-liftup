import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

def extract_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham telemetri sinyallerini segment bazında gruplar ve kanonik **18 ESA
    handcrafted** özelliğini hesaplar (kanonik modeller bu 18 özelliği bekler;
    bkz. models/test_data.joblib feature_cols ve docs/veri_ve_pipeline.md).

    Not: Eski sürüm `custom_rms/p2p/crest/zcr` ve `channel_id` (24 özellik) de
    üretiyordu; kanonik pipeline 18 ESA özelliğine geçtiği için bunlar kaldırıldı.
    """
    features = []
    
    for segment_id, group in df.groupby('segment'):
        val = group['value'].values
        
        if len(val) == 0:
            continue
            
        f_dict = {}
        f_dict['segment'] = segment_id
        
        f_dict['channel'] = group['channel'].iloc[0] if 'channel' in group.columns else 'UNKNOWN'
        f_dict['anomaly'] = group['anomaly'].iloc[0] if 'anomaly' in group.columns else 0
        f_dict['train'] = group['train'].iloc[0] if 'train' in group.columns else 0
        f_dict['sampling'] = group['sampling'].iloc[0] if 'sampling' in group.columns else 1
        
        n_len = len(val)
        f_dict['len'] = n_len
        f_dict['duration'] = n_len - 1 if n_len > 1 else 1
        
        f_dict['mean'] = np.mean(val)
        f_dict['var'] = np.var(val)
        f_dict['std'] = np.std(val)
        
        f_dict['kurtosis'] = stats.kurtosis(val) if n_len > 3 else 0
        f_dict['skew'] = stats.skew(val) if n_len > 2 else 0
        
        peaks, _ = find_peaks(val)
        f_dict['n_peaks'] = len(peaks)
        
        s10 = pd.Series(val).rolling(10, min_periods=1).mean().values
        p10, _ = find_peaks(s10)
        f_dict['smooth10_n_peaks'] = len(p10)
        
        s20 = pd.Series(val).rolling(20, min_periods=1).mean().values
        p20, _ = find_peaks(s20)
        f_dict['smooth20_n_peaks'] = len(p20)
        
        diff1 = np.diff(val)
        p_diff, _ = find_peaks(diff1)
        f_dict['diff_peaks'] = len(p_diff)
        f_dict['diff_var'] = np.var(diff1) if len(diff1) > 0 else 0
        
        diff2 = np.diff(diff1)
        p_diff2, _ = find_peaks(diff2)
        f_dict['diff2_peaks'] = len(p_diff2)
        f_dict['diff2_var'] = np.var(diff2) if len(diff2) > 0 else 0
        
        f_dict['gaps_squared'] = np.sum(diff1**2)
        f_dict['len_weighted'] = n_len
        
        f_dict['var_div_duration'] = f_dict['var'] / f_dict['duration'] if f_dict['duration'] > 0 else 0
        f_dict['var_div_len'] = f_dict['var'] / f_dict['len'] if f_dict['len'] > 0 else 0

        features.append(f_dict)

    df_features = pd.DataFrame(features)

    df_features = df_features.fillna(0)

    return df_features
