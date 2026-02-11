"""
================================================================================
LEO SATELLITE TELEMETRY ANOMALY DETECTION SYSTEM
Advanced Machine Learning Pipeline for Spacecraft Health Monitoring
================================================================================

Author: Aerospace Data Science Team
Version: 2.0.0
Purpose: Production-grade anomaly detection for reaction wheel telemetry

This system implements:
- Multiple anomaly detection algorithms (Isolation Forest, One-Class SVM, 
  Local Outlier Factor, Autoencoder)
- Comprehensive feature engineering (time-domain, frequency-domain, physics-based)
- Ensemble voting with configurable weights
- Operator alerting dashboard with severity classification
- Concept drift detection and model retraining triggers

Anomaly Types Detected:
1. Point anomalies (sudden spikes/drops)
2. Contextual anomalies (eclipse-related, orbital phase)
3. Collective anomalies (multi-parameter inconsistencies)
4. Trend/drift anomalies (slow degradation)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import json
import os

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, precision_recall_fscore_support,
    average_precision_score
)
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats, signal
from scipy.fft import fft, fftfreq

warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURATION & CONSTANTS
# ================================================================================

class AnomalyType(Enum):
    """Enumeration of anomaly types in satellite telemetry"""
    NORMAL = "normal"
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    DRIFT = "drift"
    SPEED = "speed_"
    TORQUE = "torque"
    CURRENT = "curren"
    VIBRATION = "vibrat"
    OVERTEMP = "overte"

class AlertSeverity(Enum):
    """Alert severity levels for operator notification"""
    NOMINAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TelemetryConfig:
    """Configuration for telemetry processing"""
    # Feature columns
    feature_columns: List[str] = field(default_factory=lambda: [
        'wheel_speed_rpm', 'wheel_torque', 'motor_current', 
        'motor_temp', 'vibration_level'
    ])
    
    # Operational limits (physics-based constraints)
    operational_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'wheel_speed_rpm': (0, 6000),      # RPM limits
        'wheel_torque': (-0.5, 0.5),        # Nm limits
        'motor_current': (0, 3.0),          # Ampere limits
        'motor_temp': (-40, 85),            # Celsius limits
        'vibration_level': (0, 2.0)         # g-force limits
    })
    
    # Window sizes for feature engineering
    short_window: int = 6      # 6 hours
    medium_window: int = 24    # 1 day
    long_window: int = 168     # 1 week
    
    # Model parameters
    contamination_rate: float = 0.05
    anomaly_threshold_percentile: float = 95
    
    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'isolation_forest': 0.25,
        'one_class_svm': 0.20,
        'local_outlier_factor': 0.20,
        'autoencoder': 0.35
    })
    
    # Alert thresholds
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.3,
        'medium': 0.5,
        'high': 0.7,
        'critical': 0.85
    })

@dataclass
class ModelResults:
    """Container for model evaluation results"""
    model_name: str
    predictions: np.ndarray
    scores: np.ndarray
    threshold: float
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    false_alarm_rate: float = 0.0
    detection_rate: float = 0.0

# ================================================================================
# DATA LOADING & VALIDATION
# ================================================================================

class TelemetryDataLoader:
    """
    Handles loading and initial validation of satellite telemetry data.
    Implements aerospace-standard data quality checks.
    """
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.data_quality_report = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load telemetry data with comprehensive validation"""
        print("\n" + "="*70)
        print("TELEMETRY DATA LOADING & VALIDATION")
        print("="*70)
        
        # Load CSV
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"\n[DATA SUMMARY]")
        print(f"  Total records: {len(df):,}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Duration: {(df.index.max() - df.index.min()).days} days")
        print(f"  Sampling rate: ~{self._estimate_sampling_rate(df)}")
        
        # Validate data quality
        self._validate_data_quality(df)
        
        return df
    
    def _estimate_sampling_rate(self, df: pd.DataFrame) -> str:
        """Estimate telemetry sampling rate"""
        if len(df) < 2:
            return "Unknown"
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        return str(median_diff)
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """Perform comprehensive data quality checks"""
        print(f"\n[DATA QUALITY ASSESSMENT]")
        
        # Check for missing values
        missing = df[self.config.feature_columns].isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        print(f"\n  Missing Values:")
        for col in self.config.feature_columns:
            status = "✓" if missing[col] == 0 else "⚠"
            print(f"    {status} {col}: {missing[col]} ({missing_pct[col]}%)")
        
        # Check for out-of-range values (physics violations)
        print(f"\n  Physics Constraint Violations:")
        for col, (min_val, max_val) in self.config.operational_limits.items():
            if col in df.columns:
                violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
                pct = violations / len(df) * 100
                status = "✓" if violations == 0 else "⚠"
                print(f"    {status} {col}: {violations} violations ({pct:.2f}%)")
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        print(f"\n  Duplicate timestamps: {duplicates}")
        
        # Check for time gaps
        time_gaps = self._detect_time_gaps(df)
        print(f"  Detected time gaps (>2h): {len(time_gaps)}")
        
        self.data_quality_report = {
            'missing_values': missing.to_dict(),
            'time_gaps': len(time_gaps),
            'duplicates': duplicates
        }
    
    def _detect_time_gaps(self, df: pd.DataFrame, 
                          threshold_hours: int = 2) -> List[Tuple]:
        """Detect gaps in telemetry timeline"""
        time_diffs = df.index.to_series().diff()
        threshold = pd.Timedelta(hours=threshold_hours)
        gaps = time_diffs[time_diffs > threshold]
        return [(idx, diff) for idx, diff in gaps.items()]

# ================================================================================
# DATA PREPROCESSING
# ================================================================================

class TelemetryPreprocessor:
    """
    Advanced preprocessing for satellite telemetry data.
    Handles noise, outliers, missing data, and normalization.
    """
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.scalers = {}
        self.statistics = {}
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full preprocessing pipeline"""
        print("\n" + "="*70)
        print("DATA PREPROCESSING")
        print("="*70)
        
        df_processed = df.copy()
        
        # Step 1: Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Step 2: Remove physics-violating outliers
        df_processed = self._handle_physics_violations(df_processed)
        
        # Step 3: Apply noise filtering
        df_processed = self._apply_noise_filtering(df_processed)
        
        # Step 4: Normalize features
        df_processed = self._normalize_features(df_processed)
        
        print(f"\n  Preprocessing complete. Shape: {df_processed.shape}")
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with forward/backward fill and interpolation"""
        print("\n[1/4] Handling missing values...")
        
        for col in self.config.feature_columns:
            if col in df.columns:
                # First try forward fill (most recent known value)
                df[col] = df[col].ffill()
                # Then backward fill for any remaining
                df[col] = df[col].bfill()
                # Linear interpolation for any gaps
                df[col] = df[col].interpolate(method='linear')
        
        remaining_missing = df[self.config.feature_columns].isnull().sum().sum()
        print(f"    Remaining missing values: {remaining_missing}")
        
        return df.dropna(subset=self.config.feature_columns)
    
    def _handle_physics_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap values at physical operational limits"""
        print("\n[2/4] Enforcing physics constraints...")
        
        for col, (min_val, max_val) in self.config.operational_limits.items():
            if col in df.columns:
                original_violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
                df[col] = df[col].clip(lower=min_val, upper=max_val)
                print(f"    {col}: {original_violations} values capped")
        
        return df
    
    def _apply_noise_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Savitzky-Golay filter for noise reduction"""
        print("\n[3/4] Applying noise filtering...")
        
        window_length = 5  # Must be odd
        poly_order = 2
        
        for col in self.config.feature_columns:
            if col in df.columns and len(df) > window_length:
                try:
                    df[f'{col}_filtered'] = signal.savgol_filter(
                        df[col].values, window_length, poly_order
                    )
                except Exception as e:
                    df[f'{col}_filtered'] = df[col]
                    print(f"    Warning: Could not filter {col}: {e}")
        
        print(f"    Applied Savitzky-Golay filter (window={window_length})")
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using RobustScaler (resistant to outliers)"""
        print("\n[4/4] Normalizing features...")
        
        for col in self.config.feature_columns:
            if col in df.columns:
                scaler = RobustScaler()
                df[f'{col}_normalized'] = scaler.fit_transform(
                    df[col].values.reshape(-1, 1)
                ).flatten()
                self.scalers[col] = scaler
                
                # Store statistics
                self.statistics[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                }
        
        print(f"    Normalized {len(self.config.feature_columns)} features")
        return df

# ================================================================================
# FEATURE ENGINEERING
# ================================================================================

class FeatureEngineer:
    """
    Comprehensive feature engineering for satellite telemetry.
    Extracts time-domain, frequency-domain, and physics-based features.
    """
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full feature engineering pipeline"""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70)
        
        df_features = df.copy()
        
        # Time-domain features
        df_features = self._extract_time_domain_features(df_features)
        
        # Rate of change features
        df_features = self._extract_derivative_features(df_features)
        
        # Rolling statistics (multiple windows)
        df_features = self._extract_rolling_features(df_features)
        
        # Cross-correlation features
        df_features = self._extract_cross_correlation_features(df_features)
        
        # Physics-based derived features
        df_features = self._extract_physics_features(df_features)
        
        # Lag features for temporal patterns
        df_features = self._extract_lag_features(df_features)
        
        # Statistical distribution features
        df_features = self._extract_distribution_features(df_features)
        
        # Drop rows with NaN from rolling windows
        df_features = df_features.dropna()
        
        print(f"\n  Total engineered features: {len(self.feature_names)}")
        print(f"  Final dataset shape: {df_features.shape}")
        
        return df_features
    
    def _extract_time_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic time-domain statistical features"""
        print("\n[1/7] Extracting time-domain features...")
        
        window = self.config.short_window
        
        for col in self.config.feature_columns:
            if col not in df.columns:
                continue
                
            # Basic statistics over short window
            df[f'{col}_mean_{window}h'] = df[col].rolling(window).mean()
            df[f'{col}_std_{window}h'] = df[col].rolling(window).std()
            df[f'{col}_min_{window}h'] = df[col].rolling(window).min()
            df[f'{col}_max_{window}h'] = df[col].rolling(window).max()
            df[f'{col}_range_{window}h'] = df[f'{col}_max_{window}h'] - df[f'{col}_min_{window}h']
            df[f'{col}_median_{window}h'] = df[col].rolling(window).median()
            
            # Add to feature list
            self.feature_names.extend([
                f'{col}_mean_{window}h', f'{col}_std_{window}h',
                f'{col}_min_{window}h', f'{col}_max_{window}h',
                f'{col}_range_{window}h', f'{col}_median_{window}h'
            ])
        
        print(f"    Extracted {len(self.config.feature_columns) * 6} time-domain features")
        return df
    
    def _extract_derivative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract rate of change (derivative) features"""
        print("\n[2/7] Extracting derivative features...")
        
        for col in self.config.feature_columns:
            if col not in df.columns:
                continue
            
            # First derivative (rate of change)
            df[f'{col}_diff1'] = df[col].diff(1)
            
            # Second derivative (acceleration of change)
            df[f'{col}_diff2'] = df[col].diff(2)
            
            # Percentage change
            df[f'{col}_pct_change'] = df[col].pct_change().replace([np.inf, -np.inf], 0)
            
            self.feature_names.extend([
                f'{col}_diff1', f'{col}_diff2', f'{col}_pct_change'
            ])
        
        print(f"    Extracted {len(self.config.feature_columns) * 3} derivative features")
        return df
    
    def _extract_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract multi-scale rolling window features"""
        print("\n[3/7] Extracting multi-scale rolling features...")
        
        windows = [self.config.medium_window, self.config.long_window]
        feature_count = 0
        
        for col in self.config.feature_columns:
            if col not in df.columns:
                continue
                
            for window in windows:
                if len(df) > window:
                    # Rolling mean and std
                    df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window).mean()
                    df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window).std()
                    
                    # Z-score relative to rolling window
                    df[f'{col}_zscore_{window}h'] = (
                        (df[col] - df[f'{col}_rolling_mean_{window}h']) / 
                        (df[f'{col}_rolling_std_{window}h'] + 1e-8)
                    )
                    
                    # Deviation from rolling mean
                    df[f'{col}_deviation_{window}h'] = (
                        df[col] - df[f'{col}_rolling_mean_{window}h']
                    )
                    
                    self.feature_names.extend([
                        f'{col}_rolling_mean_{window}h',
                        f'{col}_rolling_std_{window}h',
                        f'{col}_zscore_{window}h',
                        f'{col}_deviation_{window}h'
                    ])
                    feature_count += 4
        
        print(f"    Extracted {feature_count} multi-scale features")
        return df
    
    def _extract_cross_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract cross-correlation features between telemetry parameters"""
        print("\n[4/7] Extracting cross-correlation features...")
        
        feature_count = 0
        cols = [c for c in self.config.feature_columns if c in df.columns]
        
        # Speed-Current ratio (efficiency indicator)
        if 'wheel_speed_rpm' in df.columns and 'motor_current' in df.columns:
            df['speed_current_ratio'] = (
                df['wheel_speed_rpm'] / (df['motor_current'] + 0.001)
            )
            self.feature_names.append('speed_current_ratio')
            feature_count += 1
        
        # Power proxy (current * voltage approximation via torque)
        if 'motor_current' in df.columns and 'wheel_torque' in df.columns:
            df['power_proxy'] = df['motor_current'] * np.abs(df['wheel_torque'])
            self.feature_names.append('power_proxy')
            feature_count += 1
        
        # Thermal-mechanical correlation
        if 'motor_temp' in df.columns and 'motor_current' in df.columns:
            df['thermal_current_ratio'] = (
                df['motor_temp'] / (df['motor_current'] + 0.001)
            )
            self.feature_names.append('thermal_current_ratio')
            feature_count += 1
        
        # Vibration-speed correlation
        if 'vibration_level' in df.columns and 'wheel_speed_rpm' in df.columns:
            df['vibration_speed_ratio'] = (
                df['vibration_level'] / (df['wheel_speed_rpm'] / 1000 + 0.001)
            )
            self.feature_names.append('vibration_speed_ratio')
            feature_count += 1
        
        # Pairwise rolling correlations
        window = self.config.short_window
        for i, col1 in enumerate(cols[:-1]):
            for col2 in cols[i+1:]:
                corr_name = f'{col1}_{col2}_corr_{window}h'
                df[corr_name] = df[col1].rolling(window).corr(df[col2])
                self.feature_names.append(corr_name)
                feature_count += 1
        
        print(f"    Extracted {feature_count} cross-correlation features")
        return df
    
    def _extract_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract physics-informed features specific to reaction wheels"""
        print("\n[5/7] Extracting physics-based features...")
        
        feature_count = 0
        
        # Angular momentum proxy (I * omega)
        if 'wheel_speed_rpm' in df.columns:
            # Convert RPM to rad/s: omega = RPM * 2*pi/60
            df['angular_velocity_rad'] = df['wheel_speed_rpm'] * (2 * np.pi / 60)
            # Assume typical reaction wheel moment of inertia ~0.01 kg*m^2
            df['angular_momentum'] = 0.01 * df['angular_velocity_rad']
            self.feature_names.extend(['angular_velocity_rad', 'angular_momentum'])
            feature_count += 2
        
        # Mechanical power (Torque * Angular velocity)
        if 'wheel_torque' in df.columns and 'angular_velocity_rad' in df.columns:
            df['mechanical_power'] = (
                np.abs(df['wheel_torque']) * df['angular_velocity_rad']
            )
            self.feature_names.append('mechanical_power')
            feature_count += 1
        
        # Electrical power estimate (V * I, assume 28V bus)
        if 'motor_current' in df.columns:
            df['electrical_power'] = 28.0 * df['motor_current']
            self.feature_names.append('electrical_power')
            feature_count += 1
        
        # Efficiency estimate (mechanical / electrical)
        if 'mechanical_power' in df.columns and 'electrical_power' in df.columns:
            df['efficiency'] = (
                df['mechanical_power'] / (df['electrical_power'] + 0.001)
            ).clip(0, 1)
            self.feature_names.append('efficiency')
            feature_count += 1
        
        # Thermal gradient (rate of temperature change)
        if 'motor_temp' in df.columns:
            df['thermal_gradient'] = df['motor_temp'].diff(1)
            df['thermal_gradient_6h'] = df['motor_temp'].diff(6)
            self.feature_names.extend(['thermal_gradient', 'thermal_gradient_6h'])
            feature_count += 2
        
        # Bearing health indicator (vibration normalized by speed)
        if 'vibration_level' in df.columns and 'wheel_speed_rpm' in df.columns:
            df['bearing_health_indicator'] = (
                df['vibration_level'] / 
                (df['wheel_speed_rpm'] / 3000 + 0.1)  # Normalized to 3000 RPM
            )
            self.feature_names.append('bearing_health_indicator')
            feature_count += 1
        
        print(f"    Extracted {feature_count} physics-based features")
        return df
    
    def _extract_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract lag features for temporal pattern detection"""
        print("\n[6/7] Extracting lag features...")
        
        lags = [1, 3, 6, 12, 24]  # Hours
        feature_count = 0
        
        for col in self.config.feature_columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                if len(df) > lag:
                    df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
                    self.feature_names.append(f'{col}_lag_{lag}h')
                    feature_count += 1
        
        print(f"    Extracted {feature_count} lag features")
        return df
    
    def _extract_distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical distribution features"""
        print("\n[7/7] Extracting distribution features...")
        
        window = self.config.medium_window
        feature_count = 0
        
        for col in self.config.feature_columns:
            if col not in df.columns:
                continue
            
            # Skewness
            df[f'{col}_skew_{window}h'] = df[col].rolling(window).skew()
            
            # Kurtosis
            df[f'{col}_kurt_{window}h'] = df[col].rolling(window).kurt()
            
            # Quantile features
            df[f'{col}_q10_{window}h'] = df[col].rolling(window).quantile(0.1)
            df[f'{col}_q90_{window}h'] = df[col].rolling(window).quantile(0.9)
            df[f'{col}_iqr_{window}h'] = (
                df[f'{col}_q90_{window}h'] - df[f'{col}_q10_{window}h']
            )
            
            self.feature_names.extend([
                f'{col}_skew_{window}h', f'{col}_kurt_{window}h',
                f'{col}_q10_{window}h', f'{col}_q90_{window}h',
                f'{col}_iqr_{window}h'
            ])
            feature_count += 5
        
        print(f"    Extracted {feature_count} distribution features")
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get the final feature matrix for modeling"""
        # Select only engineered features
        available_features = [f for f in self.feature_names if f in df.columns]
        return df[available_features].copy()

# ================================================================================
# ANOMALY DETECTION MODELS
# ================================================================================

class AnomalyDetector:
    """
    Individual anomaly detection model wrapper.
    Supports Isolation Forest, One-Class SVM, LOF, and Autoencoder.
    """
    
    def __init__(self, model_type: str, config: TelemetryConfig):
        self.model_type = model_type
        self.config = config
        self.model = None
        self.threshold = None
        self.is_fitted = False
        
    def build_model(self):
        """Build the anomaly detection model"""
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                n_estimators=200,
                max_samples='auto',
                contamination=self.config.contamination_rate,
                max_features=1.0,
                bootstrap=False,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        elif self.model_type == 'one_class_svm':
            self.model = OneClassSVM(
                kernel='rbf',
                nu=self.config.contamination_rate,
                gamma='scale',
                shrinking=True,
                cache_size=500,
                verbose=False
            )
        elif self.model_type == 'local_outlier_factor':
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                algorithm='auto',
                leaf_size=30,
                metric='minkowski',
                contamination=self.config.contamination_rate,
                novelty=True,
                n_jobs=-1
            )
        elif self.model_type == 'autoencoder':
            # MLP-based autoencoder
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16, 8, 16, 32, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                shuffle=True,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self
    
    def fit(self, X_train: np.ndarray) -> 'AnomalyDetector':
        """Fit the model on normal training data"""
        if self.model is None:
            self.build_model()
        
        if self.model_type == 'autoencoder':
            self.model.fit(X_train, X_train)
        else:
            self.model.fit(X_train)
        
        self.is_fitted = True
        return self
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous)"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if self.model_type == 'autoencoder':
            X_reconstructed = self.model.predict(X)
            scores = np.mean(np.power(X - X_reconstructed, 2), axis=1)
        elif self.model_type in ['isolation_forest', 'one_class_svm']:
            scores = -self.model.decision_function(X)
        elif self.model_type == 'local_outlier_factor':
            scores = -self.model.decision_function(X)
        
        return scores
    
    def set_threshold(self, scores: np.ndarray, percentile: float = 95) -> float:
        """Set anomaly threshold based on score distribution"""
        self.threshold = np.percentile(scores, percentile)
        return self.threshold
    
    def predict(self, X: np.ndarray, 
                threshold: Optional[float] = None) -> np.ndarray:
        """Predict anomalies (0=normal, 1=anomaly)"""
        scores = self.predict_scores(X)
        thresh = threshold if threshold is not None else self.threshold
        
        if thresh is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        return (scores > thresh).astype(int)


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple models.
    Uses weighted voting with configurable weights.
    """
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.models: Dict[str, AnomalyDetector] = {}
        self.ensemble_scores = None
        self.ensemble_threshold = None
        
    def build_ensemble(self) -> 'EnsembleAnomalyDetector':
        """Build all models in the ensemble"""
        print("\n" + "="*70)
        print("BUILDING ENSEMBLE ANOMALY DETECTOR")
        print("="*70)
        
        model_types = list(self.config.ensemble_weights.keys())
        
        for model_type in model_types:
            print(f"\n  Building {model_type}...")
            self.models[model_type] = AnomalyDetector(model_type, self.config)
            self.models[model_type].build_model()
        
        print(f"\n  Total models: {len(self.models)}")
        return self
    
    def fit(self, X_train: np.ndarray) -> 'EnsembleAnomalyDetector':
        """Fit all models on normal training data"""
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\n  Training {name}...")
            model.fit(X_train)
            
            # Set threshold on training data scores
            train_scores = model.predict_scores(X_train)
            model.set_threshold(train_scores, self.config.anomaly_threshold_percentile)
            print(f"    Threshold: {model.threshold:.4f}")
        
        print("\n  All models trained successfully.")
        return self
    
    def predict_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get normalized anomaly scores from all models"""
        all_scores = {}
        
        for name, model in self.models.items():
            scores = model.predict_scores(X)
            # Min-max normalization to [0, 1]
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            all_scores[name] = scores_norm
        
        return all_scores
    
    def compute_ensemble_score(self, 
                               all_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted ensemble anomaly score"""
        ensemble_score = np.zeros(len(list(all_scores.values())[0]))
        
        for name, scores in all_scores.items():
            weight = self.config.ensemble_weights.get(name, 0)
            ensemble_score += weight * scores
        
        return ensemble_score
    
    def set_ensemble_threshold(self, 
                               ensemble_scores: np.ndarray,
                               percentile: float = 95) -> float:
        """Set threshold for ensemble predictions"""
        self.ensemble_threshold = np.percentile(ensemble_scores, percentile)
        return self.ensemble_threshold
    
    def predict(self, X: np.ndarray,
                return_details: bool = False) -> Union[np.ndarray, Dict]:
        """Predict anomalies using ensemble"""
        all_scores = self.predict_scores(X)
        ensemble_scores = self.compute_ensemble_score(all_scores)
        
        if self.ensemble_threshold is None:
            self.set_ensemble_threshold(ensemble_scores, 
                                        self.config.anomaly_threshold_percentile)
        
        predictions = (ensemble_scores > self.ensemble_threshold).astype(int)
        
        if return_details:
            return {
                'predictions': predictions,
                'ensemble_scores': ensemble_scores,
                'individual_scores': all_scores,
                'threshold': self.ensemble_threshold
            }
        
        return predictions
    
    def classify_severity(self, scores: np.ndarray) -> List[AlertSeverity]:
        """Classify anomaly severity levels"""
        severities = []
        thresholds = self.config.severity_thresholds
        
        for score in scores:
            if score < thresholds['low']:
                severities.append(AlertSeverity.NOMINAL)
            elif score < thresholds['medium']:
                severities.append(AlertSeverity.LOW)
            elif score < thresholds['high']:
                severities.append(AlertSeverity.MEDIUM)
            elif score < thresholds['critical']:
                severities.append(AlertSeverity.HIGH)
            else:
                severities.append(AlertSeverity.CRITICAL)
        
        return severities


# ================================================================================
# MODEL EVALUATION
# ================================================================================

class ModelEvaluator:
    """
    Comprehensive evaluation of anomaly detection models.
    Calculates metrics from operator and mission perspective.
    """
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.results: Dict[str, ModelResults] = {}
        
    def evaluate_model(self, 
                       model_name: str,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       scores: np.ndarray,
                       threshold: float) -> ModelResults:
        """Evaluate a single model"""
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, scores)
        except:
            roc_auc = 0.0
        
        # False Alarm Rate (False Positives / Total Predictions as Positive)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_pred)) > 1 else (0,0,0,0)
        false_alarm_rate = fp / (fp + tp + 1e-8)
        
        # Detection Rate (True Positives / Total Actual Positives)
        detection_rate = tp / (tp + fn + 1e-8)
        
        result = ModelResults(
            model_name=model_name,
            predictions=y_pred,
            scores=scores,
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            false_alarm_rate=false_alarm_rate,
            detection_rate=detection_rate
        )
        
        self.results[model_name] = result
        return result
    
    def evaluate_all(self,
                     y_true: np.ndarray,
                     ensemble_detector: EnsembleAnomalyDetector,
                     X: np.ndarray) -> Dict[str, ModelResults]:
        """Evaluate all models in the ensemble"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        all_scores = ensemble_detector.predict_scores(X)
        
        # Evaluate individual models
        for name, model in ensemble_detector.models.items():
            scores = all_scores[name]
            preds = (scores > np.percentile(scores, 95)).astype(int)
            self.evaluate_model(name, y_true, preds, scores, model.threshold)
        
        # Evaluate ensemble
        ensemble_scores = ensemble_detector.compute_ensemble_score(all_scores)
        ensemble_preds = (ensemble_scores > ensemble_detector.ensemble_threshold).astype(int)
        self.evaluate_model('ensemble', y_true, ensemble_preds, 
                           ensemble_scores, ensemble_detector.ensemble_threshold)
        
        # Print results
        self._print_results()
        
        return self.results
    
    def _print_results(self) -> None:
        """Print evaluation results in formatted table"""
        print("\n" + "-"*70)
        print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
        print("-"*70)
        
        for name, result in self.results.items():
            print(f"{name:<25} {result.precision:>10.4f} {result.recall:>10.4f} "
                  f"{result.f1_score:>10.4f} {result.roc_auc:>10.4f}")
        
        print("-"*70)
        
        # Print operational metrics
        print("\n[OPERATIONAL METRICS]")
        print(f"{'Model':<25} {'False Alarm Rate':>18} {'Detection Rate':>16}")
        print("-"*60)
        
        for name, result in self.results.items():
            print(f"{name:<25} {result.false_alarm_rate:>18.2%} "
                  f"{result.detection_rate:>16.2%}")
    
    def evaluate_per_anomaly_type(self,
                                  y_types: np.ndarray,
                                  y_pred: np.ndarray) -> pd.DataFrame:
        """Evaluate detection performance per anomaly type"""
        print("\n[PER-ANOMALY-TYPE DETECTION PERFORMANCE]")
        print("-"*50)
        
        results = []
        unique_types = [t for t in np.unique(y_types) if t != 'normal']
        
        for atype in unique_types:
            mask = y_types == atype
            if mask.sum() > 0:
                detected = y_pred[mask].sum()
                total = mask.sum()
                recall = detected / total
                results.append({
                    'anomaly_type': atype,
                    'total': total,
                    'detected': detected,
                    'recall': recall
                })
                print(f"  {atype:<12}: {detected:>3}/{total:<3} detected ({recall*100:>5.1f}% recall)")
        
        return pd.DataFrame(results)


# ================================================================================
# VISUALIZATION
# ================================================================================

class AnomalyVisualizer:
    """
    Comprehensive visualization for anomaly detection results.
    Creates operator-friendly dashboards and analysis plots.
    """
    
    def __init__(self, config: TelemetryConfig, output_dir: str = '.'):
        self.config = config
        self.output_dir = output_dir
        self.colors = {
            'normal': '#2ecc71',
            'anomaly': '#e74c3c',
            'nominal': '#2ecc71',
            'low': '#f1c40f',
            'medium': '#e67e22',
            'high': '#e74c3c',
            'critical': '#8e44ad'
        }
        
    def plot_telemetry_overview(self,
                                df: pd.DataFrame,
                                y_true: Optional[np.ndarray] = None,
                                save: bool = True) -> plt.Figure:
        """Plot telemetry time-series with anomaly markers"""
        print("\n  Generating telemetry overview...")
        
        fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
        
        for i, col in enumerate(self.config.feature_columns):
            if col not in df.columns:
                continue
                
            ax = axes[i]
            
            # Plot time-series
            ax.plot(df.index, df[col], 'b-', alpha=0.7, linewidth=0.8, label='Telemetry')
            
            # Mark anomalies if labels available
            if y_true is not None and 'label' in df.columns:
                anomaly_mask = df['label'] == 'anomal'
                anomaly_idx = df.index[anomaly_mask]
                ax.scatter(anomaly_idx, df.loc[anomaly_idx, col], 
                          c=self.colors['anomaly'], s=40, zorder=5, 
                          label='True Anomaly', marker='o', edgecolors='darkred')
            
            # Add operational limits
            if col in self.config.operational_limits:
                min_val, max_val = self.config.operational_limits[col]
                ax.axhline(y=min_val, color='orange', linestyle='--', 
                          alpha=0.5, label='Op. Limit')
                ax.axhline(y=max_val, color='orange', linestyle='--', alpha=0.5)
            
            ax.set_ylabel(col.replace('_', '\n'), fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7)
        
        axes[0].set_title('Reaction Wheel Telemetry Overview', 
                         fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Timestamp', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'telemetry_overview.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    Saved: {filepath}")
        
        return fig
    
    def plot_roc_curves(self,
                        evaluator: ModelEvaluator,
                        y_true: np.ndarray,
                        save: bool = True) -> plt.Figure:
        """Plot ROC curves for all models"""
        print("\n  Generating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        
        for (name, result), color in zip(evaluator.results.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, result.scores)
            ax.plot(fpr, tpr, label=f'{name} (AUC={result.roc_auc:.3f})',
                   linewidth=2, color=color)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'roc_curves.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    Saved: {filepath}")
        
        return fig
    
    def plot_confusion_matrices(self,
                               evaluator: ModelEvaluator,
                               y_true: np.ndarray,
                               save: bool = True) -> plt.Figure:
        """Plot confusion matrices for all models"""
        print("\n  Generating confusion matrices...")
        
        n_models = len(evaluator.results)
        fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, result) in zip(axes, evaluator.results.items()):
            cm = confusion_matrix(y_true, result.predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                       annot_kws={'size': 12})
            ax.set_title(name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)
            ax.set_xticklabels(['Normal', 'Anomaly'])
            ax.set_yticklabels(['Normal', 'Anomaly'])
        
        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'confusion_matrices.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    Saved: {filepath}")
        
        return fig
    
    def plot_anomaly_timeline(self,
                             timestamps: pd.DatetimeIndex,
                             ensemble_scores: np.ndarray,
                             y_true: np.ndarray,
                             threshold: float,
                             save: bool = True) -> plt.Figure:
        """Plot ensemble anomaly score timeline"""
        print("\n  Generating anomaly timeline...")
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Plot scores
        ax.plot(timestamps, ensemble_scores, 'b-', alpha=0.7, 
               linewidth=0.8, label='Anomaly Score')
        
        # Threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'Threshold ({threshold:.3f})')
        
        # Mark true anomalies
        anomaly_mask = y_true == 1
        ax.scatter(timestamps[anomaly_mask], ensemble_scores[anomaly_mask],
                  c=self.colors['anomaly'], s=60, zorder=5, 
                  label='True Anomaly', marker='o', edgecolors='darkred')
        
        ax.set_xlabel('Timestamp', fontsize=12)
        ax.set_ylabel('Ensemble Anomaly Score', fontsize=12)
        ax.set_title('Anomaly Score Timeline with Ground Truth', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'anomaly_timeline.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    Saved: {filepath}")
        
        return fig
    
    def plot_severity_dashboard(self,
                               timestamps: pd.DatetimeIndex,
                               scores: np.ndarray,
                               severities: List[AlertSeverity],
                               save: bool = True) -> plt.Figure:
        """Plot operator severity dashboard"""
        print("\n  Generating severity dashboard...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # Severity color mapping
        severity_colors = {
            AlertSeverity.NOMINAL: self.colors['nominal'],
            AlertSeverity.LOW: self.colors['low'],
            AlertSeverity.MEDIUM: self.colors['medium'],
            AlertSeverity.HIGH: self.colors['high'],
            AlertSeverity.CRITICAL: self.colors['critical']
        }
        
        colors = [severity_colors[s] for s in severities]
        
        # Main plot
        axes[0].scatter(timestamps, scores, c=colors, s=25, alpha=0.7)
        
        # Add threshold lines
        for level, thresh in self.config.severity_thresholds.items():
            axes[0].axhline(y=thresh, color='gray', linestyle=':', alpha=0.5)
            axes[0].text(timestamps[-1], thresh, f' {level.upper()}', 
                        fontsize=8, va='center')
        
        axes[0].set_ylabel('Anomaly Score', fontsize=12)
        axes[0].set_title('Operator Alert Dashboard - Severity Classification',
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Legend
        for sev, col in severity_colors.items():
            axes[0].scatter([], [], c=col, label=sev.name, s=60)
        axes[0].legend(loc='upper right', title='Severity', fontsize=9)
        
        # Severity distribution over time (bottom plot)
        severity_nums = [s.value for s in severities]
        axes[1].fill_between(timestamps, severity_nums, alpha=0.5, color='steelblue')
        axes[1].plot(timestamps, severity_nums, 'b-', linewidth=1)
        axes[1].set_xlabel('Timestamp', fontsize=12)
        axes[1].set_ylabel('Severity Level', fontsize=10)
        axes[1].set_yticks([0, 1, 2, 3, 4])
        axes[1].set_yticklabels(['NOMINAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'severity_dashboard.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    Saved: {filepath}")
        
        return fig
    
    def plot_feature_importance(self,
                               feature_names: List[str],
                               X: np.ndarray,
                               scores: np.ndarray,
                               top_n: int = 20,
                               save: bool = True) -> plt.Figure:
        """Plot feature correlation with anomaly scores"""
        print("\n  Generating feature importance plot...")
        
        correlations = []
        for i, fname in enumerate(feature_names[:min(len(feature_names), X.shape[1])]):
            corr = np.corrcoef(X[:, i], scores)[0, 1]
            correlations.append((fname, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = correlations[:top_n]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        names = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
        
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Correlation with Anomaly Score', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Anomaly Correlation',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    Saved: {filepath}")
        
        return fig
    
    def plot_per_type_recall(self,
                            type_results: pd.DataFrame,
                            save: bool = True) -> plt.Figure:
        """Plot detection recall by anomaly type"""
        print("\n  Generating per-type recall plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12'][:len(type_results)]
        
        bars = ax.bar(range(len(type_results)), type_results['recall'], 
                     color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xticks(range(len(type_results)))
        ax.set_xticklabels(type_results['anomaly_type'], fontsize=11)
        ax.set_ylabel('Detection Recall', fontsize=12)
        ax.set_xlabel('Anomaly Type', fontsize=12)
        ax.set_title('Detection Performance by Anomaly Type',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, type_results['recall']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'per_type_recall.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"    Saved: {filepath}")
        
        return fig
    
    def generate_all_visualizations(self,
                                   df: pd.DataFrame,
                                   evaluator: ModelEvaluator,
                                   ensemble_detector: EnsembleAnomalyDetector,
                                   y_true: np.ndarray,
                                   y_types: np.ndarray,
                                   feature_matrix: pd.DataFrame) -> None:
        """Generate all visualization plots"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Get ensemble predictions and details
        results = ensemble_detector.predict(feature_matrix.values, return_details=True)
        
        # 1. Telemetry Overview
        self.plot_telemetry_overview(df, y_true)
        
        # 2. ROC Curves
        self.plot_roc_curves(evaluator, y_true)
        
        # 3. Confusion Matrices
        self.plot_confusion_matrices(evaluator, y_true)
        
        # 4. Anomaly Timeline
        self.plot_anomaly_timeline(
            feature_matrix.index,
            results['ensemble_scores'],
            y_true,
            results['threshold']
        )
        
        # 5. Severity Dashboard
        severities = ensemble_detector.classify_severity(results['ensemble_scores'])
        self.plot_severity_dashboard(
            feature_matrix.index,
            results['ensemble_scores'],
            severities
        )
        
        # 6. Feature Importance
        feature_engineer = FeatureEngineer(self.config)
        feature_engineer.engineer_features(df.copy())
        self.plot_feature_importance(
            feature_engineer.feature_names,
            feature_matrix.values,
            results['ensemble_scores']
        )
        
        # 7. Per-Type Recall
        type_results = evaluator.evaluate_per_anomaly_type(y_types, results['predictions'])
        self.plot_per_type_recall(type_results)
        
        print(f"\n  All visualizations saved to: {self.output_dir}")


# ================================================================================
# MAIN PIPELINE
# ================================================================================

class SatelliteAnomalyPipeline:
    """
    Main orchestrator for the satellite anomaly detection pipeline.
    Coordinates all components from data loading to visualization.
    """
    
    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig()
        self.data_loader = TelemetryDataLoader(self.config)
        self.preprocessor = TelemetryPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.ensemble_detector = EnsembleAnomalyDetector(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.visualizer = AnomalyVisualizer(self.config, output_dir='output')
        
        # Data containers
        self.df_raw = None
        self.df_processed = None
        self.feature_matrix = None
        self.y_true = None
        self.y_types = None
        
    def run(self, data_path: str) -> Dict:
        """Execute the full pipeline"""
        print("\n" + "="*70)
        print("LEO SATELLITE ANOMALY DETECTION SYSTEM")
        print("="*70)
        print(f"Data source: {data_path}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load data
        self.df_raw = self.data_loader.load_data(data_path)
        
        # Step 2: Preprocess
        self.df_processed = self.preprocessor.preprocess(self.df_raw)
        
        # Step 3: Feature engineering
        df_engineered = self.feature_engineer.engineer_features(self.df_processed)
        self.feature_matrix = self.feature_engineer.get_feature_matrix(df_engineered)
        
        # Extract labels
        self.y_true = (df_engineered['label'] == 'anomal').astype(int).values
        self.y_types = df_engineered['anomaly_type'].values
        
        # Step 4: Train ensemble
        X_normal = self.feature_matrix[self.y_true == 0].values
        self.ensemble_detector.build_ensemble()
        self.ensemble_detector.fit(X_normal)
        
        # Step 5: Predict and set threshold
        all_scores = self.ensemble_detector.predict_scores(self.feature_matrix.values)
        ensemble_scores = self.ensemble_detector.compute_ensemble_score(all_scores)
        self.ensemble_detector.set_ensemble_threshold(
            ensemble_scores[self.y_true == 0],  # Use normal data for threshold
            self.config.anomaly_threshold_percentile
        )
        
        # Step 6: Evaluate
        self.evaluator.evaluate_all(self.y_true, self.ensemble_detector, 
                                    self.feature_matrix.values)
        
        # Step 7: Visualize
        self.visualizer.generate_all_visualizations(
            self.df_raw,
            self.evaluator,
            self.ensemble_detector,
            self.y_true,
            self.y_types,
            self.feature_matrix
        )
        
        # Print summary
        self._print_summary()
        
        return {
            'evaluator': self.evaluator,
            'ensemble_detector': self.ensemble_detector,
            'feature_matrix': self.feature_matrix
        }
    
    def _print_summary(self) -> None:
        """Print pipeline execution summary"""
        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        
        print(f"\n[DATASET]")
        print(f"  Total samples: {len(self.df_raw):,}")
        print(f"  Anomaly rate: {self.y_true.mean()*100:.2f}%")
        print(f"  Engineered features: {len(self.feature_engineer.feature_names)}")
        
        print(f"\n[MODELS TRAINED]")
        for name in self.ensemble_detector.models:
            weight = self.config.ensemble_weights.get(name, 0)
            print(f"  - {name} (weight: {weight:.0%})")
        
        print(f"\n[BEST MODEL]")
        best_model = max(self.evaluator.results.items(), 
                        key=lambda x: x[1].f1_score)
        print(f"  {best_model[0]}: F1={best_model[1].f1_score:.4f}, "
              f"AUC={best_model[1].roc_auc:.4f}")
        
        print(f"\n[OUTPUT FILES]")
        print(f"  - telemetry_overview.png")
        print(f"  - roc_curves.png")
        print(f"  - confusion_matrices.png")
        print(f"  - anomaly_timeline.png")
        print(f"  - severity_dashboard.png")
        print(f"  - feature_importance.png")
        print(f"  - per_type_recall.png")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    # Configuration
    config = TelemetryConfig()
    
    # Initialize and run pipeline
    pipeline = SatelliteAnomalyPipeline(config)
    results = pipeline.run('data.csv')
    
    print("\nExecution complete. Check generated visualization files.")
