
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from pyod.models.xgbod import XGBOD as PyOD_XGBOD
    _XGBOD_AVAILABLE = True
except ImportError:
    PyOD_XGBOD = None
    _XGBOD_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
        Dense, Dropout, BatchNormalization, LayerNormalization, Input,
        MultiHeadAttention, Add, Activation, SpatialDropout1D,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    Sequential = None
    Model = None


SEQUENCE_MODELS = {"LSTM", "BiLSTM", "GRU", "BiGRU", "CNN1D", "CNN_LSTM", "CNN_BiLSTM",
                   "CNN_GRU", "Transformer", "TCN", "Attention_BiLSTM",
                   "FCN", "ResNet1D", "InceptionTime", "LSTM_FCN"}
KERAS_MODELS = SEQUENCE_MODELS | {"MLP"}


class SupervisedAnomalyDetector:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
        self.metrics: Dict[str, Dict[str, float]] = {}

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, tune: bool = False) -> RandomForestClassifier:
        print("Random Forest eğitiliyor...")
        if tune:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            rf = RandomForestClassifier(class_weight='balanced', random_state=self.random_state, n_jobs=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(rf, param_grid, cv=tscv, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"En iyi parametreler: {grid.best_params_}")
        else:
            model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', 
                                         random_state=self.random_state, n_jobs=-1)
            model.fit(X_train, y_train)

        self.models['RandomForest'] = model
        return model

    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series, kernel: str = 'rbf') -> CalibratedClassifierCV:
        print(f"SVM ({kernel} kernel) eğitiliyor...")
        base_svm = SVC(kernel=kernel, class_weight='balanced', random_state=self.random_state, max_iter=5000)
        model = CalibratedClassifierCV(base_svm, cv=3)
        model.fit(X_train, y_train)
        
        self.models['SVM'] = model
        return model

    def train_lsvc(self, X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
        print("Linear SVC eğitiliyor...")
        base = LinearSVC(class_weight='balanced', max_iter=5000, random_state=self.random_state)
        model = CalibratedClassifierCV(base, cv=3)
        model.fit(X_train, y_train)
        self.models['LSVC'] = model
        return model

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        if xgb is None:
            raise ImportError("XGBoost kütüphanesi bulunamadı.")
            
        print("XGBoost eğitiliyor...")
        
        neg_count = sum(y_train == 0)
        pos_count = sum(y_train == 1)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_weight,
            early_stopping_rounds=50,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='auc'
        )

        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.set_params(early_stopping_rounds=None)
            model.fit(X_train, y_train)

        self.models['XGBoost'] = model
        return model

    def train_xgbod(self, X_train: pd.DataFrame, y_train: pd.Series):
        if not _XGBOD_AVAILABLE:
            raise ImportError("PyOD/XGBOD bulunamadı (pip install pyod).")
        print("XGBOD eğitiliyor...")
        model = PyOD_XGBOD(random_state=self.random_state)
        model.fit(np.asarray(X_train), np.asarray(y_train))
        self.models['XGBOD'] = model
        return model

    def train_mlp(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                   epochs: int = 50, batch_size: int = 64):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
            
        print("MLP eğitiliyor...")
        
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        self.models['MLP'] = model
        return model, history


    def train_extra_trees(self, X_train, y_train) -> ExtraTreesClassifier:
        print("Extra Trees eğitiliyor...")
        model = ExtraTreesClassifier(
            n_estimators=300, max_depth=None, class_weight='balanced',
            random_state=self.random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['ExtraTrees'] = model
        return model

    def train_gradient_boosting(self, X_train, y_train) -> GradientBoostingClassifier:
        print("Gradient Boosting eğitiliyor...")
        model = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['GradientBoosting'] = model
        return model

    def train_adaboost(self, X_train, y_train) -> AdaBoostClassifier:
        print("AdaBoost eğitiliyor...")
        model = AdaBoostClassifier(
            n_estimators=300, learning_rate=0.5,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['AdaBoost'] = model
        return model

    def train_knn(self, X_train, y_train, n_neighbors: int = 15) -> KNeighborsClassifier:
        print(f"KNN (k={n_neighbors}) eğitiliyor...")
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)
        model.fit(X_train, y_train)
        self.models['KNN'] = model
        return model

    def train_logistic_regression(self, X_train, y_train) -> LogisticRegression:
        print("Logistic Regression eğitiliyor...")
        model = LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['LogisticRegression'] = model
        return model

    def train_hist_gradient_boosting(self, X_train, y_train) -> HistGradientBoostingClassifier:
        print("HistGradientBoosting eğitiliyor...")
        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None,
            class_weight='balanced', random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['HistGradientBoosting'] = model
        return model

    def train_decision_tree(self, X_train, y_train) -> DecisionTreeClassifier:
        print("Decision Tree eğitiliyor...")
        model = DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, class_weight='balanced',
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['DecisionTree'] = model
        return model

    def train_naive_bayes(self, X_train, y_train) -> GaussianNB:
        print("Gaussian Naive Bayes eğitiliyor...")
        model = GaussianNB()
        model.fit(X_train, y_train)
        self.models['NaiveBayes'] = model
        return model

    def train_voting(self, X_train, y_train) -> VotingClassifier:
        print("Voting Ensemble (soft) eğitiliyor...")
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                          random_state=self.random_state, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                              random_state=self.random_state)),
            ('lr', LogisticRegression(max_iter=2000, class_weight='balanced',
                                      random_state=self.random_state)),
        ]
        model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        model.fit(X_train, y_train)
        self.models['Voting Ensemble'] = model
        return model

    def train_lda(self, X_train, y_train) -> LinearDiscriminantAnalysis:
        print("LDA (Linear Discriminant Analysis) eğitiliyor...")
        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        self.models['LDA'] = model
        return model

    def train_qda(self, X_train, y_train) -> QuadraticDiscriminantAnalysis:
        print("QDA (Quadratic Discriminant Analysis) eğitiliyor...")
        model = QuadraticDiscriminantAnalysis(reg_param=0.1)
        model.fit(X_train, y_train)
        self.models['QDA'] = model
        return model

    def train_bagging(self, X_train, y_train) -> BaggingClassifier:
        print("Bagging eğitiliyor...")
        model = BaggingClassifier(
            n_estimators=200, max_samples=0.8, max_features=0.8,
            random_state=self.random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Bagging'] = model
        return model

    def train_ridge(self, X_train, y_train) -> RidgeClassifier:
        print("Ridge Classifier eğitiliyor...")
        model = RidgeClassifier(class_weight='balanced', random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models['Ridge'] = model
        return model

    def train_sgd(self, X_train, y_train) -> SGDClassifier:
        print("SGD Classifier (log-loss) eğitiliyor...")
        model = SGDClassifier(
            loss='log_loss', class_weight='balanced', max_iter=2000,
            random_state=self.random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['SGD'] = model
        return model


    @staticmethod
    def _reshape_seq(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype='float32')
        if X.ndim == 2:
            return X.reshape((X.shape[0], X.shape[1], 1))
        return X

    def _fit_keras_sequence(self, name: str, model, X_train, y_train, X_val, y_val,
                            epochs: int, batch_size: int, patience: int = 12):
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        Xtr, Xvl = self._reshape_seq(X_train), self._reshape_seq(X_val)
        ytr = np.asarray(y_train).astype('float32').ravel()
        yvl = np.asarray(y_val).astype('float32').ravel()

        history = model.fit(
            Xtr, ytr,
            validation_data=(Xvl, yvl),
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stop], verbose=1
        )
        self.models[name] = model
        return model, history

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("LSTM eğitiliyor...")
        n_features = X_train.shape[1]
        model = Sequential([
            Input(shape=(n_features, 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ], name='LSTM')
        return self._fit_keras_sequence('LSTM', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_bilstm(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("BiLSTM eğitiliyor...")
        n_features = X_train.shape[1]
        model = Sequential([
            Input(shape=(n_features, 1)),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ], name='BiLSTM')
        return self._fit_keras_sequence('BiLSTM', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_gru(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("GRU eğitiliyor...")
        n_features = X_train.shape[1]
        model = Sequential([
            Input(shape=(n_features, 1)),
            GRU(64, return_sequences=True),
            Dropout(0.3),
            GRU(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ], name='GRU')
        return self._fit_keras_sequence('GRU', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_cnn_bilstm(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("CNN-BiLSTM (hibrit) eğitiliyor...")
        n_features = X_train.shape[1]
        model = Sequential([
            Input(shape=(n_features, 1)),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ], name='CNN_BiLSTM')
        return self._fit_keras_sequence('CNN_BiLSTM', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_cnn_gru(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("CNN-GRU (hibrit) eğitiliyor...")
        n_features = X_train.shape[1]
        model = Sequential([
            Input(shape=(n_features, 1)),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            GRU(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ], name='CNN_GRU')
        return self._fit_keras_sequence('CNN_GRU', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_transformer(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("Transformer (self-attention) eğitiliyor...")
        n_features = X_train.shape[1]
        head_size, num_heads, ff_dim = 64, 4, 128

        inputs = Input(shape=(n_features, 1))
        x = Conv1D(filters=head_size, kernel_size=1, padding='same')(inputs)

        attn = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=0.1)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization(epsilon=1e-6)(x)
        ff = Dense(ff_dim, activation='relu')(x)
        ff = Dropout(0.1)(ff)
        ff = Dense(head_size)(ff)
        x = Add()([x, ff])
        x = LayerNormalization(epsilon=1e-6)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs, name='Transformer')
        return self._fit_keras_sequence('Transformer', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_tcn(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("TCN (Temporal Conv Network) eğitiliyor...")
        n_features = X_train.shape[1]

        inputs = Input(shape=(n_features, 1))
        x = inputs
        for dilation in (1, 2, 4, 8):
            prev = x
            x = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=dilation,
                       activation='relu')(x)
            x = BatchNormalization()(x)
            x = SpatialDropout1D(0.1)(x)
            if prev.shape[-1] != x.shape[-1]:
                prev = Conv1D(64, kernel_size=1, padding='same')(prev)
            x = Add()([prev, x])

        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs, name='TCN')
        return self._fit_keras_sequence('TCN', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_cnn1d(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("1D-CNN eğitiliyor...")
        n_features = X_train.shape[1]
        model = Sequential([
            Input(shape=(n_features, 1)),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ], name='CNN1D')
        return self._fit_keras_sequence('CNN1D', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_bigru(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("BiGRU eğitiliyor...")
        n_features = X_train.shape[1]
        model = Sequential([
            Input(shape=(n_features, 1)),
            Bidirectional(GRU(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(GRU(32, return_sequences=False)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ], name='BiGRU')
        return self._fit_keras_sequence('BiGRU', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_cnn_lstm(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("CNN-LSTM (hibrit) eğitiliyor...")
        n_features = X_train.shape[1]
        model = Sequential([
            Input(shape=(n_features, 1)),
            Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ], name='CNN_LSTM')
        return self._fit_keras_sequence('CNN_LSTM', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_attention_bilstm(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("Attention-BiLSTM eğitiliyor...")
        n_features = X_train.shape[1]

        inputs = Input(shape=(n_features, 1))
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization(epsilon=1e-6)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs, name='Attention_BiLSTM')
        return self._fit_keras_sequence('Attention_BiLSTM', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_fcn(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("FCN (Fully Convolutional Network) eğitiliyor...")
        n_features = X_train.shape[1]
        model = Sequential([
            Input(shape=(n_features, 1)),
            Conv1D(128, kernel_size=8, padding='same'), BatchNormalization(), Activation('relu'),
            Conv1D(256, kernel_size=5, padding='same'), BatchNormalization(), Activation('relu'),
            Conv1D(128, kernel_size=3, padding='same'), BatchNormalization(), Activation('relu'),
            GlobalAveragePooling1D(),
            Dense(1, activation='sigmoid')
        ], name='FCN')
        return self._fit_keras_sequence('FCN', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_resnet1d(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("ResNet-1D eğitiliyor...")
        n_features = X_train.shape[1]

        def residual_block(x, filters):
            shortcut = x
            for k in (8, 5, 3):
                x = Conv1D(filters, kernel_size=k, padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            if shortcut.shape[-1] != filters:
                shortcut = Conv1D(filters, kernel_size=1, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
            x = Add()([shortcut, x])
            return Activation('relu')(x)

        inputs = Input(shape=(n_features, 1))
        x = residual_block(inputs, 64)
        x = residual_block(x, 128)
        x = residual_block(x, 128)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs, name='ResNet1D')
        return self._fit_keras_sequence('ResNet1D', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_inceptiontime(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("InceptionTime eğitiliyor...")
        n_features = X_train.shape[1]
        nb_filters = 32

        def inception_module(x):
            bottleneck = Conv1D(nb_filters, kernel_size=1, padding='same', use_bias=False)(x)
            convs = [Conv1D(nb_filters, kernel_size=k, padding='same', use_bias=False)(bottleneck)
                     for k in (10, 20, 40)]
            pool = MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
            pool = Conv1D(nb_filters, kernel_size=1, padding='same', use_bias=False)(pool)
            from tensorflow.keras.layers import Concatenate
            out = Concatenate(axis=2)(convs + [pool])
            out = BatchNormalization()(out)
            return Activation('relu')(out)

        inputs = Input(shape=(n_features, 1))
        x = inputs
        residual = inputs
        for d in range(6):
            x = inception_module(x)
            if d % 3 == 2:
                res = Conv1D(int(x.shape[-1]), kernel_size=1, padding='same', use_bias=False)(residual)
                res = BatchNormalization()(res)
                x = Activation('relu')(Add()([res, x]))
                residual = x
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs, name='InceptionTime')
        return self._fit_keras_sequence('InceptionTime', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def train_lstm_fcn(self, X_train, y_train, X_val, y_val, epochs: int = 60, batch_size: int = 32):
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("LSTM-FCN eğitiliyor...")
        n_features = X_train.shape[1]
        inputs = Input(shape=(n_features, 1))

        c = Conv1D(128, kernel_size=8, padding='same')(inputs); c = BatchNormalization()(c); c = Activation('relu')(c)
        c = Conv1D(256, kernel_size=5, padding='same')(c); c = BatchNormalization()(c); c = Activation('relu')(c)
        c = Conv1D(128, kernel_size=3, padding='same')(c); c = BatchNormalization()(c); c = Activation('relu')(c)
        c = GlobalAveragePooling1D()(c)

        l = LSTM(64)(inputs)
        l = Dropout(0.3)(l)

        from tensorflow.keras.layers import Concatenate
        x = Concatenate()([c, l])
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs, name='LSTM_FCN')
        return self._fit_keras_sequence('LSTM_FCN', model, X_train, y_train, X_val, y_val, epochs, batch_size)

    def evaluate_model(self, name: str, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        if name not in self.models:
            raise ValueError(f"Model bulunamadı: {name}")
            
        model = self.models[name]

        if name in SEQUENCE_MODELS:
            y_pred_prob = model.predict(self._reshape_seq(X_test), verbose=0).flatten()
        elif name == 'MLP':
            y_pred_prob = model.predict(X_test, verbose=0).flatten()
        elif hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred_prob = model.decision_function(X_test)
        else:
            y_pred_prob = model.predict(X_test)

        y_pred = (y_pred_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1score = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
        except ValueError:
            auc = 0.5
            
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'AUC': auc,
            'FAR': far
        }
        
        self.metrics[name] = metrics
        return metrics

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        for name in self.models.keys():
            self.evaluate_model(name, X_test, y_test)
                
        df_metrics = pd.DataFrame(self.metrics).T
        if not df_metrics.empty:
            self.best_model_name = df_metrics['F1_Score'].idxmax()
        return df_metrics

    def save_model(self, name: str, filepath: str):
        if name not in self.models:
            raise ValueError(f"Model bulunamadı: {name}")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if name in KERAS_MODELS:
            self.models[name].save(filepath)
        else:
            joblib.dump(self.models[name], filepath)
        print(f"Model kaydedildi: {filepath}")

    def save_metadata(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "best_model": self.best_model_name,
            "metrics": self.metrics
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Metadata kaydedildi: {filepath}")
