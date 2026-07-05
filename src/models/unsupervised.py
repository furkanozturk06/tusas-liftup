
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, BatchNormalization, RepeatVector,
        TimeDistributed, Lambda, Concatenate,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    import keras
except ImportError:
    Sequential, Model = None, None

try:
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.hbos import HBOS
    from pyod.models.cblof import CBLOF
    from pyod.models.abod import ABOD
    from pyod.models.cof import COF
    from pyod.models.sod import SOD
    from pyod.models.sos import SOS as SOS_Model
    from pyod.models.loda import LODA
    from pyod.models.inne import INNE
    from pyod.models.lmdd import LMDD
    from pyod.models.so_gaal import SO_GAAL
    from pyod.models.mo_gaal import MO_GAAL
    _PYOD_AVAILABLE = True
except ImportError:
    _PYOD_AVAILABLE = False

try:
    from pyod.models.deep_svdd import DeepSVDD
    _PYOD_DEEPSVDD_AVAILABLE = True
except ImportError:
    _PYOD_DEEPSVDD_AVAILABLE = False

try:
    from pyod.models.lunar import LUNAR as LUNAR_Model
    _PYOD_LUNAR_AVAILABLE = True
except ImportError:
    _PYOD_LUNAR_AVAILABLE = False

try:
    from pyod.models.dif import DIF
    _PYOD_DIF_AVAILABLE = True
except ImportError:
    _PYOD_DIF_AVAILABLE = False

PYOD_MODELS = {"ECOD", "COPOD", "HBOS", "CBLOF", "ABOD", "COF", "SOD", "SOS",
               "LODA", "INNE", "LMDD", "SO_GAAL", "MO_GAAL", "DeepSVDD", "LUNAR", "DIF"}


class UnsupervisedAnomalyDetector:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.thresholds: Dict[str, float] = {}

    def train_isolation_forest(self, X_train: pd.DataFrame, contamination: float = 0.05) -> IsolationForest:
        print("Isolation Forest eğitiliyor...")
        model = IsolationForest(
            n_estimators=200, 
            max_features=1.0, 
            contamination=contamination, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        model.fit(X_train)
        
        scores = -model.score_samples(X_train)
        threshold = np.mean(scores) + 3 * np.std(scores)
        
        self.models['IsolationForest'] = model
        self.thresholds['IsolationForest'] = threshold
        return model

    def train_autoencoder(self, X_train: np.ndarray, X_val: np.ndarray, epochs: int = 50, batch_size: int = 64):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
            
        print("Autoencoder eğitiliyor...")
        input_dim = X_train.shape[1]
        
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu', name='latent_space'),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(input_dim, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        reconstructions = model.predict(X_train)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        threshold = np.mean(mse) + 3 * np.std(mse)
        
        self.models['Autoencoder'] = model
        self.thresholds['Autoencoder'] = float(threshold)
        return model, history

    def train_lstm_autoencoder(self, X_train_seq: np.ndarray, X_val_seq: np.ndarray, seq_len: int, features: int, epochs: int = 20, batch_size: int = 64):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
            
        print("LSTM Autoencoder eğitiliyor...")
        
        model = Sequential([
            Input(shape=(seq_len, features)),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            
            RepeatVector(seq_len),
            
            LSTM(32, return_sequences=True),
            LSTM(64, return_sequences=True),
            TimeDistributed(Dense(features))
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train_seq, X_train_seq,
            validation_data=(X_val_seq, X_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        reconstructions = model.predict(X_train_seq)
        mse = np.mean(np.power(X_train_seq - reconstructions, 2), axis=(1,2))
        threshold = np.mean(mse) + 3 * np.std(mse)
        
        self.models['LSTM_Autoencoder'] = model
        self.thresholds['LSTM_Autoencoder'] = float(threshold)
        return model, history

    def train_one_class_svm(self, X_train: pd.DataFrame, nu: float = 0.05) -> OneClassSVM:
        print("One-Class SVM eğitiliyor...")
        model = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
        model.fit(X_train)
        
        scores = -model.decision_function(X_train)
        threshold = np.percentile(scores, 100 * (1 - nu))
        
        self.models['OneClassSVM'] = model
        self.thresholds['OneClassSVM'] = threshold
        return model

    def train_kmeans(self, X_train: pd.DataFrame, n_clusters: int = 3) -> KMeans:
        print(f"K-Means (K={n_clusters}) eğitiliyor...")
        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
        model.fit(X_train)
        
        distances = model.transform(X_train)
        min_distances = np.min(distances, axis=1)
        threshold = np.mean(min_distances) + 3 * np.std(min_distances)
        
        self.models['KMeans'] = model
        self.thresholds['KMeans'] = threshold
        return model
        
    def train_lof(self, X_train: pd.DataFrame, n_neighbors: int = 20) -> LocalOutlierFactor:
        print("Local Outlier Factor (LOF) eğitiliyor...")
        model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        model.fit(X_train)
        
        scores = -model.score_samples(X_train)
        threshold = np.mean(scores) + 3 * np.std(scores)

        self.models['LOF'] = model
        self.thresholds['LOF'] = threshold
        return model


    def train_gmm(self, X_train: np.ndarray, n_components: int = 3) -> GaussianMixture:
        print("Gaussian Mixture Model eğitiliyor...")
        model = GaussianMixture(n_components=n_components, covariance_type='full',
                                random_state=self.random_state)
        model.fit(X_train)
        scores = -model.score_samples(X_train)
        threshold = np.mean(scores) + 3 * np.std(scores)
        self.models['GMM'] = model
        self.thresholds['GMM'] = float(threshold)
        return model

    def train_elliptic_envelope(self, X_train: np.ndarray, contamination: float = 0.05) -> EllipticEnvelope:
        print("Elliptic Envelope eğitiliyor...")
        model = EllipticEnvelope(contamination=contamination, random_state=self.random_state)
        model.fit(X_train)
        scores = -model.score_samples(X_train)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        self.models['EllipticEnvelope'] = model
        self.thresholds['EllipticEnvelope'] = float(threshold)
        return model

    def train_pca(self, X_train: np.ndarray, n_components: float = 0.95) -> PCA:
        print("PCA (reconstruction error) eğitiliyor...")
        model = PCA(n_components=n_components, random_state=self.random_state)
        model.fit(X_train)
        recon = model.inverse_transform(model.transform(X_train))
        scores = np.mean(np.power(X_train - recon, 2), axis=1)
        threshold = np.mean(scores) + 3 * np.std(scores)
        self.models['PCA'] = model
        self.thresholds['PCA'] = float(threshold)
        return model

    def train_dbscan(self, X_train: np.ndarray, eps: float = 1.5, min_samples: int = 5):
        print("DBSCAN (core-distance novelty) eğitiliyor...")
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X_train)
        core = X_train[labels != -1]
        if len(core) == 0:
            core = X_train
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(core)
        nbrs2 = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(core)
        dist2, _ = nbrs2.kneighbors(X_train)
        scores = dist2[:, 1]
        threshold = float(np.mean(scores) + 3 * np.std(scores))
        self.models['DBSCAN'] = nbrs
        self.thresholds['DBSCAN'] = threshold
        return nbrs

    def train_vae(self, X_train: np.ndarray, X_val: np.ndarray, latent_dim: int = 8,
                  epochs: int = 50, batch_size: int = 64, beta: float = 1.0):
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("Variational Autoencoder (VAE) eğitiliyor...")
        input_dim = X_train.shape[1]

        class SamplingLayer(keras.layers.Layer):
            def __init__(self, beta=1.0, **kwargs):
                super().__init__(**kwargs)
                self._beta = beta

            def call(self, inputs):
                z_mean, z_log_var = inputs
                eps = keras.random.normal(shape=keras.ops.shape(z_mean))
                kl = -0.5 * self._beta * keras.ops.mean(
                    keras.ops.sum(
                        1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var),
                        axis=1,
                    )
                )
                self.add_loss(kl)
                return z_mean + keras.ops.exp(0.5 * z_log_var) * eps

        inputs = Input(shape=(input_dim,))
        h = Dense(64, activation='relu')(inputs)
        h = BatchNormalization()(h)
        h = Dense(32, activation='relu')(h)
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        z = SamplingLayer(beta=beta, name='z')([z_mean, z_log_var])

        d = Dense(32, activation='relu')(z)
        d = Dense(64, activation='relu')(d)
        outputs = Dense(input_dim, activation='linear')(d)

        vae = Model(inputs, outputs, name='VAE')
        vae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        vae.fit(X_train, X_train, validation_data=(X_val, X_val),
                epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)

        recon = vae.predict(X_train, verbose=0)
        scores = np.mean(np.power(X_train - recon, 2), axis=1)
        threshold = np.mean(scores) + 3 * np.std(scores)
        self.models['VAE'] = vae
        self.thresholds['VAE'] = float(threshold)
        return vae


    def train_pyod(self, X_train: np.ndarray, contamination: float = 0.05) -> Dict[str, Any]:
        if not _PYOD_AVAILABLE:
            print("PyOD kurulu değil — atlanıyor (pip install pyod).")
            return {}
        print("PyOD dedektörleri (ECOD, COPOD, HBOS, CBLOF) eğitiliyor...")
        detectors = {
            'ECOD':  ECOD(contamination=contamination),
            'COPOD': COPOD(contamination=contamination),
            'HBOS':  HBOS(contamination=contamination),
            'CBLOF': CBLOF(contamination=contamination, random_state=self.random_state),
        }
        trained = {}
        for name, det in detectors.items():
            det.fit(X_train)
            self.models[name] = det
            self.thresholds[name] = float(det.threshold_)
            trained[name] = det
        return trained


    def train_abod(self, X_train: np.ndarray, contamination: float = 0.05):
        if not _PYOD_AVAILABLE:
            raise ImportError("PyOD bulunamadı (pip install pyod).")
        print("ABOD eğitiliyor...")
        model = ABOD(contamination=contamination)
        model.fit(X_train)
        self.models['ABOD'] = model
        self.thresholds['ABOD'] = float(model.threshold_)
        return model

    def train_cof(self, X_train: np.ndarray, contamination: float = 0.05, n_neighbors: int = 20):
        if not _PYOD_AVAILABLE:
            raise ImportError("PyOD bulunamadı (pip install pyod).")
        print("COF eğitiliyor...")
        model = COF(contamination=contamination, n_neighbors=n_neighbors)
        model.fit(X_train)
        self.models['COF'] = model
        self.thresholds['COF'] = float(model.threshold_)
        return model

    def train_sod(self, X_train: np.ndarray, contamination: float = 0.05, n_neighbors: int = 20):
        if not _PYOD_AVAILABLE:
            raise ImportError("PyOD bulunamadı (pip install pyod).")
        print("SOD eğitiliyor...")
        model = SOD(contamination=contamination, n_neighbors=n_neighbors)
        model.fit(X_train)
        self.models['SOD'] = model
        self.thresholds['SOD'] = float(model.threshold_)
        return model

    def train_sos(self, X_train: np.ndarray, contamination: float = 0.05):
        if not _PYOD_AVAILABLE:
            raise ImportError("PyOD bulunamadı (pip install pyod).")
        print("SOS eğitiliyor...")
        model = SOS_Model(contamination=contamination)
        model.fit(X_train)
        self.models['SOS'] = model
        self.thresholds['SOS'] = float(model.threshold_)
        return model

    def train_loda(self, X_train: np.ndarray, contamination: float = 0.05):
        if not _PYOD_AVAILABLE:
            raise ImportError("PyOD bulunamadı (pip install pyod).")
        print("LODA eğitiliyor...")
        model = LODA(contamination=contamination)
        model.fit(X_train)
        self.models['LODA'] = model
        self.thresholds['LODA'] = float(model.threshold_)
        return model

    def train_inne(self, X_train: np.ndarray, contamination: float = 0.05):
        if not _PYOD_AVAILABLE:
            raise ImportError("PyOD bulunamadı (pip install pyod).")
        print("INNE eğitiliyor...")
        model = INNE(contamination=contamination, random_state=self.random_state)
        model.fit(X_train)
        self.models['INNE'] = model
        self.thresholds['INNE'] = float(model.threshold_)
        return model

    def train_lmdd(self, X_train: np.ndarray, contamination: float = 0.05):
        if not _PYOD_AVAILABLE:
            raise ImportError("PyOD bulunamadı (pip install pyod).")
        print("LMDD eğitiliyor...")
        model = LMDD(contamination=contamination, random_state=self.random_state)
        model.fit(X_train)
        self.models['LMDD'] = model
        self.thresholds['LMDD'] = float(model.threshold_)
        return model

    def train_so_gaal(self, X_train: np.ndarray, contamination: float = 0.05):
        if not _PYOD_AVAILABLE:
            raise ImportError("PyOD bulunamadı (pip install pyod).")
        print("SO-GAAL eğitiliyor...")
        model = SO_GAAL(contamination=contamination)
        model.fit(X_train)
        self.models['SO_GAAL'] = model
        self.thresholds['SO_GAAL'] = float(model.threshold_)
        return model

    def train_mo_gaal(self, X_train: np.ndarray, contamination: float = 0.05):
        if not _PYOD_AVAILABLE:
            raise ImportError("PyOD bulunamadı (pip install pyod).")
        print("MO-GAAL eğitiliyor...")
        model = MO_GAAL(contamination=contamination)
        model.fit(X_train)
        self.models['MO_GAAL'] = model
        self.thresholds['MO_GAAL'] = float(model.threshold_)
        return model


    def train_deep_svdd(self, X_train: np.ndarray, contamination: float = 0.05, epochs: int = 50):
        if not _PYOD_DEEPSVDD_AVAILABLE:
            raise ImportError("PyOD DeepSVDD bulunamadı (pip install pyod torch).")
        print("DeepSVDD eğitiliyor...")
        model = DeepSVDD(n_features=X_train.shape[1], contamination=contamination,
                         epochs=epochs, random_state=self.random_state)
        model.fit(X_train)
        self.models['DeepSVDD'] = model
        self.thresholds['DeepSVDD'] = float(model.threshold_)
        return model

    def train_lunar(self, X_train: np.ndarray, contamination: float = 0.05):
        if not _PYOD_LUNAR_AVAILABLE:
            raise ImportError("PyOD LUNAR bulunamadı (pip install pyod torch torch_geometric).")
        print("LUNAR eğitiliyor...")
        model = LUNAR_Model(contamination=contamination)
        model.fit(X_train)
        self.models['LUNAR'] = model
        self.thresholds['LUNAR'] = float(model.threshold_)
        return model

    def train_dif(self, X_train: np.ndarray, contamination: float = 0.05, epochs: int = 50):
        if not _PYOD_DIF_AVAILABLE:
            raise ImportError("PyOD DIF bulunamadı (pip install pyod torch).")
        print("DIF (Deep Isolation Forest) eğitiliyor...")
        model = DIF(contamination=contamination, epochs=epochs, random_state=self.random_state)
        model.fit(X_train)
        self.models['DIF'] = model
        self.thresholds['DIF'] = float(model.threshold_)
        return model


    def train_anogan(self, X_train: np.ndarray, X_val: np.ndarray, latent_dim: int = 32,
                     epochs: int = 100, batch_size: int = 64):
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("AnoGAN eğitiliyor...")
        input_dim = X_train.shape[1]

        generator = Sequential([
            Input(shape=(latent_dim,)),
            Dense(64, activation='relu'), BatchNormalization(),
            Dense(128, activation='relu'), BatchNormalization(),
            Dense(input_dim, activation='linear')
        ], name='anogan_gen')

        discriminator = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'), Dropout(0.3),
            Dense(64, activation='relu'), Dropout(0.3),
            Dense(1, activation='sigmoid')
        ], name='anogan_disc')

        encoder = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'), BatchNormalization(),
            Dense(64, activation='relu'),
            Dense(latent_dim, activation='linear')
        ], name='anogan_enc')

        d_opt = Adam(learning_rate=0.0002, beta_1=0.5)
        g_opt = Adam(learning_rate=0.0002, beta_1=0.5)
        e_opt = Adam(learning_rate=0.001)
        bce = tf.keras.losses.BinaryCrossentropy()

        dataset = tf.data.Dataset.from_tensor_slices(
            X_train.astype('float32')).shuffle(len(X_train)).batch(batch_size)

        for epoch in range(epochs):
            for real_batch in dataset:
                bs = tf.shape(real_batch)[0]
                noise = tf.random.normal((bs, latent_dim))
                with tf.GradientTape() as d_tape:
                    fake = generator(noise, training=True)
                    real_out = discriminator(real_batch, training=True)
                    fake_out = discriminator(fake, training=True)
                    d_loss = bce(tf.ones_like(real_out), real_out) + \
                             bce(tf.zeros_like(fake_out), fake_out)
                d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
                d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

                noise = tf.random.normal((bs, latent_dim))
                with tf.GradientTape() as g_tape:
                    fake = generator(noise, training=True)
                    fake_out = discriminator(fake, training=True)
                    g_loss = bce(tf.ones_like(fake_out), fake_out)
                g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
                g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

        generator.trainable = False
        enc_ds = tf.data.Dataset.from_tensor_slices(
            X_train.astype('float32')).shuffle(len(X_train)).batch(batch_size)
        for epoch in range(max(epochs // 2, 20)):
            for real_batch in enc_ds:
                with tf.GradientTape() as e_tape:
                    z = encoder(real_batch, training=True)
                    recon = generator(z, training=False)
                    e_loss = tf.reduce_mean(tf.square(real_batch - recon))
                e_grads = e_tape.gradient(e_loss, encoder.trainable_variables)
                e_opt.apply_gradients(zip(e_grads, encoder.trainable_variables))

        score_input = Input(shape=(input_dim,))
        anogan_model = Model(score_input, generator(encoder(score_input)), name='AnoGAN')

        recon = anogan_model.predict(X_train, verbose=0)
        scores = np.mean(np.square(X_train - recon), axis=1)
        threshold = float(np.mean(scores) + 3 * np.std(scores))

        self.models['AnoGAN'] = anogan_model
        self.thresholds['AnoGAN'] = threshold
        return anogan_model

    def train_alad(self, X_train: np.ndarray, X_val: np.ndarray, latent_dim: int = 32,
                   epochs: int = 100, batch_size: int = 64):
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("ALAD eğitiliyor...")
        input_dim = X_train.shape[1]

        enc = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'), BatchNormalization(),
            Dense(64, activation='relu'),
            Dense(latent_dim, activation='linear')
        ], name='alad_enc')

        gen = Sequential([
            Input(shape=(latent_dim,)),
            Dense(64, activation='relu'), BatchNormalization(),
            Dense(128, activation='relu'), BatchNormalization(),
            Dense(input_dim, activation='linear')
        ], name='alad_gen')

        x_in = Input(shape=(input_dim,))
        z_in = Input(shape=(latent_dim,))
        xz = Concatenate()([x_in, z_in])
        d = Dense(128, activation='relu')(xz)
        d = Dropout(0.3)(d)
        d = Dense(64, activation='relu')(d)
        d = Dropout(0.3)(d)
        d_out = Dense(1, activation='sigmoid')(d)
        disc = Model([x_in, z_in], d_out, name='alad_disc')

        d_opt = Adam(learning_rate=0.0002, beta_1=0.5)
        ge_opt = Adam(learning_rate=0.0002, beta_1=0.5)
        bce = tf.keras.losses.BinaryCrossentropy()

        dataset = tf.data.Dataset.from_tensor_slices(
            X_train.astype('float32')).shuffle(len(X_train)).batch(batch_size)
        enc_gen_vars = enc.trainable_variables + gen.trainable_variables

        for epoch in range(epochs):
            for real_batch in dataset:
                bs = tf.shape(real_batch)[0]
                noise = tf.random.normal((bs, latent_dim))

                with tf.GradientTape() as d_tape:
                    z_enc = enc(real_batch, training=True)
                    x_gen = gen(noise, training=True)
                    real_pair = disc([real_batch, z_enc], training=True)
                    fake_pair = disc([x_gen, noise], training=True)
                    d_loss = bce(tf.ones_like(real_pair), real_pair) + \
                             bce(tf.zeros_like(fake_pair), fake_pair)
                d_grads = d_tape.gradient(d_loss, disc.trainable_variables)
                d_opt.apply_gradients(zip(d_grads, disc.trainable_variables))

                noise = tf.random.normal((bs, latent_dim))
                with tf.GradientTape() as ge_tape:
                    z_enc = enc(real_batch, training=True)
                    x_gen = gen(noise, training=True)
                    real_pair = disc([real_batch, z_enc], training=True)
                    fake_pair = disc([x_gen, noise], training=True)
                    x_recon = gen(z_enc, training=True)
                    cycle_loss = tf.reduce_mean(tf.square(real_batch - x_recon))
                    ge_loss = bce(tf.zeros_like(real_pair), real_pair) + \
                              bce(tf.ones_like(fake_pair), fake_pair) + cycle_loss
                ge_grads = ge_tape.gradient(ge_loss, enc_gen_vars)
                ge_opt.apply_gradients(zip(ge_grads, enc_gen_vars))

        score_input = Input(shape=(input_dim,))
        alad_model = Model(score_input, gen(enc(score_input)), name='ALAD')

        recon = alad_model.predict(X_train, verbose=0)
        scores = np.mean(np.square(X_train - recon), axis=1)
        threshold = float(np.mean(scores) + 3 * np.std(scores))

        self.models['ALAD'] = alad_model
        self.thresholds['ALAD'] = threshold
        return alad_model

    def compute_ensemble_score(self, X_test: np.ndarray, active_models: List[str] = None) -> np.ndarray:
        if active_models is None:
            active_models = ['IsolationForest', 'Autoencoder', 'OneClassSVM', 'KMeans', 'LOF']
            
        print(f"Ensemble Anomali Skoru Hesaplanıyor ({len(active_models)} model)...")
        scores_matrix = []
        
        for name in active_models:
            if name not in self.models:
                continue
                
            model = self.models[name]
            if name == 'IsolationForest' or name == 'LOF':
                scores = -model.score_samples(X_test)
            elif name == 'OneClassSVM':
                scores = -model.decision_function(X_test)
            elif name in ('Autoencoder', 'AnoGAN', 'ALAD'):
                recon = model.predict(X_test, verbose=0)
                scores = np.mean(np.power(X_test - recon, 2), axis=1)
            elif name == 'KMeans':
                dist = model.transform(X_test)
                scores = np.min(dist, axis=1)
            elif name in PYOD_MODELS and hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
            else:
                continue
                
            scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
            scores_matrix.append(scores_norm)
            
        ensemble_score = np.mean(np.array(scores_matrix), axis=0)
        return ensemble_score

    def detect_anomalies(self, ensemble_score: np.ndarray, global_threshold: float = 0.5) -> np.ndarray:
        return (ensemble_score > global_threshold).astype(int)

    def save_models(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(path, f"{name.lower()}_model")
            if name in ['Autoencoder', 'LSTM_Autoencoder', 'VAE', 'AnoGAN', 'ALAD']:
                model.save(filepath + ".keras")
            else:
                joblib.dump(model, filepath + ".joblib")
                
        with open(os.path.join(path, "unsupervised_thresholds.json"), "w", encoding='utf-8') as f:
            json.dump(self.thresholds, f, indent=4)
        print("Gözetimsiz modeller başarıyla kaydedildi.")
