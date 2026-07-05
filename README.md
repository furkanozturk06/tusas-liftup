# 🛰️ Uydu Telemetri Anomali Tespiti (ESA OPS-SAT)

> ESA OPS-SAT uydu telemetrisinde, **tek bir kanonik eğitim motoruyla** resmi test
> seti (Ψ) üzerinde değerlendirilen, sızıntısız ve literatürle birebir
> karşılaştırılabilir bir anomali tespiti çalışması. 46 modellik bir envanter,
> Plotly Dash arayüzü, SHAP yorumlanabilirliği, güç/maliyet analizi ve sentetik veri
> üretimi içerir.

---

## 📋 Proje Özeti

Çalışma, OPS-SAT-AD veri setinde segment başına **18 elle-üretilmiş ESA özelliği**
çıkarır ve modelleri `dataset.csv`'deki `train` kolonuyla tanımlı **resmi bölme**
üzerinde değerlendirir (T = 1594 eğitim, Ψ = 529 test). Tüm modeller tek bir kanonik
kaynak (`train_all_models.py`) tarafından aynı özellik, aynı ölçekleyici ve aynı
bölmeyle eğitilir; böylece "iki ayrı boru hattının birbirini ezmesi" türü sızıntılar
önlenir. Sonuçlar Ruszczak et al. (2024) OPS-SAT baseline'ı ile karşılaştırılır
(gözetimli tarafta ortalama **|ΔAUC-PR| = 0,004**, en iyi model **ExtraTrees,
AUC-PR 0,983**).

### Model envanteri (46)

| Kategori | Sayı | Açıklama |
|---|---|---|
| **Gözetimli (tabular)** | 23 | RF, ExtraTrees, XGBoost, XGBOD, LightGBM, CatBoost, GBM, HistGBM, AdaBoost, Bagging, DecisionTree, KNN, SVM, LSVC, MLP, LogReg, Ridge, SGD, LDA, QDA, NaiveBayes, Voting, Stacking |
| **Gözetimsiz (tabular)** | 19 | IsolationForest, OneClassSVM, LOF, KMeans, GMM, EllipticEnvelope, PCA, DBSCAN, ECOD, COPOD, HBOS, CBLOF, ABOD, COF, SOD, SOS, LODA, INNE, LMDD |
| **Derin sıralı (ham sinyal)** | 2 | CNN1D, TCN — 18 özellik yerine her segmentin ham örnek dizisini girdi alır (NB04 Bölüm 9) |
| **ESA-ADB literatür baseline** | 2 | Telemanom-ESA, DC-VAE-ESA — ayrı benchmark'tan referans; OPS-SAT Ψ'de ölçülmez |

Bunlardan **44'ü** (42 tabular + 2 derin) resmi Ψ test setinde fiilen ölçülür;
2 ESA-ADB baseline'ı yalnızca literatür referansıdır. Tam liste ve kaynak
atıfları için **[`models_trained.txt`](models_trained.txt)**'ye bakınız.

### Değerlendirme metrikleri (7 zorunlu)

`Accuracy`, `Precision`, `Recall`, `F1`, **`MCC`**, `AUC_ROC`, **`AUC_PR`**
(birincil sıralama ölçütü AUC-PR). Kaynak: [`src/metrics.py`](src/metrics.py).

---

## 🏗️ Proje Yapısı

```
Uydu-Telemetri-Anomali-Tespiti/
├── data/
│   ├── raw/                  # dataset.csv (18 özellik), segments.csv (ham sinyal)
│   ├── processed/            # NB02 çıktıları (ölçekli/SMOTE'lu bölmeler)
│   ├── features/             # özellik çıktıları
│   └── synthetic/            # sentetik telemetri (NB09)
├── notebooks/                # 01–14 (aşağıdaki tablo)
├── src/
│   ├── data_loader.py        # veri yükleme
│   ├── preprocessor.py       # ön işleme (RobustScaler, outlier, impute)
│   ├── feature_engineer.py   # özellik mühendisliği
│   ├── synthetic_generator.py# profil-temelli sentetik üretici
│   ├── metrics.py            # 7 zorunlu metrik (BENCHMARK_METRICS)
│   ├── benchmark_reference.py# Ruszczak et al. 2024 Tablo 3 baseline
│   └── models/{supervised,unsupervised,evaluator}.py
├── models/                   # eğitilmiş modeller (joblib/keras) + unsupervised/ + deep_sequence/
├── reports/
│   ├── figures/              # grafikler (SHAP, ROC, vb.)
│   └── metrics/              # final_comparison.json, deep_sequence_comparison.json, ...
├── app/                      # Plotly Dash arayüzü
│   ├── app.py                # ana orkestratör (giriş noktası)
│   ├── core/                 # state.py (Singleton), constants.py (kanonik sabitler)
│   ├── layout/sidebar.py
│   ├── pages/                # 14 sayfa (dashboard, upload, analysis, ...)
│   └── utils/                # model_loader, feature_extractor, ui
├── tests/                    # pytest birim testleri
├── train_all_models.py       # KANONİK eğitim motoru -> reports/metrics/final_comparison.json
├── models_trained.txt        # 46 model listesi + kaynak atıfları
├── requirements.txt / environment.yml
└── README.md
```

---

## 🚀 Kurulum

```bash
git clone https://github.com/furkanozturk06/tusas-liftup.git
cd tusas-liftup

# pip ile
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# veya conda ile
conda env create -f environment.yml
conda activate uydu-anomali
```

---

## 💻 Kullanım

### Kanonik eğitim motoru

Tüm tabular modelleri tek noktadan eğitip kanonik artefaktları üretir
(`final_comparison.json`, eğitilmiş modeller, ölçekleyici, test verisi):

```bash
python train_all_models.py
```

### Jupyter Notebook'lar

| Notebook | Açıklama |
|---|---|
| `01_veri_inceleme.ipynb` | Keşifsel veri analizi (EDA), kanal/segment istatistikleri |
| `02_on_isleme.ipynb` | RobustScaler ölçekleme, outlier, resmi bölme + SMOTE (tek ön işleme kaynağı) |
| `03_feature_engineering.ipynb` | Özellik mühendisliği ve seçimi |
| `04_model_supervised.ipynb` | 23 gözetimli model + Bölüm 9: 2 derin sıralı model (ham sinyal) |
| `05_model_unsupervised.ipynb` | 19 gözetimsiz model (val-F1 eşik protokolü) |
| `06_model_karsilastirma.ipynb` | 46 modelin karşılaştırması; ESA-ADB baseline paneli; benchmark |
| `07_shap_analizi.ipynb` | SHAP yorumlanabilirliği (39 model: ağaç=TreeExplainer, diğer=KernelExplainer) |
| `08_ablation_study.ipynb` | Özellik ablasyonu (18 özelliğin 11'i yeterli) |
| `09_sentetik_veri_uretimi.ipynb` | Profil-temelli sentetik telemetri üretimi |
| `10_esa_feature_pipeline.ipynb` | Ham sinyalden 18 ESA özelliğini otomatik çıkarma |
| `11_guc_tuketimi_analizi.ipynb` | 44 modelin güç/enerji/karbon/bellek maliyeti |
| `12_benchmark_karsilastirma.ipynb` | Ruszczak et al. (2024) baseline'ı ile yeniden üretim |
| `13_sentetik_augmentasyon_ablasyonu.ipynb` | Sentetik augmentasyon ablasyonu (gerçek Ψ) |
| `14_augmentasyon_karsilastirma.ipynb` | SMOTE / ICCS-ω / sentetik augmentasyon karşılaştırması |

### Dashboard (Plotly Dash)

```bash
python app/app.py
```

14 sayfa: Operasyon Paneli, Veri Yükle, Anomali Analizi, Sonuçlar, Anomali Detay,
Canlı İzleme, Model Performans, SHAP Analiz, Güç Tüketimi, Benchmark, Ablasyon,
Sentetik Veri, Augmentasyon ve ESA Pipeline.

### Testler

```bash
pytest tests/ -q
```

---

## 🧪 Yöntem Notları

- **Sızıntısız değerlendirme:** Tüm modeller resmi Ψ (529 segment) üzerinde, kendi
  rastgele bölmesini kullanmadan değerlendirilir; ön işleme yalnız NB02'de yapılır.
- **18 ESA özelliği:** mean, var, std, kurtosis, skew, n_peaks, duration, len,
  gaps_squared, len_weighted, var_div_duration, var_div_len, smooth10/20_n_peaks,
  diff/diff2_peaks, diff/diff2_var.
- **Tekrarüretilebilirlik:** Sabit tohum (seed = 42); kanonik motor + 14 notebook ile
  baştan üretilebilir.
- **Tek kaynak (single source of truth):** Kanonik model listeleri
  [`app/core/constants.py`](app/core/constants.py)'tedir; dashboard ve güç kataloğu
  buradan türer, böylece liste değişse bile her yer tutarlı kalır.

---

## 🛠️ Teknoloji Yığını

| Kategori | Kütüphaneler |
|---|---|
| **Veri İşleme** | pandas, numpy, scipy |
| **Makine Öğrenmesi** | scikit-learn, imbalanced-learn, xgboost, lightgbm, catboost, pyod |
| **Derin Öğrenme** | TensorFlow / Keras |
| **Yorumlanabilirlik** | SHAP |
| **Görselleştirme** | matplotlib, seaborn, plotly |
| **Dashboard** | Dash, dash-bootstrap-components, dash-iconify |
| **Model Saklama** | joblib |

---

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE)
dosyasına bakınız.

---

<p align="center">
  <i>ESA OPS-SAT telemetri anomali tespiti — kanonik, sızıntısız, literatürle kıyaslanabilir.</i>
</p>
