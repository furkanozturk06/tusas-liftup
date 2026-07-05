"""
Guc Tuketimi Simulasyonu Sayfasi
=================================
Kanonik 44 modelin (42 tabular + 2 derin sirali) tahmini hesaplama maliyeti,
egitim suresi, bellek kullanimi ve enerji tuketimini simule eder. ESA-ADB
literatur baseline'lari fiilen egitilmedigi icin maliyet katalogunda yer almaz.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output, no_update
import dash_bootstrap_components as dbc

from utils.ui import PLT_LAYOUT, icon as _icon, metric_card as _metric_card, stat_strip

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POWER_CSV = os.path.join(ROOT, "reports", "power_profiles.csv")


POWER_PROFILES = {
    "LogisticRegression": {"cpu_watts": 8,  "train_sec": 0.5,  "infer_ms": 0.02, "memory_mb": 15,  "category": "Gözetimli", "complexity": "Düşük"},
    "Ridge":              {"cpu_watts": 6,  "train_sec": 0.3,  "infer_ms": 0.01, "memory_mb": 10,  "category": "Gözetimli", "complexity": "Düşük"},
    "SGD":                {"cpu_watts": 7,  "train_sec": 0.4,  "infer_ms": 0.01, "memory_mb": 12,  "category": "Gözetimli", "complexity": "Düşük"},
    "NaiveBayes":         {"cpu_watts": 5,  "train_sec": 0.2,  "infer_ms": 0.01, "memory_mb": 8,   "category": "Gözetimli", "complexity": "Düşük"},
    "LDA":                {"cpu_watts": 6,  "train_sec": 0.3,  "infer_ms": 0.01, "memory_mb": 12,  "category": "Gözetimli", "complexity": "Düşük"},
    "QDA":                {"cpu_watts": 7,  "train_sec": 0.4,  "infer_ms": 0.02, "memory_mb": 15,  "category": "Gözetimli", "complexity": "Düşük"},
    "DecisionTree":       {"cpu_watts": 8,  "train_sec": 0.8,  "infer_ms": 0.01, "memory_mb": 20,  "category": "Gözetimli", "complexity": "Düşük"},
    "KNN":                {"cpu_watts": 10, "train_sec": 0.1,  "infer_ms": 2.5,  "memory_mb": 150, "category": "Gözetimli", "complexity": "Orta"},
    "LSVC":               {"cpu_watts": 12, "train_sec": 1.5,  "infer_ms": 0.02, "memory_mb": 25,  "category": "Gözetimli", "complexity": "Orta"},
    "RandomForest":        {"cpu_watts": 25, "train_sec": 8,   "infer_ms": 0.5,  "memory_mb": 200, "category": "Gözetimli", "complexity": "Orta"},
    "ExtraTrees":          {"cpu_watts": 22, "train_sec": 6,   "infer_ms": 0.4,  "memory_mb": 180, "category": "Gözetimli", "complexity": "Orta"},
    "GradientBoosting":    {"cpu_watts": 30, "train_sec": 25,  "infer_ms": 0.3,  "memory_mb": 120, "category": "Gözetimli", "complexity": "Orta"},
    "HistGradientBoosting":{"cpu_watts": 28, "train_sec": 5,   "infer_ms": 0.2,  "memory_mb": 100, "category": "Gözetimli", "complexity": "Orta"},
    "AdaBoost":            {"cpu_watts": 20, "train_sec": 10,  "infer_ms": 0.3,  "memory_mb": 80,  "category": "Gözetimli", "complexity": "Orta"},
    "Bagging":             {"cpu_watts": 22, "train_sec": 7,   "infer_ms": 0.4,  "memory_mb": 170, "category": "Gözetimli", "complexity": "Orta"},
    "Voting Ensemble":     {"cpu_watts": 35, "train_sec": 15,  "infer_ms": 0.8,  "memory_mb": 250, "category": "Gözetimli", "complexity": "Yüksek"},
    "Stacking Ensemble":   {"cpu_watts": 40, "train_sec": 28,  "infer_ms": 1.0,  "memory_mb": 320, "category": "Gözetimli", "complexity": "Yüksek"},
    "XGBoost":             {"cpu_watts": 35, "train_sec": 12,  "infer_ms": 0.15, "memory_mb": 150, "category": "Gözetimli", "complexity": "Orta"},
    "LightGBM":            {"cpu_watts": 28, "train_sec": 6,   "infer_ms": 0.15, "memory_mb": 120, "category": "Gözetimli", "complexity": "Orta"},
    "CatBoost":            {"cpu_watts": 32, "train_sec": 18,  "infer_ms": 0.20, "memory_mb": 160, "category": "Gözetimli", "complexity": "Orta"},
    "XGBOD":               {"cpu_watts": 40, "train_sec": 30,  "infer_ms": 0.5,  "memory_mb": 300, "category": "Gözetimli", "complexity": "Yüksek"},
    "SVM":                 {"cpu_watts": 30, "train_sec": 20,  "infer_ms": 1.0,  "memory_mb": 200, "category": "Gözetimli", "complexity": "Yüksek"},
    "MLP":              {"cpu_watts": 45, "train_sec": 60,  "infer_ms": 0.5,  "memory_mb": 300, "category": "Gözetimli", "complexity": "Yüksek"},
    "LSTM":             {"cpu_watts": 65, "train_sec": 180, "infer_ms": 2.0,  "memory_mb": 500, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "BiLSTM":           {"cpu_watts": 75, "train_sec": 250, "infer_ms": 3.5,  "memory_mb": 650, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "GRU":              {"cpu_watts": 55, "train_sec": 150, "infer_ms": 1.8,  "memory_mb": 420, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "BiGRU":            {"cpu_watts": 65, "train_sec": 200, "infer_ms": 3.0,  "memory_mb": 550, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "CNN1D":            {"cpu_watts": 50, "train_sec": 90,  "infer_ms": 1.0,  "memory_mb": 350, "category": "Gözetimli", "complexity": "Yüksek"},
    "CNN_LSTM":         {"cpu_watts": 70, "train_sec": 220, "infer_ms": 3.0,  "memory_mb": 600, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "CNN_BiLSTM":       {"cpu_watts": 80, "train_sec": 280, "infer_ms": 4.0,  "memory_mb": 700, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "CNN_GRU":          {"cpu_watts": 65, "train_sec": 190, "infer_ms": 2.5,  "memory_mb": 520, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "Transformer":      {"cpu_watts": 85, "train_sec": 300, "infer_ms": 3.5,  "memory_mb": 800, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "TCN":              {"cpu_watts": 60, "train_sec": 160, "infer_ms": 2.0,  "memory_mb": 450, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "Attention_BiLSTM": {"cpu_watts": 80, "train_sec": 290, "infer_ms": 4.0,  "memory_mb": 750, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "FCN":              {"cpu_watts": 55, "train_sec": 100, "infer_ms": 1.2,  "memory_mb": 380, "category": "Gözetimli", "complexity": "Yüksek"},
    "ResNet1D":         {"cpu_watts": 70, "train_sec": 200, "infer_ms": 2.5,  "memory_mb": 600, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "InceptionTime":    {"cpu_watts": 75, "train_sec": 250, "infer_ms": 3.0,  "memory_mb": 700, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "LSTM_FCN":         {"cpu_watts": 70, "train_sec": 210, "infer_ms": 2.8,  "memory_mb": 580, "category": "Gözetimli", "complexity": "Çok Yüksek"},
    "IsolationForest":  {"cpu_watts": 15, "train_sec": 3,   "infer_ms": 0.3,  "memory_mb": 80,  "category": "Gözetimsiz", "complexity": "Düşük"},
    "LOF":              {"cpu_watts": 12, "train_sec": 2,   "infer_ms": 1.5,  "memory_mb": 100, "category": "Gözetimsiz", "complexity": "Düşük"},
    "OneClassSVM":      {"cpu_watts": 25, "train_sec": 15,  "infer_ms": 0.8,  "memory_mb": 150, "category": "Gözetimsiz", "complexity": "Orta"},
    "KMeans":           {"cpu_watts": 10, "train_sec": 2,   "infer_ms": 0.1,  "memory_mb": 50,  "category": "Gözetimsiz", "complexity": "Düşük"},
    "GMM":              {"cpu_watts": 15, "train_sec": 5,   "infer_ms": 0.3,  "memory_mb": 80,  "category": "Gözetimsiz", "complexity": "Orta"},
    "EllipticEnvelope": {"cpu_watts": 18, "train_sec": 8,   "infer_ms": 0.2,  "memory_mb": 90,  "category": "Gözetimsiz", "complexity": "Orta"},
    "PCA":              {"cpu_watts": 8,  "train_sec": 1,   "infer_ms": 0.05, "memory_mb": 30,  "category": "Gözetimsiz", "complexity": "Düşük"},
    "DBSCAN":           {"cpu_watts": 20, "train_sec": 10,  "infer_ms": 0.5,  "memory_mb": 200, "category": "Gözetimsiz", "complexity": "Orta"},
    "Autoencoder":      {"cpu_watts": 40, "train_sec": 80,  "infer_ms": 0.4,  "memory_mb": 250, "category": "Gözetimsiz", "complexity": "Yüksek"},
    "LSTM_Autoencoder": {"cpu_watts": 60, "train_sec": 150, "infer_ms": 2.0,  "memory_mb": 400, "category": "Gözetimsiz", "complexity": "Çok Yüksek"},
    "VAE":              {"cpu_watts": 45, "train_sec": 90,  "infer_ms": 0.5,  "memory_mb": 280, "category": "Gözetimsiz", "complexity": "Yüksek"},
    "AnoGAN":           {"cpu_watts": 70, "train_sec": 400, "infer_ms": 1.5,  "memory_mb": 500, "category": "Gözetimsiz", "complexity": "Çok Yüksek"},
    "ALAD":             {"cpu_watts": 75, "train_sec": 350, "infer_ms": 1.2,  "memory_mb": 480, "category": "Gözetimsiz", "complexity": "Çok Yüksek"},
    "ECOD":  {"cpu_watts": 5,  "train_sec": 1,   "infer_ms": 0.05, "memory_mb": 20,  "category": "Gözetimsiz", "complexity": "Düşük"},
    "COPOD": {"cpu_watts": 6,  "train_sec": 1.5, "infer_ms": 0.08, "memory_mb": 25,  "category": "Gözetimsiz", "complexity": "Düşük"},
    "HBOS":  {"cpu_watts": 4,  "train_sec": 0.5, "infer_ms": 0.03, "memory_mb": 15,  "category": "Gözetimsiz", "complexity": "Düşük"},
    "CBLOF": {"cpu_watts": 10, "train_sec": 3,   "infer_ms": 0.2,  "memory_mb": 60,  "category": "Gözetimsiz", "complexity": "Orta"},
    "ABOD":  {"cpu_watts": 18, "train_sec": 25,  "infer_ms": 5.0,  "memory_mb": 200, "category": "Gözetimsiz", "complexity": "Yüksek"},
    "COF":   {"cpu_watts": 15, "train_sec": 20,  "infer_ms": 3.0,  "memory_mb": 180, "category": "Gözetimsiz", "complexity": "Orta"},
    "SOD":   {"cpu_watts": 14, "train_sec": 15,  "infer_ms": 2.0,  "memory_mb": 150, "category": "Gözetimsiz", "complexity": "Orta"},
    "SOS":   {"cpu_watts": 12, "train_sec": 10,  "infer_ms": 1.5,  "memory_mb": 120, "category": "Gözetimsiz", "complexity": "Orta"},
    "LODA":  {"cpu_watts": 5,  "train_sec": 1,   "infer_ms": 0.05, "memory_mb": 20,  "category": "Gözetimsiz", "complexity": "Düşük"},
    "INNE":  {"cpu_watts": 12, "train_sec": 5,   "infer_ms": 0.8,  "memory_mb": 80,  "category": "Gözetimsiz", "complexity": "Orta"},
    "LMDD":  {"cpu_watts": 10, "train_sec": 8,   "infer_ms": 0.5,  "memory_mb": 60,  "category": "Gözetimsiz", "complexity": "Orta"},
    "SO_GAAL":  {"cpu_watts": 50, "train_sec": 200, "infer_ms": 1.0, "memory_mb": 350, "category": "Gözetimsiz", "complexity": "Çok Yüksek"},
    "MO_GAAL":  {"cpu_watts": 55, "train_sec": 250, "infer_ms": 1.2, "memory_mb": 400, "category": "Gözetimsiz", "complexity": "Çok Yüksek"},
    "DeepSVDD": {"cpu_watts": 45, "train_sec": 120, "infer_ms": 0.8, "memory_mb": 300, "category": "Gözetimsiz", "complexity": "Yüksek"},
    "LUNAR":    {"cpu_watts": 50, "train_sec": 180, "infer_ms": 1.5, "memory_mb": 400, "category": "Gözetimsiz", "complexity": "Çok Yüksek"},
    "DIF":      {"cpu_watts": 40, "train_sec": 100, "infer_ms": 0.8, "memory_mb": 250, "category": "Gözetimsiz", "complexity": "Yüksek"},
}

# Kanonik 44 modele indirge: 23 gözetimli + 19 gözetimsiz + 2 derin sıralı (CNN1D, TCN).
# ESA-ADB literatür baseline'ları (Telemanom-ESA, DC-VAE-ESA) bu projede fiilen
# eğitilmediği için ampirik maliyet kataloğunda yer ALMAZ. Tek kaynak: core.constants
# (model listesi değişirse katalog otomatik senkron kalır).
from core.constants import SUP_MODEL_NAMES as _SUP, UNSUP_MODEL_NAMES as _UNSUP, DEEP_SEQ_MODELS as _DEEP
_CANONICAL_ORDER = list(_SUP) + list(_DEEP) + list(_UNSUP)
POWER_PROFILES = {k: POWER_PROFILES[k] for k in _CANONICAL_ORDER if k in POWER_PROFILES}

COMPLEXITY_COLORS = {
    "Düşük": "#10B981",
    "Orta": "#F59E0B",
    "Yüksek": "#EF4444",
    "Çok Yüksek": "#DC2626",
}
CATEGORY_COLORS = {"Gözetimli": "#3B82F6", "Gözetimsiz": "#8B5CF6"}

CO2_FACTOR = 400


def _active_profiles():
    """NB11 çıktısı reports/power_profiles.csv varsa onu (tek kaynak) kullanır;
    yoksa yerleşik POWER_PROFILES tahminlerine düşer. Böylece dashboard ile
    notebook çıktısı ayrışmaz."""
    if os.path.exists(POWER_CSV):
        try:
            df = pd.read_csv(POWER_CSV)
            prof = {}
            for _, r in df.iterrows():
                prof[str(r["Model"])] = {
                    "cpu_watts": float(r["CPU (W)"]), "train_sec": float(r["Eğitim (s)"]),
                    "infer_ms": float(r["Çıkarım (ms)"]), "memory_mb": float(r["Bellek (MB)"]),
                    "category": r["Kategori"], "complexity": r["Karmaşıklık"]}
            if prof:
                return prof, "reports/power_profiles.csv (Notebook 11)"
        except Exception as e:
            print("power_profiles.csv okunamadı, yerleşik profillere düşülüyor:", e)
    return POWER_PROFILES, "yerleşik tahmin profilleri"


def _build_df(dataset_size: int = 10000):
    """Aktif profil verisinden ölçekli DataFrame oluştur (CSV varsa CSV'den)."""
    scale = dataset_size / 10000
    profiles, _ = _active_profiles()
    rows = []
    for name, p in profiles.items():
        t = p["train_sec"] * scale
        energy_wh = p["cpu_watts"] * t / 3600
        rows.append({
            "Model": name,
            "Kategori": p["category"],
            "Karmaşıklık": p["complexity"],
            "CPU (W)": p["cpu_watts"],
            "Egitim (s)": round(t, 2),
            "Cikarim (ms)": p["infer_ms"],
            "Bellek (MB)": p["memory_mb"],
            "Enerji (Wh)": round(energy_wh, 4),
            "CO2 (g)": round(energy_wh / 1000 * CO2_FACTOR, 4),
        })
    return pd.DataFrame(rows)


def get_power_layout(ALL_METRICS=None):
    if ALL_METRICS is None:
        ALL_METRICS = {}

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Guc Tuketimi Simulasyonu", className="page-title"),
            html.Div("Algoritmalarin hesaplama maliyeti ve enerji analizi", className="page-subtitle"),
        ]),

        html.Div(className="panel mb-4", children=[
            html.Div(className="panel-title", children=[
                _icon("mdi:database-cog-outline", 16), " Veri Seti Boyutu (örnek sayısı)"]),
            dcc.Slider(
                id="power-dataset-slider",
                min=1000, max=100000, step=1000, value=10000,
                marks={1000: "1K", 10000: "10K", 25000: "25K", 50000: "50K", 100000: "100K"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ]),

        html.Div(id="power-summary-cards"),

        html.Div(id="power-charts"),

        html.Div(id="power-table"),
    ])


def register_power_callbacks(app, ALL_METRICS=None):
    if ALL_METRICS is None:
        ALL_METRICS = {}

    @app.callback(
        Output("power-summary-cards", "children"),
        Output("power-charts", "children"),
        Output("power-table", "children"),
        Input("power-dataset-slider", "value"),
    )
    def update_power_page(dataset_size):
        if dataset_size is None:
            dataset_size = 10000

        df = _build_df(dataset_size)
        total_energy = df["Enerji (Wh)"].sum()
        total_co2 = df["CO2 (g)"].sum()
        most_efficient = df.loc[df["Enerji (Wh)"].idxmin(), "Model"]
        most_costly = df.loc[df["Enerji (Wh)"].idxmax(), "Model"]
        eff_val = df["Enerji (Wh)"].min()
        cost_val = df["Enerji (Wh)"].max()

        cards = stat_strip([
            ("Toplam Enerji", f"{total_energy:.2f} Wh", f"{len(POWER_PROFILES)} model toplamı", "cyan"),
            ("CO2 Emisyonu", f"{total_co2:.3f} g", "Tahmini karbon", "green"),
            ("En Verimli", most_efficient, f"{eff_val:.4f} Wh", "green"),
            ("En Maliyetli", most_costly, f"{cost_val:.4f} Wh", "red"),
        ])

        df_sorted = df.sort_values("Enerji (Wh)", ascending=True)
        bar_colors = [COMPLEXITY_COLORS.get(c, "#64748B") for c in df_sorted["Karmaşıklık"]]

        fig_energy = go.Figure(go.Bar(
            y=df_sorted["Model"], x=df_sorted["Enerji (Wh)"],
            orientation="h", marker_color=bar_colors,
            text=[f"{v:.4f}" for v in df_sorted["Enerji (Wh)"]],
            textposition="outside", textfont=dict(size=9, color="#475569"),
        ))
        fig_energy.update_layout(
            **PLT_LAYOUT, height=1200,
            title=f"Egitim Enerji Tuketimi (Wh) - {dataset_size:,} örnek",
            xaxis_title="Enerji (Wh)",
        )

        fig_scatter = go.Figure()
        if ALL_METRICS:
            scatter_rows = []
            for _, row in df.iterrows():
                m = ALL_METRICS.get(row["Model"], {})
                f1 = m.get("F1", m.get("F1_Score", None))
                if f1 is not None:
                    scatter_rows.append({
                        "Model": row["Model"], "Enerji (Wh)": row["Enerji (Wh)"],
                        "F1": f1, "Kategori": row["Kategori"],
                    })
            if scatter_rows:
                sdf = pd.DataFrame(scatter_rows)
                for cat, color in CATEGORY_COLORS.items():
                    sub = sdf[sdf["Kategori"] == cat]
                    if sub.empty:
                        continue
                    fig_scatter.add_trace(go.Scatter(
                        x=sub["Enerji (Wh)"], y=sub["F1"],
                        mode="markers+text", text=sub["Model"],
                        textposition="top center", textfont=dict(size=9, color="#475569"),
                        marker=dict(size=10, color=color, opacity=0.85),
                        name=cat,
                    ))
        fig_scatter.update_layout(
            **PLT_LAYOUT, height=500,
            title="Enerji vs F1 Skoru (Verimlilik Haritasi)",
            xaxis_title="Enerji (Wh)", yaxis_title="F1 Skoru",
            xaxis_type="log",
        )

        cat_energy = df.groupby("Kategori")["Enerji (Wh)"].sum().reset_index()
        fig_pie = go.Figure(go.Pie(
            labels=cat_energy["Kategori"], values=cat_energy["Enerji (Wh)"],
            marker=dict(colors=["#3B82F6", "#8B5CF6"]),
            textinfo="label+percent", textfont=dict(size=12),
            hole=0.45,
        ))
        fig_pie.update_layout(**PLT_LAYOUT, height=350, title="Kategori Bazlı Enerji Dağılımi",
                              showlegend=False)

        comp_energy = df.groupby("Karmaşıklık")["Enerji (Wh)"].mean().reset_index()
        comp_order = ["Düşük", "Orta", "Yüksek", "Çok Yüksek"]
        comp_energy["_order"] = comp_energy["Karmaşıklık"].map({v: i for i, v in enumerate(comp_order)})
        comp_energy = comp_energy.sort_values("_order")
        fig_comp = go.Figure(go.Bar(
            x=comp_energy["Karmaşıklık"], y=comp_energy["Enerji (Wh)"],
            marker_color=[COMPLEXITY_COLORS.get(c, "#64748B") for c in comp_energy["Karmaşıklık"]],
            text=[f"{v:.4f}" for v in comp_energy["Enerji (Wh)"]],
            textposition="outside",
        ))
        fig_comp.update_layout(**PLT_LAYOUT, height=350, title="Karmaşıklık Seviyesine Gore Ort. Enerji",
                               yaxis_title="Ort. Enerji (Wh)")

        df_mem = df.sort_values("Bellek (MB)", ascending=True)
        fig_mem = go.Figure(go.Bar(
            y=df_mem["Model"], x=df_mem["Bellek (MB)"],
            orientation="h",
            marker_color=["#06B6D4" if v < 200 else "#F59E0B" if v < 500 else "#EF4444" for v in df_mem["Bellek (MB)"]],
            text=[f"{v} MB" for v in df_mem["Bellek (MB)"]],
            textposition="outside", textfont=dict(size=9, color="#475569"),
        ))
        fig_mem.update_layout(**PLT_LAYOUT, height=1200, title="Bellek Kullanimi (MB)",
                              xaxis_title="Bellek (MB)")

        df_time = df.nlargest(15, "Egitim (s)")
        fig_time = go.Figure(go.Bar(
            x=df_time["Model"], y=df_time["Egitim (s)"],
            marker_color="#F59E0B",
            text=[f"{v:.1f}s" for v in df_time["Egitim (s)"]],
            textposition="outside",
        ))
        fig_time.update_layout(**PLT_LAYOUT, height=400,
                               title=f"En Yavas 15 Model - Egitim Süresi ({dataset_size:,} örnek)",
                               yaxis_title="Sure (saniye)")

        legend_items = [
            html.Span([html.Span(style={"display": "inline-block", "width": "12px", "height": "12px",
                                         "borderRadius": "0", "backgroundColor": c, "marginRight": "6px"}),
                        t], style={"marginRight": "18px", "fontSize": "12px", "color": "#334155"})
            for t, c in COMPLEXITY_COLORS.items()
        ]
        legend_bar = html.Div(style={"display": "flex", "alignItems": "center", "padding": "10px 0",
                                      "marginBottom": "8px"}, children=[
            html.Span("Karmaşıklık: ", style={"fontWeight": "600", "fontSize": "12px", "color": "#64748B",
                                                "marginRight": "12px"}),
            *legend_items,
        ])

        charts = html.Div([
            legend_bar,
            dbc.Row([
                dbc.Col(html.Div(className="panel", children=[
                    dcc.Graph(figure=fig_energy, config={"displayModeBar": False})]), md=6),
                dbc.Col(html.Div(className="panel", children=[
                    dcc.Graph(figure=fig_mem, config={"displayModeBar": False})]), md=6),
            ], className="mb-4 g-3"),
            dbc.Row([
                dbc.Col(html.Div(className="panel", children=[
                    dcc.Graph(figure=fig_scatter, config={"displayModeBar": False})]), md=8),
                dbc.Col(html.Div([
                    html.Div(className="panel mb-3", children=[
                        dcc.Graph(figure=fig_pie, config={"displayModeBar": False})]),
                    html.Div(className="panel", children=[
                        dcc.Graph(figure=fig_comp, config={"displayModeBar": False})]),
                ]), md=4),
            ], className="mb-4 g-3"),
            html.Div(className="panel mb-4", children=[
                dcc.Graph(figure=fig_time, config={"displayModeBar": False})]),
        ])

        from dash import dash_table
        table = html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[
                _icon("mdi:table-large", 16), f" Detayli Guc Profili ({len(df)} model)"]),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in df.columns],
                data=df.sort_values("Enerji (Wh)", ascending=False).to_dict("records"),
                page_size=20,
                sort_action="native",
                filter_action="native",
                style_header={"backgroundColor": "#EEF2F8", "color": "#64748B",
                               "fontWeight": "600", "border": "1px solid #E2E8F0",
                               "fontSize": "11px", "textTransform": "uppercase"},
                style_cell={"backgroundColor": "#FFFFFF", "color": "#1E293B",
                             "border": "1px solid #E2E8F0", "fontFamily": "IBM Plex Sans",
                             "fontSize": "12px", "padding": "8px"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#F4F6FB"},
                    {"if": {"filter_query": '{Karmaşıklık} = "Çok Yüksek"', "column_id": "Karmaşıklık"},
                     "color": "#DC2626", "fontWeight": "600"},
                    {"if": {"filter_query": '{Karmaşıklık} = "Yüksek"', "column_id": "Karmaşıklık"},
                     "color": "#D97706"},
                    {"if": {"filter_query": '{Karmaşıklık} = "Düşük"', "column_id": "Karmaşıklık"},
                     "color": "#16A34A"},
                ],
            ),
        ])

        return cards, charts, table
