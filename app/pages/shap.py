"""SHAP Analiz sayfasi: layout + callback'ler."""
import os
import io
import json
import time
import base64
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, dash_table, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from utils.model_loader import predict
from utils.feature_extractor import extract_features_from_raw
from utils.ui import PLT_LAYOUT, icon, metric_card
from core.constants import (DEMO_PATH, LIVE_DATA_PATH, SHAP_PKL, BENCHMARK_METRICS,
                            PRIMARY_METRIC, DROP_COLS, SUP_MODEL_NAMES, UNSUP_MODEL_NAMES,
                            ANALYSIS_PRESETS)
from core.state import (MODELS, THRESHOLDS, SCALER, TEST_DATA, ALL_METRICS, FEATURE_COLS,
                        LIVE_DATA, SHAP_DATA, get_tree_explainer, best_model)


def _shap_model_list():
    """SHAP değeri hesaplanmış tüm modellerin adları (NB07 'shap_models' listesi).
    Geriye dönük: liste yoksa anahtarlardan türetilir (kısa rf/xgb/mlp aliaslar hariç)."""
    if SHAP_DATA is None:
        return []
    models = SHAP_DATA.get("shap_models")
    if models:
        return list(models)
    aliases = ("rf_", "xgb_", "mlp_")
    return sorted(k[:-len("_shap_values")] for k in SHAP_DATA
                  if k.endswith("_shap_values") and not k.startswith(aliases))


def _default_shap_model(models):
    """Açıklama için varsayılan model: RandomForest (kesin ağaç SHAP) varsa o, yoksa ilk."""
    if "RandomForest" in models:
        return "RandomForest"
    return models[0] if models else None


def page_shap():
    if SHAP_DATA is None:
        return html.Div([
            html.Div(className="page-header", children=[
                html.Div("SHAP Analiz", className="page-title"),
                html.Div("Model yorumlanabilirlik analizi", className="page-subtitle")]),
            html.Div(className="warning-box", children=[
                icon("mdi:alert-outline", 40, "#F59E0B"),
                html.Div("SHAP Verileri Bulunamadi", className="warning-title"),
                html.Div([
                    "SHAP analiz verileri henuz hesaplanmamis. ",
                    "Lutfen once ", html.Code("notebooks/07_shap_analizi.ipynb"),
                    " notebook'unu calistirin.",
                    html.Br(), html.Br(),
                    "Notebook calistirildiktan sonra ",
                    html.Code("models/shap_values.pkl"),
                    " dosyası olusturulacak ve bu sayfa aktif hale gelecektir."
                ], className="warning-body")
            ])
        ])

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    y_test = SHAP_DATA['y_test']
    anomaly_indices = np.where(y_test == 1)[0]
    anomaly_options = [{"label": f"Segment #{i} (index {idx})", "value": int(idx)}
                       for i, idx in enumerate(anomaly_indices, 1)]

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("SHAP Analiz", className="page-title"),
            html.Div("Model yorumlanabilirlik ve özellik önemi analizi", className="page-subtitle")]),
        dcc.Tabs(id="shap-tabs", value="tab-importance", className="custom-tabs", children=[
            dcc.Tab(label="Özellik Onemi", value="tab-importance", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Anomali Açıklama", value="tab-explain", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Model Karşılaştırma", value="tab-compare", className="tab", selected_className="tab--selected"),
        ]),
        html.Div(id="shap-tab-content", style={"marginTop": "20px"}),
        dcc.Store(id="shap-anomaly-options", data=[o["value"] for o in anomaly_options]),
    ])


@callback(Output("shap-tab-content", "children"),
          Input("shap-tabs", "value"),
          prevent_initial_call=False)
def render_shap_tab(tab):
    if SHAP_DATA is None:
        return html.Div()

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    feature_cols = SHAP_DATA.get('feature_cols', [])
    y_test = SHAP_DATA['y_test']
    anomaly_indices = np.where(y_test == 1)[0]

    if tab == "tab-importance":
        models = _shap_model_list()
        return html.Div([
            html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[icon("mdi:chart-bar", 16),
                         f"Model Seçimi ({len(models)} model)"]),
                dcc.Dropdown(
                    id="shap-model-select",
                    options=[{"label": m, "value": m} for m in models],
                    value=_default_shap_model(models),
                    className="shap-dropdown", clearable=False,
                ),
            ]),
            html.Div(id="shap-importance-chart", className="panel", style={"marginTop": "16px"}),
            html.Div(id="shap-importance-text", className="panel", style={"marginTop": "16px"}),
        ])

    elif tab == "tab-explain":
        anomaly_options = [{"label": f"Segment #{i} (index {idx})", "value": int(idx)}
                           for i, idx in enumerate(anomaly_indices, 1)]
        models = _shap_model_list()
        return html.Div([
            html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[icon("mdi:magnify", 16), "Anomali ve Model Seçimi"]),
                dbc.Row([
                    dbc.Col([
                        html.Div("Anomali segmenti", className="section-label"),
                        dcc.Dropdown(
                            id="shap-anomaly-select", options=anomaly_options,
                            value=anomaly_options[0]["value"] if anomaly_options else None,
                            placeholder="Bir anomali segmenti seçin...",
                            className="shap-dropdown", clearable=False)], md=8),
                    dbc.Col([
                        html.Div("Açıklayıcı model", className="section-label"),
                        dcc.Dropdown(
                            id="shap-explain-model", options=[{"label": m, "value": m} for m in models],
                            value=_default_shap_model(models),
                            className="shap-dropdown", clearable=False)], md=4),
                ], className="g-2"),
            ]),
            html.Div(id="shap-waterfall-chart", className="panel", style={"marginTop": "16px"}),
            html.Div(id="shap-waterfall-text", className="panel", style={"marginTop": "16px"}),
        ])

    elif tab == "tab-compare":
        models = _shap_model_list()
        m1 = "RandomForest" if "RandomForest" in models else (models[0] if models else None)
        m2 = "XGBoost" if "XGBoost" in models else next((m for m in models if m != m1), m1)
        if not m1 or not m2 or f"{m1}_shap_values" not in SHAP_DATA or f"{m2}_shap_values" not in SHAP_DATA:
            return html.Div("Karşılaştırma için en az iki modelin SHAP verisi gerekli.", className="info-box")
        rf_shap = SHAP_DATA[f'{m1}_shap_values']
        xgb_shap = SHAP_DATA[f'{m2}_shap_values']
        rf_imp = np.abs(rf_shap).mean(axis=0)
        xgb_imp = np.abs(xgb_shap).mean(axis=0)

        combined = rf_imp + xgb_imp
        top_idx = np.argsort(combined)[-10:][::-1]
        top_labels = [feature_labels[i] for i in top_idx]

        fig = go.Figure()
        fig.add_trace(go.Bar(name=m1, y=top_labels, x=rf_imp[top_idx],
                             orientation='h', marker_color='#3B82F6', opacity=0.85))
        fig.add_trace(go.Bar(name=m2, y=top_labels, x=xgb_imp[top_idx],
                             orientation='h', marker_color='#10B981', opacity=0.85))
        fig.update_layout(**PLT_LAYOUT, height=500, barmode='group',
                          title=f"{m1} vs {m2} - SHAP Özellik Onemi Karşılaştırmasi",
                          xaxis_title="Ortalama |SHAP Değeri|",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(autorange="reversed")

        diff = np.abs(rf_imp - xgb_imp)
        diff_idx = np.argsort(diff)[-3:][::-1]
        diff_text_parts = []
        for idx in diff_idx:
            lbl = feature_labels[idx]
            rf_v = rf_imp[idx]
            xgb_v = xgb_imp[idx]
            dominant = m1 if rf_v > xgb_v else m2
            diff_text_parts.append(f"{lbl}: {dominant} modeli bu ozelligi daha önemli buluyor ({m1}: {rf_v:.4f}, {m2}: {xgb_v:.4f})")

        return html.Div([
            html.Div(className="panel", children=[
                dcc.Graph(figure=fig, config={"displayModeBar": False})
            ]),
            html.Div(className="panel", style={"marginTop": "16px"}, children=[
                html.Div(className="panel-title", children=[icon("mdi:compare-horizontal", 16), "Karşılaştırma Notlari"]),
                html.Div([
                    html.P("Iki model arasinda en buyuk fark gosteren özellikler:", style={"color": "#475569", "marginBottom": "12px"}),
                    *[html.Div(className="shap-note-item", children=[
                        icon("mdi:circle-small", 16, "#F59E0B"),
                        html.Span(t, style={"color": "#334155", "fontSize": "13px"})
                    ]) for t in diff_text_parts]
                ])
            ])
        ])

    return html.Div()


@callback(Output("shap-importance-chart", "children"),
          Output("shap-importance-text", "children"),
          Input("shap-model-select", "value"),
          prevent_initial_call=False)
def update_shap_importance(model):
    if SHAP_DATA is None or model is None:
        return html.Div(), html.Div()

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    feature_cols = SHAP_DATA.get('feature_cols', [])

    sv_key = f"{model}_shap_values"
    if sv_key not in SHAP_DATA:
        return html.Div(), html.Div()
    shap_vals = SHAP_DATA[sv_key]
    model_name = model

    importance = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(importance)[-10:][::-1]
    top_labels = [feature_labels[i] for i in top_idx]
    top_values = importance[top_idx]

    colors = []
    max_val = top_values[0] if len(top_values) > 0 else 1
    for v in top_values:
        ratio = v / max_val
        if ratio > 0.7:
            colors.append("#3B82F6")
        elif ratio > 0.4:
            colors.append("#06B6D4")
        else:
            colors.append("#64748B")

    fig = go.Figure(go.Bar(
        y=top_labels, x=top_values, orientation='h',
        marker_color=colors, text=[f"{v:.4f}" for v in top_values],
        textposition='outside', textfont=dict(size=11, color='#475569')
    ))
    fig.update_layout(**PLT_LAYOUT, height=450,
                      title=f"{model_name} - En Onemli 10 Özellik (SHAP)",
                      xaxis_title="Ortalama |SHAP Değeri|")
    fig.update_yaxes(autorange="reversed")

    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    feature_explanations = {
        'Varyans': 'Sinyal varyansindaki ani degisimler, uydu alt sistemlerindeki beklenmeyen davranislari yansitir.',
        'Standart Sapma': 'Sinyal dağılıminin genişliği; yüksek sapma operasyonel anomaliye isaret eder.',
        'Fark Varyansi': 'Sinyalin türevindeki değişkenlik, ani geçişleri ve bozulmaları yakalar.',
        '2. Fark Varyansi': 'Sinyalin ikinci türevindeki değişkenlik, ivmelenme anomalilerini gosterir.',
        'Tepe Sayısı': 'Sinyaldeki tepe noktası sayısı; normalden sapma mekanik sorunlara isaret edebilir.',
        'Ortalama Değer': 'Sinyal ortalamasi; kaymalar kalibrasyon sorunlarini gosterir.',
        'RMS Değeri': 'Karekok ortalama sinyal gucu; enerji seviyesindeki anomalileri tespit eder.',
        'Tepeden Tepeye': 'Sinyal genliginin tam araligi; asiri dalgalanmalar anomalidir.',
        'Tepe Faktoru': 'Tepe-RMS oranı; impulsif bozulmaları tespit eder.',
        'Sıfır Geçiş Oranı': 'Sinyalin sıfır cizgisini geçme sıklığı; frekans anomalilerini gosterir.',
        'Basıklık (Kurtosis)': 'Dağılımin sivriliği; yüksek kurtosis ani sapmalara isaret eder.',
        'Çarpıklık (Skewness)': 'Dağılımin asimetrisi; tek yonlu sapmalar anomali belirtisidir.',
        'Segment Süresi': 'Veri segmentinin suresi; beklenmeyen sure anomali gostergesidir.',
        'Segment Uzunluğu': 'Veri noktası sayısı; eksik veya fazla veri anomalidir.',
        'Örnekleme Frekansı': 'Veri toplama hızı; sapma sensor sorunlarini gosterir.',
        'Yumusatilmis Tepe (w=10)': 'Kisa pencere ile yumusatilmis tepe sayısı.',
        'Yumusatilmis Tepe (w=20)': 'Genis pencere ile yumusatilmis tepe sayısı.',
        'Fark Tepe Sayısı': 'Türev sinyalindeki tepe sayısı.',
        '2. Fark Tepe Sayısı': 'Ikinci türev sinyalindeki tepe sayısı.',
        'Boşluk Karesi': 'Veri boşluk karelerinin toplamı; veri kaybi gostergesi.',
        'Agirlikli Uzunluk': 'Sure ile agirliklandirilmis segment uzunlugu.',
        'Varyans/Sure': 'Birim zamandaki varyans; normalize edilmis oynaklik.',
        'Varyans/Uzunluk': 'Veri noktası basina varyans.',
        'Kanal Numarasi': 'Telemetri kanal kimlik numarasi.',
    }

    text_items = []
    for rank, idx in enumerate(top_idx[:3], 1):
        lbl = feature_labels[idx]
        exp = feature_explanations.get(lbl, f"{lbl} ozelligi anomali tespitinde önemli bir rol oynamaktadir.")
        text_items.append(
            html.Div(className="shap-explanation-item", children=[
                html.Div(f"{rank}. {lbl}", className="shap-exp-title"),
                html.Div(f"SHAP Değeri: {importance[idx]:.4f}", className="shap-exp-value"),
                html.Div(exp, className="shap-exp-desc")
            ])
        )

    text_block = html.Div([
        html.Div(className="panel-title", children=[icon("mdi:text-box-outline", 16), "En Onemli Uc Özellik Açıklamasi"]),
        *text_items
    ])

    return chart, text_block


@callback(Output("shap-waterfall-chart", "children"),
          Output("shap-waterfall-text", "children"),
          Input("shap-anomaly-select", "value"),
          Input("shap-explain-model", "value"),
          prevent_initial_call=False)
def update_shap_waterfall(selected_idx, model):
    if SHAP_DATA is None or selected_idx is None or not model:
        return html.Div(), html.Div()

    sv_key = f"{model}_shap_values"
    if sv_key not in SHAP_DATA:
        return html.Div("Seçilen model için SHAP verisi yok.", className="info-box"), html.Div()

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    shap_vals = SHAP_DATA[sv_key]
    expected = SHAP_DATA.get(f'{model}_expected_value', 0)
    X_test = SHAP_DATA['X_test']

    idx = int(selected_idx)
    if idx >= len(shap_vals):
        return html.Div("Gecersiz index."), html.Div()

    vals = shap_vals[idx]
    data_row = X_test[idx]

    abs_vals = np.abs(vals)
    top_idx = np.argsort(abs_vals)[-15:][::-1]
    sorted_labels = [feature_labels[i] for i in top_idx]
    sorted_vals = vals[top_idx]
    sorted_data = data_row[top_idx]

    colors = ['#EF4444' if v > 0 else '#10B981' for v in sorted_vals]

    fig = go.Figure(go.Bar(
        y=sorted_labels, x=sorted_vals, orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in sorted_vals],
        textposition='outside', textfont=dict(size=10, color='#475569')
    ))
    fig.update_layout(**PLT_LAYOUT, height=500,
                      title=f"Anomali Açıklamasi - Segment Index: {idx} ({model})",
                      xaxis_title="SHAP Değeri",
                      annotations=[dict(text="Kirmizi: Anomaliye iter | Yesil: Normale iter",
                                        xref="paper", yref="paper", x=0.5, y=-0.08,
                                        showarrow=False, font=dict(size=11, color="#64748B"))])
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=0, line_dash="dash", line_color="#CBD5E1", line_width=1)

    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    top3_items = []
    for rank, i in enumerate(top_idx[:3], 1):
        lbl = feature_labels[i]
        val = vals[i]
        direction = "anomaliye dogru itiyor" if val > 0 else "normale dogru itiyor"
        color = "#DC2626" if val > 0 else "#16A34A"
        top3_items.append(
            html.Div(className="shap-explanation-item", children=[
                html.Div(f"{rank}. {lbl}", className="shap-exp-title"),
                html.Div([
                    html.Span(f"SHAP: {val:+.4f}", style={"color": color, "fontFamily": "Inter, sans-serif", "fontSize": "13px", "fontWeight": "600"}),
                    html.Span(f" - {direction}", style={"color": "#475569", "fontSize": "13px"}),
                ], className="shap-exp-value"),
                html.Div(f"Özellik değeri: {data_row[i]:.4f}", className="shap-exp-desc")
            ])
        )

    text_block = html.Div([
        html.Div(className="panel-title", children=[icon("mdi:text-box-outline", 16), "En Cok Katkida Bulunan Uc Özellik"]),
        *top3_items
    ])

    return chart, text_block


@callback(Output("shap-mini-waterfall-container", "children"),
          Input("results-table", "selected_rows"),
          State("results-table", "data"),
          prevent_initial_call=True)
def update_mini_waterfall(selected_rows, table_data):
    if not selected_rows or not table_data or SHAP_DATA is None:
        return html.Div()

    row = table_data[selected_rows[0]]
    segment_no = row.get("Segment", 0)

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    shap_vals = SHAP_DATA.get('rf_shap_values', SHAP_DATA.get('RandomForest_shap_values'))
    if shap_vals is None:
        return html.Div()
    X_test = SHAP_DATA['X_test']
    y_test = SHAP_DATA['y_test']

    anomaly_indices = np.where(y_test == 1)[0]
    row_no = row.get("NO", 1) - 1
    if row_no >= len(anomaly_indices):
        return html.Div("Bu segment icin SHAP verisi bulunamadi.", style={"color": "#F59E0B", "padding": "12px"})

    idx = anomaly_indices[row_no]
    if idx >= len(shap_vals):
        return html.Div("SHAP index araligi disinda.", style={"color": "#F59E0B", "padding": "12px"})

    vals = shap_vals[idx]
    abs_vals = np.abs(vals)
    top_idx = np.argsort(abs_vals)[-10:][::-1]
    sorted_labels = [feature_labels[i] for i in top_idx]
    sorted_vals = vals[top_idx]

    colors = ['#EF4444' if v > 0 else '#10B981' for v in sorted_vals]

    fig = go.Figure(go.Bar(
        y=sorted_labels, x=sorted_vals, orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in sorted_vals],
        textposition='outside', textfont=dict(size=10, color='#475569')
    ))
    fig.update_layout(**PLT_LAYOUT, height=350,
                      title=f"SHAP Açıklamasi - Segment {segment_no}",
                      xaxis_title="SHAP Değeri")
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=0, line_dash="dash", line_color="#CBD5E1", line_width=1)

    return html.Div(className="panel", children=[
        html.Div(className="panel-title", children=[icon("mdi:brain", 16), f"SHAP Anomali Açıklamasi - Segment {segment_no}"]),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div("Kirmizi: Anomaliye iter | Yesil: Normale iter",
                 style={"color": "#64748B", "fontSize": "11px", "textAlign": "center", "marginTop": "8px"})
    ])
