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
                            ANALYSIS_PRESETS, DEEP_SEQ_MODELS, ESA_ADB_BASELINES,
                            CANONICAL_MODEL_COUNT, model_category)
from core.state import (MODELS, THRESHOLDS, SCALER, TEST_DATA, ALL_METRICS, FEATURE_COLS,
                        LIVE_DATA, SHAP_DATA, get_tree_explainer, best_model)

_PERF_FIGS_CACHE = None


def build_performance_figures(top_n=10):
    global _PERF_FIGS_CACHE
    if _PERF_FIGS_CACHE is not None:
        return _PERF_FIGS_CACHE
    if not TEST_DATA:
        return html.Div("Test verisi bulunamadı.", className="info-box")
    from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                 average_precision_score, confusion_matrix)
    import numpy as _np
    X_t, y_t = TEST_DATA["X_test"], _np.asarray(TEST_DATA["y_test"])
    ranked = [n for n in sorted(ALL_METRICS, key=lambda n: ALL_METRICS[n].get(PRIMARY_METRIC, 0),
                                reverse=True) if n in MODELS]
    top = ranked[:top_n]
    clrs = ["#3B82F6","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4","#F778A1","#A78BFA","#FB923C","#22D3EE"]
    preds = {}
    for name in top:
        try:
            pr, sc = predict(MODELS[name], name, X_t, THRESHOLDS, 1.0)
            preds[name] = (_np.asarray(pr), _np.asarray(sc))
        except Exception as e:
            print(f"Performans tahmini başarısız ({name}):", e)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 line=dict(dash="dash", color="#CBD5E1"), showlegend=False))
    fig_pr = go.Figure()
    base = float(y_t.mean()) if len(y_t) else 0
    fig_pr.add_hline(y=base, line_dash="dash", line_color="#CBD5E1",
                     annotation_text=f"taban {base:.2f}", annotation_font_color="#64748B")
    for i, name in enumerate(top):
        if name not in preds:
            continue
        _, sc = preds[name]
        c = clrs[i % len(clrs)]
        fpr, tpr, _ = roc_curve(y_t, sc); a = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({a:.3f})",
                                     line=dict(color=c, width=2)))
        prec, rec, _ = precision_recall_curve(y_t, sc); ap = average_precision_score(y_t, sc)
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{name} ({ap:.3f})",
                                    line=dict(color=c, width=2)))
    fig_roc.update_layout(**PLT_LAYOUT, height=420, title=f"ROC Eğrileri (en iyi {len(top)})",
                          xaxis_title="FPR", yaxis_title="TPR")
    fig_pr.update_layout(**PLT_LAYOUT, height=420, title=f"PR Eğrileri (en iyi {len(top)}) · birincil ölçüt",
                         xaxis_title="Recall", yaxis_title="Precision")

    conf_cols = []
    for name in top[:3]:
        if name not in preds:
            continue
        pr, _ = preds[name]
        cm = confusion_matrix(y_t, pr)
        labels = ["Normal", "Anomali"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=[f"Tahmin {l}" for l in labels], y=[f"Gerçek {l}" for l in labels],
            text=cm, texttemplate="%{text}", textfont=dict(size=16),
            colorscale="Blues", showscale=False))
        fig_cm.update_layout(**PLT_LAYOUT, height=300, title=f"{name}")
        conf_cols.append(dbc.Col(html.Div(className="panel", children=[
            dcc.Graph(figure=fig_cm, config={"displayModeBar": False})]), md=4))

    component = html.Div([
        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_pr, config={"displayModeBar": False})]), md=6),
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_roc, config={"displayModeBar": False})]), md=6),
        ], className="mb-4 g-3"),
        html.Div(className="panel-title", children=[icon("mdi:grid", 16), "Confusion Matrisleri (en iyi 3 model)"]),
        dbc.Row(conf_cols, className="g-3"),
    ])
    _PERF_FIGS_CACHE = component
    return component


def page_performance():
    if not ALL_METRICS:
        return html.Div("Metrik verisi bulunamadi.")

    mdf = pd.DataFrame(ALL_METRICS).T
    cols = [c for c in BENCHMARK_METRICS + ["FAR"] if c in mdf.columns]
    ranked = sorted(ALL_METRICS, key=lambda n: ALL_METRICS[n].get(PRIMARY_METRIC, 0), reverse=True)

    top = ranked[:6]
    cats = ["Accuracy","Precision","Recall","F1","MCC","AUC_ROC","AUC_PR"]
    fig_radar = go.Figure()
    for n in top:
        vals = [ALL_METRICS[n].get(c, 0) for c in cats]
        fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], name=n, fill="toself", opacity=0.5))
    fig_radar.update_layout(**PLT_LAYOUT, height=400, polar=dict(bgcolor="#F4F6FB", radialaxis=dict(range=[0,1], showticklabels=True, tickfont=dict(size=10))))

    n_metric = len(ALL_METRICS)
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Model Performans", className="page-title"),
            html.Div(f"{CANONICAL_MODEL_COUNT} kanonik modelin karşılaştırmalı analizi "
                     f"({n_metric} Ψ-ölçümlü + {len(ESA_ADB_BASELINES)} ESA-ADB literatür)",
                     className="page-subtitle")]),
        html.Div(className="panel mb-4", children=[
            html.Div(className="panel-title", children=[icon("mdi:table", 16),
                     f"Metrik Tablosu ({n_metric} model, AUC_PR sıralı)"]),
            dash_table.DataTable(
                columns=([{"name": "Model", "id": "Model"}, {"name": "Kategori", "id": "Kategori"}]
                         + [{"name": c, "id": c} for c in cols]),
                data=[{"Model": n, "Kategori": model_category(n),
                       **{c: f"{ALL_METRICS[n].get(c,0):.4f}" for c in cols}} for n in ranked],
                style_header={"backgroundColor": "#EEF2F8", "color": "#64748B", "fontWeight": "600",
                               "border": "1px solid #E2E8F0", "textTransform": "uppercase", "fontSize": "11px"},
                style_cell={"backgroundColor": "#FFFFFF", "color": "#1E293B", "border": "1px solid #E2E8F0",
                             "fontFamily": "IBM Plex Sans", "fontSize": "12.5px", "padding": "10px"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#F4F6FB"},
                ] + [
                    cond
                    for col in ["AUC_PR", "AUC_ROC", "F1", "MCC"] if col in cols
                    for cond in [
                        {"if": {"filter_query": f'{{{col}}} > 0.95', "column_id": col}, "color": "#059669", "fontWeight": "600"},
                        {"if": {"filter_query": f'{{{col}}} > 0.80 && {{{col}}} <= 0.95', "column_id": col}, "color": "#D97706"},
                        {"if": {"filter_query": f'{{{col}}} <= 0.80', "column_id": col}, "color": "#FF3B5C"},
                    ]
                ],
            )
        ]),
        html.Div(className="panel mb-4", children=[
            html.Div(className="panel-title", children=[icon("mdi:book-open-variant", 16),
                     "ESA-ADB Literatür Baseline'ları (Ψ-dışı referans)"]),
            html.Div(className="info-box", style={"marginBottom": "12px"},
                     children="Aynı ESA uydu telemetrisi alanından; OPS-SAT Ψ test setinde "
                              "çalıştırılmadı, ayrı ESA-ADB benchmark'ında raporlanır. Bu yüzden "
                              "yukarıdaki nicel metrik tablosunda yer almaz, referans olarak listelenir."),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in ["Model", "Tip", "Kaynak"]],
                data=[{"Model": b["name"], "Tip": b["type"], "Kaynak": b["source"]}
                      for b in ESA_ADB_BASELINES],
                style_header={"backgroundColor": "#EEF2F8", "color": "#64748B", "fontWeight": "600",
                               "border": "1px solid #E2E8F0", "textTransform": "uppercase", "fontSize": "11px"},
                style_cell={"backgroundColor": "#FFFFFF", "color": "#1E293B", "border": "1px solid #E2E8F0",
                             "fontFamily": "IBM Plex Sans", "fontSize": "12.5px", "padding": "10px",
                             "textAlign": "left", "whiteSpace": "normal", "height": "auto"},
            )
        ]),
        html.Div(className="panel mb-4", children=[dcc.Graph(figure=fig_radar, config={"displayModeBar": False})]),
        dcc.Loading(id="loading-roc", type="circle", color="#3B82F6",
                    children=html.Div(id="performance-roc", children=[
                        html.Div([icon("mdi:chart-bell-curve-cumulative", 28, "#3B82F6"),
                                  html.Br(), html.Br(),
                                  "Performans grafikleri hesaplanıyor (ROC + PR + confusion)..."],
                                 className="info-box", style={"textAlign": "center"})]))
    ])


@callback(Output("performance-roc", "children"), Input("current-page", "data"))
def load_performance_roc(page_id):
    if page_id != "performance":
        return no_update
    return build_performance_figures()
