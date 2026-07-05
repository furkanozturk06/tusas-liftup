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
from utils.ui import PLT_LAYOUT, icon, metric_card, stat_strip
from core.constants import (DEMO_PATH, LIVE_DATA_PATH, SHAP_PKL, BENCHMARK_METRICS,
                            PRIMARY_METRIC, DROP_COLS, SUP_MODEL_NAMES, UNSUP_MODEL_NAMES,
                            ANALYSIS_PRESETS)
from core.state import (MODELS, THRESHOLDS, SCALER, TEST_DATA, ALL_METRICS, FEATURE_COLS,
                        LIVE_DATA, SHAP_DATA, get_tree_explainer, best_model)


def page_analysis():
    sup = [n for n in SUP_MODEL_NAMES if n in MODELS]
    unsup = [n for n in UNSUP_MODEL_NAMES if n in MODELS]
    extra = [n for n in MODELS if n not in SUP_MODEL_NAMES and n not in UNSUP_MODEL_NAMES]
    unsup = unsup + extra
    def model_option(name):
        f1 = ALL_METRICS.get(name, {}).get("F1", 0)
        return html.Span([name, html.Span(f"F1: {f1:.3f}", className="model-f1-badge")])

    def preset_option(key, p):
        return {"value": key, "label": html.Span([
            html.Span(p["title"], style={"fontWeight": "600"}),
            html.Div(p["desc"], style={"fontSize": "11px", "color": "#64748B",
                                       "lineHeight": "1.4"}),
        ])}
    default = ANALYSIS_PRESETS["dogru"]
    def_sup = [m for m in default["sup"] if m in MODELS]

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Anomali Analizi", className="page-title"),
            html.Div("Tespit profili seçin ve analizi başlatın", className="page-subtitle")]),
        dbc.Row([
            dbc.Col([html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[icon("mdi:tune-vertical", 16), "Tespit Profili"]),
                dcc.RadioItems(
                    id="preset-select",
                    options=[preset_option(k, p) for k, p in ANALYSIS_PRESETS.items()],
                    value="dogru", className="preset-radio",
                    inputStyle={"marginRight": "8px"},
                    labelStyle={"display": "block", "padding": "8px 0", "cursor": "pointer"}),
                html.Button("Analizi Başlat", id="btn-analyze", n_clicks=0, className="btn-primary",
                            style={"width": "100%", "marginTop": "12px"}),
                html.Div(id="selection-counter", className="selection-counter"),

                html.Details(open=False, style={"marginTop": "16px"}, children=[
                    html.Summary("Gelişmiş · model seçimi", style={
                        "fontSize": "11px", "letterSpacing": "1px", "color": "#94A3B8",
                        "fontWeight": "600", "cursor": "pointer", "userSelect": "none",
                        "outline": "none", "padding": "4px 0"}),
                    html.Div("GÖZETİMLİ", className="section-label"),
                    dcc.Checklist(id="sup-models", options=[{"label": model_option(n), "value": n} for n in sup],
                                  value=def_sup, className="model-checklist", inputStyle={"marginRight": "8px"}),
                    html.Div("GÖZETİMSİZ", className="section-label"),
                    dcc.Checklist(id="unsup-models", options=[{"label": model_option(n), "value": n} for n in unsup],
                                  value=[], className="model-checklist", inputStyle={"marginRight": "8px"}),
                    html.Div("EŞİK ÇARPANI", className="section-label"),
                    dcc.Slider(id="threshold-slider", min=0.5, max=1.5, step=0.05, value=default["thr"],
                               marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5"},
                               tooltip={"placement": "bottom", "always_visible": False}),
                    html.Div("Düşük değer: hassas tespit, yüksek yanlış alarm. Yüksek değer: güvenilir ama az tespit.",
                             style={"fontSize": "11px", "color": "#64748B", "marginTop": "8px", "lineHeight": "1.5"}),
                ]),
            ])], md=3),
            dbc.Col([
                dcc.Loading(
                    id="loading-analysis",
                    type="circle",
                    color="#3B82F6",
                    children=[
                        html.Div(id="analysis-output", className="panel", children=[
                            html.Div(className="info-box", children=[
                                icon("mdi:information-outline", 32, "#3B82F6"), html.Br(), html.Br(),
                                "Sol panelden model seçip analizi başlatınız."])
                        ])
                    ]
                )
            ], md=9)
        ], className="g-3")
    ])


@callback(Output("analysis-output", "children"), Output("prediction-results", "data"),
          Input("btn-analyze", "n_clicks"),
          State("sup-models", "value"), State("unsup-models", "value"),
          State("threshold-slider", "value"), State("uploaded-data", "data"),
          prevent_initial_call=True)
def run_analysis(n, sup_sel, unsup_sel, thresh_mult, data_json):
    if not n: return no_update, no_update
    selected = (sup_sel or []) + (unsup_sel or [])
    if not selected:
        return html.Div("En az bir model seçiniz.", style={"color": "#EF4444"}), no_update

    if data_json:
        df = pd.read_json(io.StringIO(data_json), orient='split')
    elif os.path.exists(DEMO_PATH):
        df = pd.read_parquet(DEMO_PATH)
    else:
        return html.Div("Veri bulunamadı.", style={"color": "#EF4444"}), no_update

    if FEATURE_COLS:
        for c in FEATURE_COLS:
            if c not in df.columns:
                df[c] = 0
        X = df[FEATURE_COLS].fillna(0).values
    else:
        feature_cols = [c for c in df.columns if c not in DROP_COLS]
        X = df[feature_cols].fillna(0).values

    if SCALER:
        try: X = SCALER.transform(X)
        except Exception as e:
            return html.Div(f"Scaler hatası: {e}", style={"color": "#EF4444"}), no_update

    results = {}
    rows = []
    for name in selected:
        if name not in MODELS: continue
        try:
            pr, sc = predict(MODELS[name], name, X, THRESHOLDS, thresh_mult)
            n_anom = int(pr.sum())
            results[name] = {"preds": pr.tolist(), "scores": sc.tolist(), "n_anomaly": n_anom}
            rows.append(html.Div(className="progress-row", children=[
                html.Span(name, className="progress-model-name"),
                dbc.Progress(value=100, color="success", style={"flex": 1, "height": "6px"}),
                html.Span(f"{n_anom} anomali", style={"fontSize": "12px", "color": "#10B981", "width": "100px", "textAlign": "right"}),
            ]))
        except Exception as e:
            rows.append(html.Div(className="progress-row", children=[
                html.Span(name, className="progress-model-name"),
                html.Span(f"Hata: {str(e)[:40]}", style={"fontSize": "12px", "color": "#EF4444"}),
            ]))

    total = sum(r["n_anomaly"] for r in results.values())
    summary = stat_strip([
        ("Başarılı Model", len(results), None, "green"),
        ("Toplam Anomali", total, None, "red"),
        ("Çalışan Model", len(selected), None, "blue"),
    ])

    return html.Div([summary, html.Div(className="panel-title", children=[icon("mdi:format-list-bulleted",16), "Model Sonuçları"]), *rows]), json.dumps(results)


@callback(Output("sup-models", "value"), Output("unsup-models", "value"),
          Output("threshold-slider", "value"),
          Input("preset-select", "value"), prevent_initial_call=True)
def apply_preset(preset):
    p = ANALYSIS_PRESETS.get(preset)
    if not p:
        return no_update, no_update, no_update
    sup = [m for m in p["sup"] if m in MODELS]
    unsup = [m for m in p["unsup"] if m in MODELS]
    return sup, unsup, p["thr"]


@callback(Output("selection-counter", "children"),
          Input("sup-models", "value"), Input("unsup-models", "value"))
def update_counter(sup_sel, unsup_sel):
    ns = len(sup_sel or [])
    nu = len(unsup_sel or [])
    return f"{ns} gözetimli + {nu} gözetimsiz model etkin"
