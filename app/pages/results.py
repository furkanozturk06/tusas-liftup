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
                            ANALYSIS_PRESETS, channel_label)
from core.state import (MODELS, THRESHOLDS, SCALER, TEST_DATA, ALL_METRICS, FEATURE_COLS,
                        LIVE_DATA, SHAP_DATA, get_tree_explainer, best_model)


def page_results():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Sonuçlar", className="page-title"),
            html.Div("Anomali tespit sonuçları ve görselleştirme", className="page-subtitle")]),
        html.Div(id="results-content")
    ])


def _table_panel(table_data):
    return html.Div(className="panel", children=[
        html.Div(className="panel-title", children=[icon("mdi:format-list-bulleted", 16),
                                                    f"Anomali Listesi ({len(table_data)} kayıt)"]),
        dash_table.DataTable(
            id="results-table",
            columns=[{"name": c, "id": c} for c in ["NO", "Segment", "Kanal", "Skor", "Şiddet", "Detay"]],
            data=table_data, page_size=12, row_selectable="single",
            style_header={"backgroundColor": "#EEF2F8", "color": "#64748B", "fontWeight": "600", "border": "1px solid #E2E8F0", "fontSize": "11px"},
            style_cell={"backgroundColor": "#FFFFFF", "color": "#1E293B", "border": "1px solid #E2E8F0", "fontFamily": "IBM Plex Sans", "fontSize": "12px", "padding": "8px"},
            style_data_conditional=[
                {"if": {"filter_query": '{Şiddet} = "Kritik"'}, "backgroundColor": "rgba(239,68,68,0.08)", "color": "#DC2626"},
                {"if": {"filter_query": '{Şiddet} = "Uyarı"'}, "backgroundColor": "rgba(245,158,11,0.08)", "color": "#D97706"},
                {"if": {"filter_query": '{Şiddet} = "Düşük"'}, "backgroundColor": "rgba(16,185,129,0.08)", "color": "#16A34A"},
                {"if": {"column_id": "Detay"}, "color": "#0284C7", "cursor": "pointer", "textDecoration": "underline", "fontWeight": "bold"},
                {"if": {"row_index": "odd"}, "backgroundColor": "#F4F6FB"},
            ],
            sort_action="native", filter_action="native",
        ),
        html.Div(id="detail-info-msg", className="info-box", style={"marginTop": "15px", "textAlign": "center"},
                 children="Detay görüntülemek için tabloda bir anomali satırına tıklayın."),
        html.Div(style={"marginTop": "12px", "textAlign": "right"}, children=[
            html.Button("CSV Olarak İndir", id="btn-csv-download", n_clicks=0, className="btn-download"),
        ]),
        dcc.Store(id="csv-store", data=table_data),
        html.Div(id="shap-mini-waterfall-container", style={"marginTop": "20px"}),
    ])


@callback(Output("results-content", "children"),
          Input("prediction-results", "data"),
          Input("current-page", "data"),
          State("uploaded-data", "data"),
          prevent_initial_call=True)
def update_results(pred_json, page, data_json):
    if ctx.triggered_id == "current-page" and page != "results":
        return no_update

    if not pred_json or not json.loads(pred_json):
        return html.Div(className="info-box", children=["Henüz analiz yapılmadı."])

    results = json.loads(pred_json)

    if data_json:
        df = pd.read_json(io.StringIO(data_json), orient='split')
    elif os.path.exists(DEMO_PATH):
        df = pd.read_parquet(DEMO_PATH)
    else:
        return html.Div("Veri yok.")

    ensemble_binary = np.zeros(len(df))
    for r in results.values():
        ensemble_binary += np.array(r["preds"])
    ensemble_binary /= max(len(results), 1)
    anom_mask = ensemble_binary > 0
    n_anom = int(anom_mask.sum())
    agreement = sum(1 for r in results.values() for p in r["preds"] if p == 1) / max(len(results) * len(df), 1)

    score_ensemble = np.zeros(len(df))
    n_score_models = 0
    for name, r in results.items():
        sc = np.array(r["scores"])
        sc_min, sc_max = sc.min(), sc.max()
        if sc_max - sc_min > 1e-10:
            sc_n = (sc - sc_min) / (sc_max - sc_min)
        else:
            sc_n = np.zeros_like(sc)
        score_ensemble += sc_n
        n_score_models += 1
    if n_score_models > 0:
        score_ensemble /= n_score_models

    avg_score = float(np.mean(score_ensemble[anom_mask])) if n_anom > 0 else 0

    fig_scores = go.Figure()
    clrs = ["#3B82F6","#10B981","#EF4444","#F59E0B","#8B5CF6","#06B6D4","#F778A1","#A78BFA","#FB923C"]
    for i, (name, r) in enumerate(results.items()):
        sc = np.array(r["scores"])
        sc_n = (sc - sc.min()) / (sc.max() - sc.min() + 1e-10)
        fig_scores.add_trace(go.Scatter(y=sc_n, mode="lines", name=name, line=dict(color=clrs[i%len(clrs)], width=1.5)))
    in_region = False; start = 0
    for i in range(len(score_ensemble)):
        if anom_mask[i] and not in_region: start = i; in_region = True
        elif (not anom_mask[i] or i == len(score_ensemble)-1) and in_region:
            fig_scores.add_vrect(x0=start, x1=i, fillcolor="rgba(239,68,68,0.08)", line_width=0, layer="below")
            in_region = False
    fig_scores.update_layout(**PLT_LAYOUT, height=400, title="Anomali Skorları (Normalize)",
                              yaxis_title="Normalize Anomali Skoru", xaxis_title="Segment")

    anom_indices = np.where(anom_mask)[0]
    
    table_data = []
    for row_no, idx in enumerate(anom_indices, 1):
        sev = "Kritik" if score_ensemble[idx] > 0.8 else "Uyarı" if score_ensemble[idx] > 0.5 else "Düşük"
        ch = df.iloc[idx].get("channel", "N/A") if "channel" in df.columns else "N/A"
        table_data.append({"NO": row_no, "Segment": int(df.iloc[idx].get("segment", idx)),
                           "Kanal": channel_label(ch), "_channel": ch, "Skor": f"{score_ensemble[idx]:.2f}",
                           "Şiddet": sev, "Detay": "İncele", "_idx": int(idx)})

    n_shown = len(table_data)
    n_crit = sum(1 for r in table_data if r["Şiddet"] == "Kritik")
    n_warn = sum(1 for r in table_data if r["Şiddet"] == "Uyarı")
    n_low = sum(1 for r in table_data if r["Şiddet"] == "Düşük")

    return html.Div([
        stat_strip([
            ("Analiz Edilen", len(df), None, "blue"),
            ("Tespit Edilen", n_shown, None, "red"),
            ("Ortalama Skor", f"{avg_score:.3f}", None, "yellow"),
            ("Model Uzlaşması", f"%{agreement*100:.1f}", None, "green"),
        ]),
        html.Div(className="panel mb-4", children=[dcc.Graph(figure=fig_scores, config={"displayModeBar": False})]),
        dbc.Row([
            dbc.Col(html.Div(className="metric-card red", style={"padding":"12px"}, children=[
                html.Span(f"{n_crit}", style={"fontSize":"20px","fontWeight":"700","fontFamily":"Inter, sans-serif"}),
                html.Span(" Kritik", style={"color":"#DC2626","fontSize":"12px","marginLeft":"6px"})]), md=4),
            dbc.Col(html.Div(className="metric-card yellow", style={"padding":"12px"}, children=[
                html.Span(f"{n_warn}", style={"fontSize":"20px","fontWeight":"700","fontFamily":"Inter, sans-serif"}),
                html.Span(" Uyarı", style={"color":"#D97706","fontSize":"12px","marginLeft":"6px"})]), md=4),
            dbc.Col(html.Div(className="metric-card green", style={"padding":"12px"}, children=[
                html.Span(f"{n_low}", style={"fontSize":"20px","fontWeight":"700","fontFamily":"Inter, sans-serif"}),
                html.Span(" Düşük", style={"color":"#16A34A","fontSize":"12px","marginLeft":"6px"})]), md=4),
        ], className="mb-3 g-3"),
        _table_panel(table_data),
    ])


@callback(Output("selected-anomaly", "data"), Output("anomaly-list", "data"),
          Output("current-page", "data", allow_duplicate=True), Output("detail-info-msg", "children"),
          Input("results-table", "active_cell"), State("results-table", "data"), prevent_initial_call=True)
def select_anomaly(active_cell, data):
    if not active_cell or not data: return no_update, no_update, no_update, "Detay görüntülemek için tabloda bir anomali satırına tıklayın."
    selected = data[active_cell["row"]]
    return selected, data, "detail", no_update


@callback(Output("download-csv", "data"),
          Input("btn-csv-download", "n_clicks"),
          State("csv-store", "data"),
          prevent_initial_call=True)
def download_csv(n, data):
    if not n or not data: return no_update
    df_out = pd.DataFrame(data)
    return dcc.send_data_frame(df_out.to_csv, "anomali_sonuclari.csv", index=False)
