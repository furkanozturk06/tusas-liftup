"""Anomali Detay sayfasi: layout + callback'ler."""
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


def page_detail():
    return html.Div()


@callback(Output("detail-page-content", "children"),
          Input("selected-anomaly", "data"),
          Input("current-page", "data"),
          State("anomaly-list", "data"),
          State("uploaded-data", "data"),
          prevent_initial_call=False)
def render_anomaly_detail(selected, current_page, all_anomalies, data_json):
    if current_page != "detail":
        return no_update
    if not selected:
        return html.Div(className="info-box", style={"marginTop":"50px"}, children=["Detaylarını görmek istediğiniz anomaliyi Sonuçlar sayfasındaki tablodan seçiniz."])
    
    seg = selected.get("Segment", 0)
    ch = selected.get("_channel") or selected.get("Kanal", "N/A")
    score = selected.get("Skor", 0)
    sev = selected.get("Şiddet", "Bilinmiyor")
    idx = selected.get("_idx", 0)
    row_no = selected.get("NO", 0)
    
    badge_color = "#FF3B5C" if sev == "Kritik" else "#D97706" if sev == "Uyarı" else "#16A34A"
    
    header = html.Div(className="anomaly-detail-header", children=[
        html.Div(children=[
            html.Div("ANOMALİ DETAY", className="detail-header-title"),
            html.Div(f"SEGMENT  #{seg}", className="detail-header-segment")
        ]),
        html.Div(className="detail-header-right", children=[
            html.Div(sev.upper(), className="severity-badge", style={"borderColor": badge_color, "color": badge_color}),
            html.Div(f"SKOR  {score}", className="score-display"),
            html.Div("MODEL  Topluluk", className="model-display")
        ])
    ])
    
    metrics = stat_strip([
        ("Segment Numarası", seg, None, "blue"),
        ("Kanal Adı", channel_label(ch), None, "blue"),
        ("Anomali Skoru", score, None, "blue"),
        ("Şiddet Seviyesi", sev, None, "red" if sev == "Kritik" else "yellow"),
        ("Tespit Eden", "1+", None, "green"),
    ])
    
    context_fig = go.Figure()
    context_fig.update_layout(**PLT_LAYOUT, height=350, title="Anomali Bağlamı (±100 Segment)", xaxis_title="Segment", yaxis_title="Sinyal")
    
    stats_table_content = html.Div("Veri yüklenemedi.", className="info-box")
    
    if not LIVE_DATA.empty and ch != "N/A":
        ch_data = LIVE_DATA[LIVE_DATA['channel'] == ch].reset_index(drop=True)
        if not ch_data.empty:
            start_idx = max(0, seg - 100)
            end_idx = min(len(ch_data) - 1, seg + 100)
            ctx_df = ch_data.iloc[start_idx:end_idx+1]
            
            context_fig.add_trace(go.Scatter(x=ctx_df['segment'], y=ctx_df['value'], mode='lines', line=dict(color='#64748B', width=1.5), name='Sinyal'))
            
            anom_df = ch_data[ch_data['segment'] == seg]
            if not anom_df.empty:
                val = anom_df['value'].values[0]
                context_fig.add_trace(go.Scatter(x=[seg], y=[val], mode='markers', marker=dict(color='#FF3B5C', size=10), name='Anomali'))
                context_fig.add_vrect(x0=seg-1, x1=seg+1, fillcolor="rgba(239,68,68,0.08)", line_width=1, line_dash="dash", line_color="#FF3B5C")
    
    df = None
    if data_json:
        df = pd.read_json(io.StringIO(data_json), orient='split')
    elif os.path.exists(DEMO_PATH):
        df = pd.read_parquet(DEMO_PATH)
        
    row_feats = {}
    shap_vals = []
    shap_feats = []
    
    if df is not None and idx < len(df):
        row_data = df.iloc[idx]
        if FEATURE_COLS:
            table_rows = []
            for feat in FEATURE_COLS:
                if feat in df.columns:
                    val = row_data[feat]
                    mean_val = df[feat].mean()
                    std_val = df[feat].std()
                    diff = val - mean_val
                    z_score = diff / (std_val + 1e-9)
                    
                    if abs(z_score) > 2:
                        color = "#FF3B5C" if z_score > 0 else "#16A34A"
                        sign = "+" if z_score > 0 else ""
                        diff_str = f"{sign}{z_score:.1f}σ"
                    else:
                        color = "#64748B"
                        diff_str = "Normal"
                        
                    table_rows.append(html.Tr([
                        html.Td(feat), html.Td(f"{mean_val:.2f}"), html.Td(f"{val:.2f}"), html.Td(diff_str, style={"color": color, "fontWeight": "bold" if abs(z_score)>2 else "normal"})
                    ]))
            
            stats_table_content = html.Table(className="custom-table", children=[
                html.Thead(html.Tr([html.Th("Özellik"), html.Th("Normal Ort."), html.Th("Bu Segment"), html.Th("Fark")])),
                html.Tbody(table_rows)
            ])
            
            if "XGBoost" in MODELS:
                try:
                    xgb_model = MODELS["XGBoost"]
                    X_row = row_data[FEATURE_COLS].to_frame().T
                    explainer = get_tree_explainer(xgb_model)
                    sv = explainer.shap_values(X_row)[0]
                    
                    if len(sv.shape) > 1:
                        sv = sv[:, 1]
                    
                    for f, s, v in zip(FEATURE_COLS, sv, X_row.values[0]):
                        shap_vals.append(float(s))
                        shap_feats.append(f)
                        row_feats[f] = v
                except Exception as e:
                    print("SHAP Error:", e)

    signal_analysis = dbc.Row([
        dbc.Col(html.Div(className="panel", children=[
            dcc.Graph(figure=context_fig, config={"displayModeBar": False})
        ]), md=7),
        dbc.Col(html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[icon("mdi:table-compare", 16), " İstatistik Karşılaştırma"]),
            html.Div(stats_table_content, style={"maxHeight": "300px", "overflowY": "auto"})
        ]), md=5)
    ], className="mb-4 g-3")
    
    shap_section = html.Div("Bu model için SHAP değerleri hesaplanamadı.", className="info-box")
    
    if shap_vals and len(shap_vals) > 0:
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
        top_feats = [shap_feats[i] for i in sorted_idx]
        top_shaps = [shap_vals[i] for i in sorted_idx]
        
        colors = ["#FF3B5C" if s > 0 else "#16A34A" for s in top_shaps]
        
        shap_fig = go.Figure()
        shap_fig.add_trace(go.Bar(
            y=top_feats[::-1], x=top_shaps[::-1], orientation='h',
            marker_color=colors[::-1]
        ))
        shap_fig.update_layout(**PLT_LAYOUT, height=350, title="Bu Anomaliye Katkıda Bulunan Özellikler", 
                               margin=dict(l=10, r=20, t=50, b=30), yaxis=dict(tickmode="linear"))
        
        top_positive = [f for f, s in zip(top_feats, top_shaps) if s > 0]
        if len(top_positive) > 0:
            f1 = top_positive[0]
            desc_text = f"Bu segment anomali olarak tespit edildi. Tespitin birincil nedeni '{f1}' değerindeki anormal sapmadır. "
            if len(top_positive) > 1:
                desc_text += f"Buna ek olarak '{top_positive[1]}' özelliği de anomali kararını desteklemiştir."
        else:
            desc_text = "Bu segmentteki anomali kararı birçok özelliğin küçük sapmalarının birleşimiyle alınmıştır."
            
        shap_section = dbc.Row([
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=shap_fig, config={"displayModeBar": False})]), md=6),
            dbc.Col(html.Div(className="panel", style={"height": "100%"}, children=[
                html.Div("NEDEN ANOMALİ?", style={"fontSize": "11px", "letterSpacing": "2px", "color": "#94A3B8", "fontWeight": "bold", "marginBottom": "15px"}),
                html.P(desc_text, style={"fontSize": "14px", "lineHeight": "1.6", "color": "#1E293B"}),
                html.Div(style={"marginTop": "20px"}, children=[
                    html.Div(className="shap-feat-card", style={"borderLeft": "4px solid #FF3B5C" if top_shaps[0]>0 else "4px solid #16A34A"}, children=[
                        html.Div(top_feats[0], style={"fontWeight": "bold"}),
                        html.Div(f"SHAP: {top_shaps[0]:.3f}", style={"fontFamily": "Inter, sans-serif", "color": "#FF3B5C" if top_shaps[0]>0 else "#16A34A"})
                    ])
                ])
            ]), md=6)
        ], className="mb-4 g-3")
        
    action_panel = html.Div(className="panel anomaly-action-panel", children=[
        html.Div(className="nav-buttons", children=[
            html.Button([icon("mdi:chevron-left"), " Önceki Anomali"], id="btn-prev-anomaly", className="btn-nav"),
            html.Button(["Sonraki Anomali ", icon("mdi:chevron-right")], id="btn-next-anomaly", className="btn-nav")
        ]),
        html.Div(f"{row_no} / {len(all_anomalies) if all_anomalies else '?'} Anomali", className="nav-counter"),
        html.Div(className="action-buttons", children=[
            html.Button("Sonuçlara Dön", id="btn-back-results", className="btn-nav")
        ])
    ])
    
    return html.Div([header, metrics, signal_analysis, shap_section, action_panel])


@callback(Output("selected-anomaly", "data", allow_duplicate=True),
          Input("btn-prev-anomaly", "n_clicks"),
          Input("btn-next-anomaly", "n_clicks"),
          State("selected-anomaly", "data"),
          State("anomaly-list", "data"),
          prevent_initial_call=True)
def navigate_anomaly(n_prev, n_next, current, anomaly_list):
    if not current or not anomaly_list: return no_update
    trig = ctx.triggered_id

    # NO (liste içi benzersiz sıra) ile eşleştir; canlı anomalilerde aynı segment
    # birden çok kez geçebileceğinden _idx tek başına ayırt edici değildir.
    key = "NO" if current.get("NO") is not None else "_idx"
    current_idx = -1
    for i, a in enumerate(anomaly_list):
        if a.get(key) == current.get(key):
            current_idx = i
            break
            
    if trig == "btn-prev-anomaly" and current_idx > 0:
        return anomaly_list[current_idx - 1]
    elif trig == "btn-next-anomaly" and current_idx < len(anomaly_list) - 1 and current_idx != -1:
        return anomaly_list[current_idx + 1]
    
    return no_update


@callback(Output("current-page", "data", allow_duplicate=True),
          Input("btn-back-results", "n_clicks"), prevent_initial_call=True)
def back_to_results(n):
    if n: return "results"
    return no_update
