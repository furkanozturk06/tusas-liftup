"""Canli Izleme sayfasi: layout + callback'ler."""
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
from dash import html, dcc, dash_table, callback, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

from utils.model_loader import predict
from utils.feature_extractor import extract_features_from_raw
from utils.ui import PLT_LAYOUT, icon, metric_card
from core.constants import (DEMO_PATH, LIVE_DATA_PATH, SHAP_PKL, BENCHMARK_METRICS,
                            PRIMARY_METRIC, DROP_COLS, SUP_MODEL_NAMES, UNSUP_MODEL_NAMES,
                            ANALYSIS_PRESETS, channel_label)
from core.state import (MODELS, THRESHOLDS, SCALER, TEST_DATA, ALL_METRICS, FEATURE_COLS,
                        LIVE_DATA, SHAP_DATA, get_tree_explainer, best_model)


def _model_label(m):
    """Model adı + temel performans (operatör hangi modeli seçeceğine karar verebilsin)."""
    mt = ALL_METRICS.get(m, {})
    f1, ap = mt.get("F1"), mt.get("AUC_PR")
    if f1 is not None and ap is not None:
        return f"{m}  ·  F1 {f1:.2f} · AUC-PR {ap:.2f}"
    return m


def _alarm_card(rec):
    """Tespit edilen bir anomali için tıklanabilir detaylı alarm kartı.

    Tıklanınca (Analiz/Sonuçlar tablosundaki gibi) ilgili anomalinin Anomali Detay
    sayfasına gidilir; karta değer / okuma no / model bilgisi de gömülüdür.
    """
    sev_class = "critical" if rec.get("score", 0) > 0.8 else "warning"
    val = rec.get("value")
    val_str = f"{val:.4g}" if isinstance(val, (int, float)) else str(val)
    return html.Div(id={"type": "live-alarm", "id": rec.get("id", 0)}, n_clicks=0,
                    className=f"alarm-card {sev_class}", style={"cursor": "pointer"}, children=[
        html.Div(className="alarm-card-top", children=[
            html.Span(str(rec.get("time", "")).split("T")[-1][:8], className="alarm-time"),
            html.Span(rec.get("severity", ""), className="alarm-badge"),
        ]),
        html.Div(className="alarm-card-bottom", children=[
            html.Span(channel_label(rec.get("channel", "")), className="alarm-channel"),
            html.Span(f"Skor: {rec.get('score', 0):.2f}", className="alarm-score"),
        ]),
        html.Div(style={"fontSize": "10px", "color": "#64748B", "marginTop": "4px",
                        "display": "flex", "justifyContent": "space-between", "gap": "6px"}, children=[
            html.Span(f"Değer: {val_str}"),
            html.Span(f"Okuma #{rec.get('reading', '-')}"),
            html.Span(rec.get("model", ""), style={"fontWeight": "600"}),
        ]),
        html.Div("Detayları gör →", style={"fontSize": "9px", "color": "#0284C7",
                                           "marginTop": "3px", "textAlign": "right", "fontWeight": "600"}),
    ])


def _empty_signal_fig():
    f = go.Figure()
    f.update_layout(**PLT_LAYOUT, height=300, xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                    yaxis=dict(showgrid=True, gridcolor="#E2E8F0"))
    f.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#64748B", width=1.5), name="Sinyal"))
    f.add_trace(go.Scatter(x=[], y=[], mode="markers", marker=dict(color="#FF3B5C", size=8), name="Anomali"))
    return f


def _empty_score_fig():
    f = go.Figure()
    f.update_layout(**PLT_LAYOUT, height=150, xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                    yaxis=dict(range=[0, 1.05]))
    f.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#0284C7", width=2),
                           fill='tozeroy', fillcolor='rgba(0,200,255,0.1)', name="Skor"))
    f.add_hline(y=0.5, line_dash="dash", line_color="#FF3B5C")
    return f


def live_detail_records(state):
    """Canlı izleme anomalilerini Sonuçlar/Detay tablosu formatında (NO sıralı)
    kayıtlara çevirir. Sonuçlar sayfası ve alarm-tıklama bu kayıtları paylaşır;
    her anomali gerçekleştiği segmente ve özellik matrisindeki satır konumuna eşlenir."""
    anomalies = (state or {}).get("anomalies", [])
    if not anomalies:
        return []
    seg2idx = {}
    try:
        if os.path.exists(DEMO_PATH):
            fdf = pd.read_parquet(DEMO_PATH)
            if 'segment' in fdf.columns:
                for pos, seg in enumerate(fdf['segment'].values):
                    seg2idx.setdefault(int(seg), pos)
    except Exception:
        pass
    recs = []
    for i, a in enumerate(anomalies, 1):
        seg = int(a.get("segment", 0))
        recs.append({"NO": i, "Segment": seg, "Kanal": channel_label(a.get("channel", "")),
                     "_channel": a.get("channel"), "Skor": f"{a.get('score', 0):.2f}",
                     "Şiddet": "Kritik" if a.get("score", 0) > 0.8 else "Uyarı",
                     "Detay": "İncele", "_idx": seg2idx.get(seg, 0)})
    return recs


def page_live():
    channels = LIVE_DATA['channel'].unique().tolist() if not LIVE_DATA.empty and 'channel' in LIVE_DATA.columns else []
    # Çalıştırılabilir (yüklü) TÜM modeller listelensin: kanonik liste filtresi yerine
    # doğrudan MODELS üzerinden, performansa göre sıralı (en iyi üstte) — böylece hiçbir
    # runnable model dropdown dışında kalmaz.
    live_models = sorted(MODELS, key=lambda n: ALL_METRICS.get(n, {}).get("AUC_PR", 0), reverse=True)
    default_model = next((m for m in ["HistGradientBoosting", "RandomForest", "IsolationForest"] if m in MODELS),
                         live_models[0] if live_models else None)

    # Grafikler boş başlar; Canlı İzleme bir overlay olduğu için (app.layout'ta) DOM'da
    # kalıcıdır — başka sekmeye gidip dönünce grafik/alarm/çalışma durumu korunur.
    fig_sig = _empty_signal_fig()
    fig_score = _empty_score_fig()

    return html.Div(className="live-page-container", children=[
        html.Div(className="page-header", children=[
            html.Div("Canlı İzleme", className="page-title"),
            html.Div("Gerçek zamanlı telemetri akışı ve anında anomali tespiti", className="page-subtitle")
        ]),
        
        html.Div(className="panel live-control-panel", children=[
            html.Div(className="live-controls-left", children=[
                html.Div([
                    html.Label("Kanal:"),
                    dcc.Dropdown(id="live-channel", options=[{"label": channel_label(c), "value": c} for c in channels],
                                 value=channels[0] if channels else None, className="custom-dropdown", clearable=False,
                                 persistence=True, persistence_type="session")
                ], className="control-group"),
                html.Div([
                    html.Label("Model (performans ile):"),
                    dcc.Dropdown(id="live-model", options=[{"label": _model_label(m), "value": m} for m in live_models],
                                 value=default_model, className="custom-dropdown", clearable=False,
                                 persistence=True, persistence_type="session")
                ], className="control-group"),
                html.Div([
                    html.Label("Hız:"),
                    dcc.Dropdown(id="live-speed", options=[
                        {"label": "Yavaş (1x)", "value": 1},
                        {"label": "Normal (5x)", "value": 5},
                        {"label": "Hızlı (20x)", "value": 20}
                    ], value=5, className="custom-dropdown", clearable=False,
                       persistence=True, persistence_type="session")
                ], className="control-group"),
            ]),
            html.Div(className="live-controls-right", children=[
                html.Button([icon("mdi:play", 18), " Başlat"], id="live-start", n_clicks=0, className="btn-primary"),
                html.Button([icon("mdi:stop", 18), " Durdur"], id="live-stop", n_clicks=0, className="btn-error", disabled=True),
                html.Button([icon("mdi:refresh", 18), " Sıfırla"], id="live-reset", n_clicks=0, className="btn-outline"),
            ])
        ]),
        
        html.Div(className="live-status-bar", children=[
            html.Span(id="live-stat-read", children="OKUNAN: 0"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-total", children=f"TOPLAM: {len(LIVE_DATA)}"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-prog", children="%0.0"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-anom", children="ANOMALİ: 0"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-last", children="SON ALARM: Yok"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-model", children="MODEL: -"),
        ]),
        
        html.Div(className="live-main-area", children=[
            html.Div(className="live-charts-area", children=[
                html.Div(className="panel live-chart-panel", children=[
                    html.Div(className="panel-title", style={"display": "flex", "justifyContent": "space-between"}, children=[
                        html.Span([icon("mdi:chart-timeline-variant", 16), " Telemetri Sinyali"]),
                        html.Span(id="live-indicator", children=[html.Span(className="status-dot"), "DURDURULDU"], className="live-indicator-badge")
                    ]),
                    dcc.Graph(id="live-signal-graph", figure=fig_sig, config={"displayModeBar": False}, style={"height": "300px"})
                ]),
                html.Div(className="panel live-chart-panel", style={"marginTop": "16px"}, children=[
                    html.Div(className="panel-title", children=[icon("mdi:chart-bell-curve", 16), " Anomali Skoru"]),
                    dcc.Graph(id="live-score-graph", figure=fig_score, config={"displayModeBar": False}, style={"height": "150px"})
                ]),
            ]),
            
            html.Div(className="live-alarm-panel", children=[
                html.Div("ALARM KAYITLARI", className="alarm-panel-title"),
                html.Div(id="live-alarm-list", className="alarm-list-container", children=[
                    html.Div("Anomali Yok", className="no-alarm-msg")
                ]),
                html.Div(id="live-alarm-count", className="alarm-count-footer", children="0 Alarm")
            ])
        ])
    ])


@callback(
    Output("live-interval", "disabled"),
    Output("live-sim-state", "data"),
    Output("live-start", "disabled"),
    Output("live-stop", "disabled"),
    Output("live-indicator", "children"),
    Output("live-indicator", "className"),
    Output("global-live-dot", "className"),
    Output("live-signal-graph", "figure"),
    Output("live-score-graph", "figure"),
    Output("live-alarm-list", "children"),
    Input("live-start", "n_clicks"),
    Input("live-stop", "n_clicks"),
    Input("live-reset", "n_clicks"),
    State("live-sim-state", "data"),
    prevent_initial_call=True
)
def control_live_sim(start_n, stop_n, reset_n, state):
    ctx_id = ctx.triggered_id
    if ctx_id == "live-start":
        state["is_running"] = True
        ind = [html.Span(className="status-dot"), " CANLI"]
        return False, state, True, False, ind, "live-indicator-badge live-active", "topbar-dot blink", no_update, no_update, no_update
    elif ctx_id == "live-stop":
        state["is_running"] = False
        ind = [html.Span(className="status-dot"), " DURDURULDU"]
        return True, state, False, True, ind, "live-indicator-badge", "topbar-dot slow-blink", no_update, no_update, no_update
    elif ctx_id == "live-reset":
        state = {"index": 0, "is_running": False, "anomalies": [], "scores": [], "channel": None, "model": None}
        ind = [html.Span(className="status-dot"), " DURDURULDU"]
        alarm_msg = html.Div("Anomali Yok", className="no-alarm-msg")
        return (True, state, False, True, ind, "live-indicator-badge", "topbar-dot",
                _empty_signal_fig(), _empty_score_fig(), [alarm_msg])
    return no_update


@callback(
    Output("live-signal-graph", "extendData"),
    Output("live-score-graph", "extendData"),
    Output("live-sim-state", "data", allow_duplicate=True),
    Output("live-alarm-list", "children", allow_duplicate=True),
    Output("live-stat-read", "children"),
    Output("live-stat-prog", "children"),
    Output("live-stat-anom", "children"),
    Output("live-stat-last", "children"),
    Output("live-stat-model", "children"),
    Output("live-alarm-count", "children"),
    Input("live-interval", "n_intervals"),
    State("live-sim-state", "data"),
    State("live-channel", "value"),
    State("live-model", "value"),
    State("live-speed", "value"),
    State("live-alarm-list", "children"),
    prevent_initial_call=True
)
def update_live_sim(n_int, state, channel, model_name, speed, current_alarms):
    if not state.get("is_running", False) or LIVE_DATA.empty or not channel or not model_name:
        return no_update
        
    idx = state["index"]
    df_slice = LIVE_DATA[LIVE_DATA['channel'] == channel]
    
    if idx >= len(df_slice):
        state["is_running"] = False
        return no_update
        
    end_idx = min(idx + speed, len(df_slice))
    chunk = df_slice.iloc[idx:end_idx]
    
    state["index"] = end_idx
    
    times = chunk['timestamp'].tolist()
    vals = chunk['value'].tolist()
    
    start_win = max(0, end_idx - 30)
    win_data = df_slice.iloc[start_win:end_idx]['value'].values

    if FEATURE_COLS and len(win_data) >= 3:
        samp = df_slice['sampling'].iloc[0] if 'sampling' in df_slice.columns else 1
        win_df = pd.DataFrame({'value': win_data, 'segment': 0,
                               'channel': channel, 'sampling': samp})
        feats = extract_features_from_raw(win_df)
        X = feats.reindex(columns=FEATURE_COLS, fill_value=0).fillna(0).values
    elif FEATURE_COLS:
        X = np.zeros((1, len(FEATURE_COLS)))
    else:
        X = np.array([[np.mean(win_data) if len(win_data) else 0,
                       np.std(win_data) if len(win_data) else 0, 0, 0]])

    if SCALER:
        try: X = SCALER.transform(X)
        except Exception as e: print("Live scaler hatası:", e)
        
    model = MODELS.get(model_name)
    if not model: return no_update
    
    try:
        pr, sc = predict(model, model_name, X, THRESHOLDS, 1.0)
        score = sc[0]
        t = THRESHOLDS.get(model_name, 0)
        if t == 0: t = 0.5
        norm_score = max(0, min(1, 0.5 + (score - t)/ (abs(t) + 1e-6)))
        if pr[0] == 1: norm_score = max(norm_score, 0.6)
        is_anom = int(pr[0]) == 1
    except Exception as e:
        print("Prediction error:", e)
        norm_score = 0
        is_anom = False

    # State'i zenginleştir: aktif kanal/model + skor geçmişi (sayfaya dönünce restore_live kullanır)
    state["channel"] = channel
    state["model"] = model_name
    state.setdefault("scores", []).append([times[-1], float(norm_score)])
    state["scores"] = state["scores"][-300:]

    sig_x = [times]
    sig_y = [vals]

    # Anomali iz'ini sinyalle AYNI uzunlukta besle (ayni x'ler, yalniz anomali
    # noktasında y dolu, gerisi None). Boylece iki iz maxpoints=200 ile ayni
    # pencerede birlikte kayar; eski anomali noktaları cizgiyle birlikte soldan
    # dusulur (aksi halde stale anomali isaretleri x-eksenini gererek cizgiyi
    # "soldan kayboluyor" gibi gosteriyordu).
    anom_marks = [None] * len(times)
    if is_anom and times:
        anom_marks[-1] = vals[-1]
    anom_x = [times]
    anom_y = [anom_marks]

    sig_update = (dict(x=sig_x + anom_x, y=sig_y + anom_y), [0, 1], 200)
    score_update = (dict(x=[[times[-1]]], y=[[norm_score]]), [0], 200)
    
    alarms = current_alarms if isinstance(current_alarms, list) and not getattr(current_alarms[0], 'props', {}).get('className', '') == 'no-alarm-msg' else []
    
    if is_anom:
        sev_text = "KRİTİK" if norm_score > 0.8 else "UYARI"
        seg_id = int(df_slice.iloc[end_idx - 1]['segment']) if 'segment' in df_slice.columns and end_idx > 0 else end_idx
        state["anom_seq"] = state.get("anom_seq", 0) + 1
        rec = {"id": state["anom_seq"], "time": times[-1], "channel": channel, "segment": seg_id,
               "value": float(vals[-1]), "score": float(norm_score), "severity": sev_text,
               "reading": end_idx, "model": model_name}
        state["anomalies"].append(rec)
        alarms.insert(0, _alarm_card(rec))
        alarms = alarms[:20]
        
    if not alarms:
        alarms = [html.Div("Anomali Yok", className="no-alarm-msg")]
        
    prog = (end_idx / len(df_slice)) * 100 if len(df_slice) > 0 else 0
    n_anom = len(state["anomalies"])
    last_anom = state["anomalies"][-1]["time"].split("T")[-1][:8] if n_anom > 0 else "Yok"
    
    return (
        sig_update, score_update, state, alarms,
        f"OKUNAN: {end_idx}", f"%{prog:.1f}", f"ANOMALİ: {n_anom}", f"SON ALARM: {last_anom}", f"MODEL: {model_name}",
        f"{n_anom} Alarm"
    )


@callback(
    Output("selected-anomaly", "data", allow_duplicate=True),
    Output("anomaly-list", "data", allow_duplicate=True),
    Output("current-page", "data", allow_duplicate=True),
    Input({"type": "live-alarm", "id": ALL}, "n_clicks"),
    State("live-sim-state", "data"),
    prevent_initial_call=True,
)
def open_live_anomaly_detail(clicks, state):
    """Bir canlı alarm kartına tıklanınca o anomalinin Anomali Detay sayfasını açar.

    Canlı anomaliler Sonuçlar listesini ve detay gezinme sırasını ETKİLEMEZ:
    anomaly-list yalnızca tıklanan tek anomaliyi içerir, böylece detaydaki
    Önceki/Sonraki tuşları canlı izleme anomalilerini dolaşmaz (1/1 gösterir).
    Yalnızca gerçek bir tıklamada çalışır: izleme sürerken yeni alarm kartı
    eklenmesi (n_clicks=0) callback'i tetiklese de işlem yapılmaz."""
    trig = ctx.triggered
    if not trig or not trig[0].get("value") or not ctx.triggered_id:
        return no_update, no_update, no_update
    target_id = ctx.triggered_id.get("id")
    anomalies = (state or {}).get("anomalies", [])
    recs = live_detail_records(state)
    sel = next((recs[i] for i, a in enumerate(anomalies)
                if a.get("id") == target_id and i < len(recs)), None)
    if sel is None:
        return no_update, no_update, no_update
    sel = dict(sel); sel["NO"] = 1   # tek öğeli liste; sayaç "1 / 1" gösterir
    return sel, [sel], "detail"
