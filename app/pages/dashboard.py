"""Operasyon Paneli sayfasi: layout + callback'ler."""
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
                            ANALYSIS_PRESETS, channel_label, CANONICAL_MODEL_COUNT)
from core.state import (MODELS, THRESHOLDS, SCALER, TEST_DATA, ALL_METRICS, FEATURE_COLS,
                        LIVE_DATA, SHAP_DATA, get_tree_explainer, best_model)


def page_dashboard():
    """Operasyon paneli — anomali tespit ekibi için telemetri sağlık durumu ve alarm özeti.
    (Model performans/benchmark görünümü Geliştirici > Model Performans sayfasındadır.)"""
    df_seg = pd.read_parquet(DEMO_PATH) if os.path.exists(DEMO_PATH) else pd.DataFrame()
    n_seg = len(df_seg)
    has_anom = 'anomaly' in df_seg.columns and n_seg > 0
    has_ch = 'channel' in df_seg.columns and n_seg > 0
    n_anomaly = int(df_seg['anomaly'].sum()) if has_anom else 0
    anom_ratio = f"%{df_seg['anomaly'].mean()*100:.1f}" if has_anom else "N/A"
    n_channels = int(df_seg['channel'].nunique()) if has_ch else 0
    n_raw = len(LIVE_DATA)

    # ── Kanal sağlığı: kanal başına anomali oranı (hangi sensör sorunlu) ──
    fig_health = go.Figure()
    if has_anom and has_ch:
        ch = df_seg.groupby('channel')['anomaly'].agg(['sum', 'count'])
        ch['rate'] = 100 * ch['sum'] / ch['count']
        ch = ch.sort_values('rate')
        bar_clr = ["#10B981" if r < 15 else "#F59E0B" if r < 30 else "#EF4444" for r in ch['rate']]
        fig_health = go.Figure(go.Bar(
            y=[channel_label(c) for c in ch.index], x=ch['rate'].tolist(), orientation='h', marker_color=bar_clr,
            text=[f"{r:.0f}%  ({int(s)}/{int(c)})" for r, s, c in zip(ch['rate'], ch['sum'], ch['count'])],
            textposition='outside', textfont=dict(size=10, color="#475569")))
        fig_health.update_layout(**PLT_LAYOUT, height=360, title="Kanal Sağlığı — Anomali Oranı (%)",
                                 xaxis_title="Anomali oranı (%)", xaxis_range=[0, max(ch['rate']) * 1.25 + 5])

    # ── Son alarmlar: anomalik segmentler, değişkenliğe (var) göre şiddet ──
    alarm_rows = []
    if has_anom:
        anoms = df_seg[df_seg['anomaly'] == 1].copy()
        if 'var' in df_seg.columns and len(anoms):
            p50, p85 = df_seg['var'].quantile(0.50), df_seg['var'].quantile(0.85)
            anoms = anoms.sort_values('var', ascending=False)
            for _, r in anoms.head(12).iterrows():
                v = r['var']
                sev = ("Kritik", "badge-error") if v >= p85 else (("Uyarı", "badge-warning") if v >= p50 else ("Düşük", "badge-success"))
                seg_id = int(r['segment']) if 'segment' in r else "-"
                chn = channel_label(r['channel']) if 'channel' in r else "-"
                alarm_rows.append((seg_id, chn, f"{v:.3g}", sev))

    def sev_badge(sev):
        return html.Span(sev[0], className=sev[1])

    now = time.strftime("%d.%m.%Y %H:%M")

    # ── Kompakt KPI şeridi (tekli büyük kutular yerine tek satır, profesyonel) ──
    stat_bar = stat_strip([
        ("İzlenen Segment", f"{n_seg:,}", f"{n_raw:,} ham ölçüm", None),
        ("Aktif Anomali", f"{n_anomaly:,}", f"{anom_ratio} oran", "red"),
        ("İzlenen Kanal", n_channels, "Manyetometre + Fotodiyot", None),
        ("Model Envanteri", CANONICAL_MODEL_COUNT, f"{len(MODELS)} çalıştırılabilir", "green"),
    ])

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Operasyon Paneli", className="page-title"),
            html.Div("Telemetri sağlık durumu ve anomali alarm özeti", className="page-subtitle")]),

        stat_bar,

        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[
                dcc.Graph(figure=fig_health, config={"displayModeBar": False})]), md=7),
            dbc.Col(html.Div(className="panel", style={"height": "100%"}, children=[
                html.Div(className="panel-title", children=[icon("mdi:information-outline", 16), "Sistem Durumu"]),
                html.Table(className="log-table", children=[html.Tbody([
                    html.Tr([html.Td("Veri akışı"), html.Td(html.Span("AKTİF", className="badge-success"))]),
                    html.Tr([html.Td("Ham telemetri"), html.Td(f"{n_raw:,} ölçüm")]),
                    html.Tr([html.Td("İzlenen kanal"), html.Td(f"{n_channels} kanal")]),
                    html.Tr([html.Td("Model envanteri"), html.Td(f"{CANONICAL_MODEL_COUNT} kanonik · {len(MODELS)} çalıştırılabilir")]),
                    html.Tr([html.Td("Son güncelleme"), html.Td(now)]),
                ])]),
                html.Div(style={"marginTop": "12px", "fontSize": "11px", "color": "#64748B"},
                         children="Model geliştirme ve benchmark için sol menüde Geliştirici / Araştırma bölümüne bakınız."),
            ]), md=5),
        ], className="mb-4 g-3"),

        html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[
                icon("mdi:bell-alert-outline", 16), f"Son Alarmlar ({len(alarm_rows)} anomali)"]),
            html.Table(className="log-table", children=[
                html.Thead(html.Tr([html.Th(c) for c in ["Segment", "Kanal", "Değişkenlik (var)", "Şiddet"]])),
                html.Tbody([html.Tr([html.Td(seg), html.Td(chn), html.Td(val), html.Td(sev_badge(sev))])
                            for seg, chn, val, sev in alarm_rows]
                           or [html.Tr([html.Td("Anomali kaydı yok", colSpan=4)])])
            ])
        ])
    ])
