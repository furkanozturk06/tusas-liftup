"""
Sentetik Veri Laboratuvari Sayfasi
===================================
ESA OPS-SAT benzeri sentetik uydu telemetri verisi üretimi,
görselleştirilmesi ve analiz pipeline'ina aktarimi.
"""

import os, sys, io, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from utils.ui import PLT_LAYOUT, icon as _icon, metric_card as _metric_card, stat_strip

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

CHANNEL_INFO = {
    "CADC0872": {"type": "Manyetometre", "axis": "X", "color": "#3B82F6"},
    "CADC0873": {"type": "Manyetometre", "axis": "Y", "color": "#10B981"},
    "CADC0874": {"type": "Manyetometre", "axis": "Z", "color": "#F59E0B"},
    "CADC0884": {"type": "Fotodiyot", "axis": "1", "color": "#EF4444"},
    "CADC0888": {"type": "Fotodiyot", "axis": "2", "color": "#8B5CF6"},
    "CADC0892": {"type": "Fotodiyot", "axis": "3", "color": "#06B6D4"},
    "CADC0894": {"type": "Fotodiyot", "axis": "4", "color": "#F778A1"},
}

def _build_ks_panel(synth_feats):
    """Üretilen sentetik özellikleri gerçek dataset.csv ile karşılaştırır (özellik
    başına Kolmogorov-Smirnov mesafesi). Gerçek veri yoksa None döner."""
    real_path = os.path.join(ROOT, "data", "raw", "dataset.csv")
    if not os.path.exists(real_path):
        return None
    try:
        real = pd.read_csv(real_path)
        from scipy.stats import ks_2samp
    except Exception:
        return None
    meta = {"segment", "channel", "anomaly", "train", "sampling"}
    rows = []
    for f in synth_feats.columns:
        if f in meta or f not in real.columns or not pd.api.types.is_numeric_dtype(synth_feats[f]):
            continue
        s = synth_feats[f].dropna().values
        r = real[f].dropna().values
        if len(s) < 5 or len(r) < 5:
            continue
        rows.append((f, float(ks_2samp(s, r).statistic)))
    if not rows:
        return None
    rows.sort(key=lambda x: x[1])
    feats = [f for f, _ in rows]
    ks = [d for _, d in rows]
    mean_ks = sum(ks) / len(ks)
    clr = ["#EF4444" if d >= 0.45 else "#F59E0B" if d >= 0.30 else "#10B981" for d in ks]
    fig = go.Figure(go.Bar(y=feats, x=ks, orientation="h", marker_color=clr,
                           text=[f"{d:.2f}" for d in ks], textposition="outside",
                           textfont=dict(size=9, color="#475569")))
    fig.update_layout(**PLT_LAYOUT, height=460,
                      title=f"Dogrulama: Sentetik vs Gercek KS (ortalama {mean_ks:.2f})",
                      xaxis_title="KS mesafesi (0 = birebir, 1 = tamamen farkli)")
    return html.Div(className="panel mb-4", children=[
        html.Div(className="panel-title", children=[
            _icon("mdi:check-decagram-outline", 16),
            f" Dogrulama: Gercek Veriyle Dağılım Karşılaştırmasi (ort. KS = {mean_ks:.2f})"]),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div([f"En iyi uyum: {feats[0]} (KS={ks[0]:.2f})  |  En kotu: {feats[-1]} (KS={ks[-1]:.2f}). ",
                  "Düşük KS, sentetik dağılımin gercege yakinligini gosterir; yüksek değerli "
                  "özellikler ureteci iyilestirmek icin onceliklidir."],
                 style={"fontSize": "12px", "color": "#64748B", "marginTop": "8px", "lineHeight": "1.5"}),
    ])


def get_synthetic_layout():
    channel_options = []
    for ch, info in CHANNEL_INFO.items():
        channel_options.append({
            "label": html.Span([
                html.Span(ch, style={"fontFamily": "Inter, sans-serif", "fontSize": "11px"}),
                html.Span(f"  {info['type']} ({info['axis']})",
                          style={"color": "#64748B", "fontSize": "11px", "marginLeft": "8px"})
            ]),
            "value": ch,
        })

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Sentetik Veri Laboratuvari", className="page-title"),
            html.Div("ESA OPS-SAT benzeri sentetik uydu telemetrisi üretimi ve görselleştirilmesi",
                     className="page-subtitle"),
        ]),

        dbc.Row([
            dbc.Col([html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[
                    _icon("mdi:tune-vertical", 16), " Üretim Parametreleri"]),

                html.Div("KANAL SECIMI", className="section-label"),
                dcc.Checklist(
                    id="synth-channels",
                    options=channel_options,
                    value=list(CHANNEL_INFO.keys()),
                    className="model-checklist",
                    inputStyle={"marginRight": "8px"},
                ),

                html.Div("SEGMENT SAYISI", className="section-label", style={"marginTop": "16px"}),
                dcc.Slider(id="synth-n-segments", min=50, max=2000, step=50, value=500,
                           marks={50: "50", 250: "250", 500: "500", 1000: "1K", 2000: "2K"},
                           tooltip={"placement": "bottom", "always_visible": False}),

                html.Div("ANOMALI ORANI", className="section-label", style={"marginTop": "16px"}),
                dcc.Slider(id="synth-anomaly-ratio", min=0.05, max=0.50, step=0.01, value=0.20,
                           marks={0.05: "%5", 0.10: "%10", 0.20: "%20", 0.35: "%35", 0.50: "%50"},
                           tooltip={"placement": "bottom", "always_visible": False}),

                html.Div("RASTGELELIK TOHUMU", className="section-label", style={"marginTop": "16px"}),
                dcc.Input(id="synth-seed", type="number", value=42,
                          style={"backgroundColor": "#FFFFFF", "border": "1px solid #E2E8F0",
                                 "color": "#1E293B", "borderRadius": "0", "padding": "8px",
                                 "width": "100%", "fontFamily": "Inter, sans-serif"}),

                html.Div(style={"marginTop": "20px"}, children=[
                    html.Button([_icon("mdi:play-circle", 18), " Veri Uret"],
                                id="btn-synth-generate", n_clicks=0, className="btn-primary",
                                style={"width": "100%"}),
                ]),

                html.Div(id="synth-status", style={"marginTop": "12px", "fontSize": "12px",
                                                     "color": "#64748B", "textAlign": "center"}),
            ])], md=3),

            dbc.Col([
                dcc.Loading(
                    id="loading-synth",
                    type="circle",
                    color="#3B82F6",
                    children=[html.Div(id="synth-output", children=[
                        html.Div(className="info-box", children=[
                            _icon("mdi:flask-outline", 32, "#3B82F6"), html.Br(), html.Br(),
                            "Sol panelden parametreleri ayarlayip 'Veri Uret' butonuna tiklayin.",
                            html.Br(),
                            "7 kanal profili, 6 anomali turu ve 5 onboard artefakt desteklenir."
                        ])
                    ])]
                )
            ], md=9),
        ], className="g-3"),

        dcc.Store(id="synth-segments-store"),
        dcc.Store(id="synth-features-store"),
        dcc.Download(id="synth-download-segments"),
        dcc.Download(id="synth-download-features"),
    ])


def register_synthetic_callbacks(app):

    @app.callback(
        Output("synth-output", "children"),
        Output("synth-segments-store", "data"),
        Output("synth-features-store", "data"),
        Output("synth-status", "children"),
        Input("btn-synth-generate", "n_clicks"),
        State("synth-channels", "value"),
        State("synth-n-segments", "value"),
        State("synth-anomaly-ratio", "value"),
        State("synth-seed", "value"),
        prevent_initial_call=True,
    )
    def generate_synthetic(n, channels, n_segments, anomaly_ratio, seed):
        if not n or not channels:
            return no_update, no_update, no_update, "En az bir kanal seçin."

        from synthetic_generator import SyntheticTelemetryGenerator
        from feature_engineer import extract_esa_features

        gen = SyntheticTelemetryGenerator(seed=seed or 42)
        segments_df = gen.generate(
            n_segments=n_segments or 500,
            anomaly_ratio=anomaly_ratio or 0.20,
            channels=channels,
        )

        features_df = extract_esa_features(segments_df)

        n_anom = int(features_df["anomaly"].sum()) if "anomaly" in features_df.columns else 0
        n_normal = len(features_df) - n_anom
        n_rows = len(segments_df)
        n_channels_used = segments_df["channel"].nunique()

        fig_signals = make_subplots(
            rows=min(len(channels), 4), cols=1,
            shared_xaxes=False,
            subplot_titles=[f"{ch} ({CHANNEL_INFO.get(ch, {}).get('type', '')})" for ch in channels[:4]],
            vertical_spacing=0.08,
        )
        for i, ch in enumerate(channels[:4]):
            ch_data = segments_df[segments_df["channel"] == ch]
            first_seg = ch_data["segment"].iloc[0] if len(ch_data) > 0 else None
            if first_seg is not None:
                seg_data = ch_data[ch_data["segment"] == first_seg]
                color = CHANNEL_INFO.get(ch, {}).get("color", "#3B82F6")
                fig_signals.add_trace(
                    go.Scatter(y=seg_data["value"].values, mode="lines",
                               line=dict(color=color, width=1.5), name=ch, showlegend=False),
                    row=i + 1, col=1,
                )
        fig_signals.update_layout(
            **PLT_LAYOUT,
            height=120 * min(len(channels), 4) + 60,
            title="Örnek Segment Sinyalleri (ilk segment / kanal)",
        )

        anomaly_segs = segments_df[segments_df["anomaly"] == 1]
        ch_dist = features_df["channel"].value_counts()
        fig_dist = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Kanal Dağılımi", "Normal vs Anomali"],
                                 specs=[[{"type": "pie"}, {"type": "pie"}]])
        fig_dist.add_trace(
            go.Pie(labels=ch_dist.index.tolist(), values=ch_dist.values.tolist(),
                   marker=dict(colors=[CHANNEL_INFO.get(c, {}).get("color", "#64748B") for c in ch_dist.index]),
                   textinfo="label+percent", textfont=dict(size=10), hole=0.4),
            row=1, col=1,
        )
        fig_dist.add_trace(
            go.Pie(labels=["Normal", "Anomali"], values=[n_normal, n_anom],
                   marker=dict(colors=["#10B981", "#EF4444"]),
                   textinfo="label+value+percent", hole=0.4),
            row=1, col=2,
        )
        fig_dist.update_layout(**PLT_LAYOUT, height=300, showlegend=False)

        key_feats = ["mean", "var", "std", "n_peaks", "diff_var", "kurtosis"]
        available_feats = [f for f in key_feats if f in features_df.columns]
        fig_box = go.Figure()
        for feat in available_feats:
            normal_vals = features_df[features_df["anomaly"] == 0][feat].values
            anom_vals = features_df[features_df["anomaly"] == 1][feat].values
            fig_box.add_trace(go.Box(y=normal_vals, name=f"{feat} (N)", marker_color="#10B981",
                                     boxmean=True, showlegend=False))
            fig_box.add_trace(go.Box(y=anom_vals, name=f"{feat} (A)", marker_color="#EF4444",
                                     boxmean=True, showlegend=False))
        fig_box.update_layout(**PLT_LAYOUT, height=350, title="Özellik Dağılımi: Normal (N) vs Anomali (A)")

        from dash import dash_table
        preview_cols = ["segment", "channel", "anomaly", "sampling", "duration", "len",
                        "mean", "var", "std", "n_peaks", "kurtosis", "skew"]
        show_cols = [c for c in preview_cols if c in features_df.columns]
        table_data = features_df[show_cols].head(30).copy()
        for c in table_data.select_dtypes(include=[np.floating]).columns:
            table_data[c] = table_data[c].round(6)

        ks_block = _build_ks_panel(features_df) or html.Div()

        output = html.Div([
            stat_strip([
                ("Segment", len(features_df), f"{n_rows:,} satır ham", "blue"),
                ("Anomali", n_anom, f"%{anomaly_ratio * 100:.0f} oran", "red"),
                ("Kanal", n_channels_used, f"{', '.join(channels[:3])}...", "cyan"),
                ("Özellik", 18, "ESA handcrafted", "green"),
            ]),

            html.Div(className="panel mb-4", children=[
                dcc.Graph(figure=fig_signals, config={"displayModeBar": False})
            ]),

            dbc.Row([
                dbc.Col(html.Div(className="panel", children=[
                    dcc.Graph(figure=fig_dist, config={"displayModeBar": False})
                ]), md=5),
                dbc.Col(html.Div(className="panel", children=[
                    dcc.Graph(figure=fig_box, config={"displayModeBar": False})
                ]), md=7),
            ], className="mb-4 g-3"),

            ks_block,

            html.Div(className="panel mb-4", children=[
                html.Div(className="panel-title", children=[
                    _icon("mdi:table-large", 16),
                    f" Cikarilan Özellikler ({len(features_df)} segment, {len(features_df.columns)} sutun)"
                ]),
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in show_cols],
                    data=table_data.to_dict("records"),
                    page_size=10, sort_action="native", filter_action="native",
                    style_header={"backgroundColor": "#EEF2F8", "color": "#64748B",
                                   "fontWeight": "600", "border": "1px solid #E2E8F0", "fontSize": "11px"},
                    style_cell={"backgroundColor": "#FFFFFF", "color": "#1E293B",
                                 "border": "1px solid #E2E8F0", "fontFamily": "IBM Plex Sans",
                                 "fontSize": "12px", "padding": "8px"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#F4F6FB"},
                        {"if": {"filter_query": '{anomaly} = 1'}, "backgroundColor": "rgba(239,68,68,0.06)"},
                    ],
                ),
            ]),

            html.Div(className="panel", style={"display": "flex", "gap": "12px",
                                                 "justifyContent": "flex-end", "padding": "16px"}, children=[
                html.Button([_icon("mdi:download", 16), " Segments CSV"],
                            id="btn-synth-dl-seg", n_clicks=0, className="btn-outline"),
                html.Button([_icon("mdi:download", 16), " Features CSV"],
                            id="btn-synth-dl-feat", n_clicks=0, className="btn-outline"),
                html.Button([_icon("mdi:chart-timeline-variant", 16), " Analize Gonder"],
                            id="btn-synth-to-analysis", n_clicks=0, className="btn-primary"),
            ]),
        ])

        seg_json = segments_df.to_json(date_format="iso", orient="split")
        feat_json = features_df.to_json(date_format="iso", orient="split")

        return output, seg_json, feat_json, f"{len(features_df)} segment uretildi."

    @app.callback(
        Output("synth-download-segments", "data"),
        Input("btn-synth-dl-seg", "n_clicks"),
        State("synth-segments-store", "data"),
        prevent_initial_call=True,
    )
    def download_segments(n, data):
        if not n or not data:
            return no_update
        df = pd.read_json(io.StringIO(data), orient="split")
        return dcc.send_data_frame(df.to_csv, "sentetik_segments.csv", index=False)

    @app.callback(
        Output("synth-download-features", "data"),
        Input("btn-synth-dl-feat", "n_clicks"),
        State("synth-features-store", "data"),
        prevent_initial_call=True,
    )
    def download_features(n, data):
        if not n or not data:
            return no_update
        df = pd.read_json(io.StringIO(data), orient="split")
        return dcc.send_data_frame(df.to_csv, "sentetik_features.csv", index=False)

    @app.callback(
        Output("uploaded-data", "data", allow_duplicate=True),
        Output("current-page", "data", allow_duplicate=True),
        Input("btn-synth-to-analysis", "n_clicks"),
        State("synth-features-store", "data"),
        prevent_initial_call=True,
    )
    def send_to_analysis(n, data):
        if not n or not data:
            return no_update, no_update
        return data, "analysis"
