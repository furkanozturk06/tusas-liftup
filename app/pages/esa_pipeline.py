
import os, sys, io, base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, callback, Input, Output, State, no_update, ctx, dash_table
import dash_bootstrap_components as dbc

from utils.ui import PLT_LAYOUT, icon as _icon, metric_card as _metric_card, stat_strip

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

SEGMENTS_PATH = os.path.join(ROOT, "data", "raw", "segments.csv")
DATASET_PATH = os.path.join(ROOT, "data", "raw", "dataset.csv")
SYNTH_SEG_PATH = os.path.join(ROOT, "data", "synthetic", "synthetic_segments.csv")

ESA_FEATURES = [
    ("sampling", "Örnekleme Frekansı", "Veri toplama hızı (Hz)"),
    ("duration", "Segment Süresi", "Başlangıç-bitis arasi saniye"),
    ("len", "Segment Uzunluğu", "Veri noktası sayısı"),
    ("mean", "Ortalama", "Sinyal ortalamasi"),
    ("var", "Varyans", "Populasyon varyansi"),
    ("std", "Standart Sapma", "Populasyon standart sapmasi"),
    ("kurtosis", "Basıklık", "Excess kurtosis (dağılım sivriliği)"),
    ("skew", "Çarpıklık", "Dağılım asimetrisi"),
    ("n_peaks", "Tepe Sayısı", "Ham sinyaldeki tepe sayısı (%10 prominence)"),
    ("smooth10_n_peaks", "Yumusak Tepe (w=10)", "10-pkt yumusatilmis sinyaldeki tepe sayısı"),
    ("smooth20_n_peaks", "Yumusak Tepe (w=20)", "20-pkt yumusatilmis sinyaldeki tepe sayısı"),
    ("diff_peaks", "1. Türev Tepe", "Birinci türev sinyalindeki tepe sayısı"),
    ("diff2_peaks", "2. Türev Tepe", "Ikinci türev sinyalindeki tepe sayısı"),
    ("diff_var", "1. Türev Varyansi", "Birinci türevin varyansi"),
    ("diff2_var", "2. Türev Varyansi", "Ikinci türevin varyansi"),
    ("gaps_squared", "Boşluk Karesi", "Zaman farklarinin kareler toplamı"),
    ("len_weighted", "Agirlikli Uzunluk", "len * sampling"),
    ("var_div_duration", "Varyans/Sure", "Birim zamandaki varyans"),
    ("var_div_len", "Varyans/Uzunluk", "Veri noktası basina varyans"),
]

def get_esa_pipeline_layout():
    has_segments = os.path.exists(SEGMENTS_PATH)
    has_dataset = os.path.exists(DATASET_PATH)
    has_synth = os.path.exists(SYNTH_SEG_PATH)

    source_options = []
    if has_segments:
        seg_size = os.path.getsize(SEGMENTS_PATH) / (1024 * 1024)
        source_options.append({"label": f"segments.csv ({seg_size:.1f} MB)", "value": "segments"})
    if has_synth:
        syn_size = os.path.getsize(SYNTH_SEG_PATH) / (1024 * 1024)
        source_options.append({"label": f"synthetic_segments.csv ({syn_size:.1f} MB)", "value": "synthetic"})
    source_options.append({"label": "Dosya Yükle (CSV)", "value": "upload"})

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("ESA Feature Extraction Pipeline", className="page-title"),
            html.Div("Ham telemetri segmentlerinden 18 handcrafted özellik çıkarımı",
                     className="page-subtitle"),
        ]),

        dbc.Row([
            dbc.Col([html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[
                    _icon("mdi:database-import-outline", 16), " Veri Kaynagi"]),

                html.Div("KAYNAK SECIMI", className="section-label"),
                dcc.RadioItems(
                    id="esa-source",
                    options=source_options,
                    value=source_options[0]["value"] if source_options else "upload",
                    className="model-checklist",
                    inputStyle={"marginRight": "8px"},
                ),

                html.Div(id="esa-upload-area", style={"display": "none", "marginTop": "16px"}, children=[
                    dcc.Upload(id="esa-file-upload", children=html.Div(className="upload-area",
                               style={"padding": "20px"}, children=[
                        _icon("mdi:cloud-upload-outline", 36, "#3B82F6"),
                        html.Div("CSV dosyası yükleyin", style={"color": "#475569", "fontSize": "12px"}),
                    ]), multiple=False),
                ]),

                html.Div(style={"marginTop": "20px"}, children=[
                    html.Button([_icon("mdi:cog-play-outline", 18), " Feature Extraction Başlat"],
                                id="btn-esa-extract", n_clicks=0, className="btn-primary",
                                style={"width": "100%"}),
                ]),

                html.Div(id="esa-status", style={"marginTop": "12px", "fontSize": "12px",
                                                    "color": "#64748B", "textAlign": "center"}),

                html.Div(style={"marginTop": "24px"}, children=[
                    html.Div("18 OZELLIK KATALOGU", className="section-label"),
                    html.Div(style={"maxHeight": "400px", "overflowY": "auto"}, children=[
                        html.Div([
                            html.Div(style={"display": "flex", "justifyContent": "space-between",
                                             "padding": "6px 0", "borderBottom": "1px solid #E2E8F0"}, children=[
                                html.Div([
                                    html.Span(f_name, style={"fontFamily": "Inter, sans-serif", "fontSize": "11px",
                                                               "color": "#06B6D4"}),
                                    html.Br(),
                                    html.Span(f_desc, style={"fontSize": "10px", "color": "#64748B"}),
                                ]),
                            ])
                            for f_name, f_label, f_desc in ESA_FEATURES
                        ]),
                    ]),
                ]),
            ])], md=3),

            dbc.Col([
                dcc.Loading(
                    id="loading-esa",
                    type="circle",
                    color="#3B82F6",
                    children=[html.Div(id="esa-output", children=[
                        _build_existing_dataset_preview() if has_dataset else
                        html.Div(className="info-box", children=[
                            _icon("mdi:rocket-launch-outline", 32, "#3B82F6"), html.Br(), html.Br(),
                            "Veri kaynağı secip 'Feature Extraction Başlat' butonuna tiklayin.",
                            html.Br(),
                            "Pipeline, her segment icin 18 handcrafted özellik cikaracaktir."
                        ])
                    ])]
                )
            ], md=9),
        ], className="g-3"),

        dcc.Store(id="esa-features-store"),
        dcc.Download(id="esa-download"),
    ])


def _build_existing_dataset_preview():
    if not os.path.exists(DATASET_PATH):
        return html.Div()
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception:
        return html.Div()

    n_anom = int(df["anomaly"].sum()) if "anomaly" in df.columns else 0
    n_feat = len([c for c in df.columns if c not in ["segment", "anomaly", "train", "channel"]])

    return html.Div([
        html.Div(className="panel mb-3",
                 style={"borderLeft": "4px solid #10B981", "padding": "12px"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px"}, children=[
                _icon("mdi:check-circle", 20, "#10B981"),
                html.Span("Mevcut dataset.csv bulundu",
                           style={"color": "#10B981", "fontWeight": "600", "fontSize": "13px"}),
                html.Span(f"({len(df)} segment, {n_feat} özellik, {n_anom} anomali)",
                           style={"color": "#64748B", "fontSize": "12px", "marginLeft": "8px"}),
            ]),
        ]),
        html.Div("Yeniden çıkarım icin 'Feature Extraction Başlat' butonunu kullanabilirsiniz.",
                 style={"color": "#64748B", "fontSize": "12px", "marginTop": "8px"}),
    ])


def _build_extraction_results(features_df, source_label):
    n = len(features_df)
    n_anom = int(features_df["anomaly"].sum()) if "anomaly" in features_df.columns else 0
    n_channels = features_df["channel"].nunique() if "channel" in features_df.columns else 0
    feat_cols = [c for c in features_df.columns if c not in ["segment", "anomaly", "train", "channel"]]

    corr_cols = [c for c in ["mean", "var", "std", "kurtosis", "skew", "n_peaks",
                              "diff_var", "diff2_var", "gaps_squared"] if c in features_df.columns]
    if corr_cols:
        corr = features_df[corr_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=9),
        ))
        fig_corr.update_layout(**PLT_LAYOUT, height=400, title="Özellik Korelasyon Matrisi")
    else:
        fig_corr = go.Figure()
        fig_corr.update_layout(**PLT_LAYOUT, height=200)

    var_ratios = {}
    if "anomaly" in features_df.columns and n_anom > 0:
        normal = features_df[features_df["anomaly"] == 0]
        anomaly = features_df[features_df["anomaly"] == 1]
        for c in feat_cols:
            if c in features_df.columns and features_df[c].dtype in [np.float64, np.int64, np.float32]:
                n_var = normal[c].var()
                a_var = anomaly[c].var()
                if n_var > 0:
                    var_ratios[c] = a_var / n_var
    if var_ratios:
        sorted_ratios = dict(sorted(var_ratios.items(), key=lambda x: x[1], reverse=True)[:15])
        fig_ratio = go.Figure(go.Bar(
            x=list(sorted_ratios.values()), y=list(sorted_ratios.keys()), orientation="h",
            marker_color=["#EF4444" if v > 2 else "#F59E0B" if v > 1 else "#10B981"
                          for v in sorted_ratios.values()],
            text=[f"{v:.2f}x" for v in sorted_ratios.values()],
            textposition="outside", textfont=dict(size=10, color="#475569"),
        ))
        fig_ratio.update_layout(**PLT_LAYOUT, height=400,
                                 title="Varyans Oranı (Anomali / Normal)",
                                 xaxis_title="Varyans Oranı")
        fig_ratio.update_yaxes(autorange="reversed")
    else:
        fig_ratio = go.Figure()
        fig_ratio.update_layout(**PLT_LAYOUT, height=200)

    show_cols = [c for c in ["segment", "channel", "anomaly", "sampling", "duration", "len",
                              "mean", "var", "std", "n_peaks", "kurtosis", "skew",
                              "diff_var", "gaps_squared"] if c in features_df.columns]
    table_data = features_df[show_cols].head(30).copy()
    for c in table_data.select_dtypes(include=[np.floating]).columns:
        table_data[c] = table_data[c].round(6)

    return html.Div([
        stat_strip([
            ("Segment", n, f"Kaynak: {source_label}", "blue"),
            ("Anomali", n_anom, f"%{n_anom / max(n, 1) * 100:.1f}", "red"),
            ("Kanal", n_channels, None, "cyan"),
            ("Özellik", len(feat_cols), "18 handcrafted + meta", "green"),
        ]),

        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[
                dcc.Graph(figure=fig_corr, config={"displayModeBar": False})
            ]), md=6),
            dbc.Col(html.Div(className="panel", children=[
                dcc.Graph(figure=fig_ratio, config={"displayModeBar": False})
            ]), md=6),
        ], className="mb-4 g-3"),

        html.Div(className="panel mb-4", children=[
            html.Div(className="panel-title", children=[
                _icon("mdi:table-large", 16),
                f" Cikarilan Özellikler ({n} segment)"
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
            html.Button([_icon("mdi:download", 16), " Dataset CSV Indir"],
                        id="btn-esa-download", n_clicks=0, className="btn-outline"),
            html.Button([_icon("mdi:chart-timeline-variant", 16), " Analize Gonder"],
                        id="btn-esa-to-analysis", n_clicks=0, className="btn-primary"),
        ]),
    ])


def register_esa_pipeline_callbacks(app):

    @app.callback(
        Output("esa-upload-area", "style"),
        Input("esa-source", "value"),
    )
    def toggle_upload(source):
        if source == "upload":
            return {"display": "block", "marginTop": "16px"}
        return {"display": "none", "marginTop": "16px"}

    @app.callback(
        Output("esa-output", "children"),
        Output("esa-features-store", "data"),
        Output("esa-status", "children"),
        Input("btn-esa-extract", "n_clicks"),
        State("esa-source", "value"),
        State("esa-file-upload", "contents"),
        State("esa-file-upload", "filename"),
        prevent_initial_call=True,
    )
    def run_extraction(n, source, upload_contents, upload_filename):
        if not n:
            return no_update, no_update, no_update

        from feature_engineer import extract_esa_features

        source_label = ""
        try:
            if source == "segments":
                segments_df = pd.read_csv(SEGMENTS_PATH)
                source_label = "segments.csv"
            elif source == "synthetic":
                segments_df = pd.read_csv(SYNTH_SEG_PATH)
                source_label = "synthetic_segments.csv"
            elif source == "upload" and upload_contents:
                _, content_string = upload_contents.split(",")
                decoded = base64.b64decode(content_string)
                segments_df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                source_label = upload_filename or "yuklenen_dosya.csv"
            else:
                return no_update, no_update, "Lütfen bir veri kaynağı seçin veya dosya yükleyin."

            features_df = extract_esa_features(segments_df)

            feat_json = features_df.to_json(date_format="iso", orient="split")
            results = _build_extraction_results(features_df, source_label)

            return results, feat_json, f"{len(features_df)} segment icin 18 özellik cikarildi."

        except Exception as e:
            import traceback
            err = str(e) + "\n" + traceback.format_exc()
            return (
                html.Div(f"Hata: {err}",
                         style={"color": "#EF4444", "whiteSpace": "pre-wrap", "fontSize": "11px"}),
                no_update,
                "Hata olustu.",
            )

    @app.callback(
        Output("esa-download", "data"),
        Input("btn-esa-download", "n_clicks"),
        State("esa-features-store", "data"),
        prevent_initial_call=True,
    )
    def download_dataset(n, data):
        if not n or not data:
            return no_update
        df = pd.read_json(io.StringIO(data), orient="split")
        return dcc.send_data_frame(df.to_csv, "esa_features.csv", index=False)

    @app.callback(
        Output("uploaded-data", "data", allow_duplicate=True),
        Output("current-page", "data", allow_duplicate=True),
        Input("btn-esa-to-analysis", "n_clicks"),
        State("esa-features-store", "data"),
        prevent_initial_call=True,
    )
    def send_to_analysis(n, data):
        if not n or not data:
            return no_update, no_update
        return data, "analysis"
