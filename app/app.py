import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys

from dash import Dash, html, dcc, callback, Input, Output, ctx
import dash_bootstrap_components as dbc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from utils.ui import icon
from core.state import ALL_METRICS
from layout.sidebar import build_sidebar, hidden_refs

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Uydu Telemetri", update_title=None)

from pages.ablation import get_ablation_layout, register_ablation_callbacks
from pages.power import get_power_layout, register_power_callbacks
from pages.synthetic import get_synthetic_layout, register_synthetic_callbacks
from pages.esa_pipeline import get_esa_pipeline_layout, register_esa_pipeline_callbacks
from pages.benchmark import get_benchmark_layout
from pages.augmentation import get_augmentation_layout
from pages import dashboard, upload, analysis, results, performance, shap, live, detail

app.layout = html.Div(id="app-root", children=[
    dcc.Store(id="current-page", data="dashboard"),
    dcc.Store(id="uploaded-data"),
    dcc.Store(id="prediction-results"),
    dcc.Store(id="selected-anomaly"),
    dcc.Store(id="anomaly-list"),
    dcc.Store(id="live-sim-state", data={"index": 0, "is_running": False, "anomalies": []}),
    dcc.Interval(id="live-interval", interval=500, n_intervals=0, disabled=True),
    dcc.Download(id="download-csv"),
    hidden_refs,
    build_sidebar(),
    html.Div(id="page-content", className="main-content"),
    html.Div(id="results-overlay", className="main-content",
             style={"display": "none"}, children=[
        html.Div(className="page-header", children=[
            html.Div("Sonuçlar", className="page-title"),
            html.Div("Anomali tespit sonuçları ve görselleştirme", className="page-subtitle")]),
        dcc.Loading(
            id="loading-results",
            type="circle",
            color="#3B82F6",
            children=[
                html.Div(id="results-content", children=[
                    html.Div(className="info-box", children=[
                        icon("mdi:chart-scatter-plot", 32, "#3B82F6"), html.Br(), html.Br(),
                        "Henüz analiz yapılmadı. Önce Analiz sayfasından işlem başlatınız."])
                ])
            ]
        )
    ]),
    html.Div(id="detail-overlay", className="main-content",
             style={"display": "none"}, children=[
        html.Div(id="detail-page-content", className="detail-page-container", children=[
            html.Div("Henüz anomali seçilmedi.", className="info-box")
        ])
    ]),
    html.Div(id="live-overlay", className="main-content",
             style={"display": "none"}, children=live.page_live()),
])

PAGES = {"dashboard": dashboard.page_dashboard, "upload": upload.page_upload, "analysis": analysis.page_analysis,
         "results": results.page_results, "shap": shap.page_shap, "performance": performance.page_performance, "live": live.page_live, "detail": detail.page_detail, "ablation": get_ablation_layout,
         "power": lambda: get_power_layout(ALL_METRICS),
         "synthetic": get_synthetic_layout, "esa_pipeline": get_esa_pipeline_layout,
         "benchmark": get_benchmark_layout, "augmentation": get_augmentation_layout}

@callback(Output("current-page", "data"),
          [Input({"type": "nav", "page": p}, "n_clicks") for p in PAGES],
          prevent_initial_call=True)
def navigate(*clicks):
    if not ctx.triggered_id: return "dashboard"
    return ctx.triggered_id["page"]

@callback(Output("page-content", "children"), Output("page-content", "style"),
          Output("results-overlay", "style"),
          Output("detail-overlay", "style"),
          Output("live-overlay", "style"),
          Input("current-page", "data"))
def render_page(page_id):
    hide, show = {"display": "none"}, {"display": "block"}
    if page_id == "results":
        return html.Div(), hide, show, hide, hide
    if page_id == "detail":
        return html.Div(), hide, hide, show, hide
    if page_id == "live":
        return html.Div(), hide, hide, hide, show
    return PAGES.get(page_id, dashboard.page_dashboard)(), show, hide, hide, hide

register_ablation_callbacks(app)
register_power_callbacks(app, ALL_METRICS)
register_synthetic_callbacks(app)
register_esa_pipeline_callbacks(app)

if __name__ == "__main__":
    app.run(
        debug=os.environ.get("DASH_DEBUG", "0") == "1",
        host=os.environ.get("DASH_HOST", "0.0.0.0"),
        port=int(os.environ.get("DASH_PORT", "8050")),
    )
