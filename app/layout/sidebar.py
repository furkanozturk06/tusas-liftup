from dash import html

from utils.ui import icon


def nav_item(ic, text, page_id):
    return html.Button(id={"type": "nav", "page": page_id}, n_clicks=0,
                       className="nav-item", children=[icon(ic, 18), html.Span(text)])


def nav_subgroup(text):
    return html.Div(text, style={"fontSize": "9px", "letterSpacing": "1.5px", "color": "#94A3B8",
                                 "fontWeight": "600", "padding": "10px 16px 2px"})


hidden_refs = html.Div(style={"display": "none"}, children=[
    html.Span(id="global-live-dot"),
])


def build_sidebar():
    return html.Div(className="sidebar", children=[
        html.Div(className="sidebar-logo", children=[
            html.Div([icon("mdi:satellite-variant", 26, "#06B6D4")], className="logo-icon"),
            html.Div([html.Div("Uydu Telemetri", className="logo-text"),
                      html.Div("Anomali Tespit Sistemi", className="logo-sub")])
        ]),
        html.Div(className="sidebar-nav", children=[
            html.Div("OPERASYON", className="nav-section-label",
                     style={"fontSize": "10px", "letterSpacing": "2px", "color": "#94A3B8",
                            "fontWeight": "600", "padding": "8px 16px 4px"}),
            nav_item("mdi:view-dashboard", "Operasyon Paneli", "dashboard"),
            nav_item("mdi:satellite-variant", "Canlı İzleme", "live"),
            nav_item("mdi:upload", "Veri Yükle", "upload"),
            nav_item("mdi:chart-timeline-variant", "Analiz", "analysis"),
            nav_item("mdi:chart-scatter-plot", "Sonuçlar", "results"),
            nav_item("mdi:magnify-expand", "Anomali Detay", "detail"),
            html.Details(open=False, className="nav-group", children=[
                html.Summary("GELİŞTİRİCİ / ARAŞTIRMA", className="nav-group-header",
                             style={"fontSize": "10px", "letterSpacing": "2px", "color": "#94A3B8",
                                    "fontWeight": "600", "padding": "12px 16px 4px", "cursor": "pointer",
                                    "userSelect": "none", "outline": "none"}),
                nav_subgroup("MODELLER"),
                nav_item("mdi:gauge", "Model Performans", "performance"),
                nav_item("mdi:brain", "SHAP Analiz", "shap"),
                nav_item("mdi:test-tube", "Ablasyon Analizi", "ablation"),
                nav_subgroup("ARAŞTIRMA BULGULARI"),
                nav_item("mdi:scale-balance", "Benchmark", "benchmark"),
                nav_item("mdi:chart-box-outline", "Augmentasyon", "augmentation"),
                nav_subgroup("VERİ & PIPELINE"),
                nav_item("mdi:flask-outline", "Sentetik Lab", "synthetic"),
                nav_item("mdi:rocket-launch-outline", "ESA Pipeline", "esa_pipeline"),
                nav_item("mdi:lightning-bolt", "Güç Tüketimi", "power"),
            ]),
        ]),
    ])
