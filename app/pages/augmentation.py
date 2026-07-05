import os
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

from utils.ui import PLT_LAYOUT, icon as _icon, metric_card as _metric_card, stat_strip

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MDIR = os.path.join(ROOT, "reports", "metrics")
AUG_CSV = os.path.join(MDIR, "augmentation_comparison.csv")
KS_CSV = os.path.join(MDIR, "synthetic_real_ks_distance.csv")
FULL_CSV = os.path.join(MDIR, "ablation_synthetic_fulldata.csv")
LOW_CSV = os.path.join(MDIR, "ablation_synthetic_lowdata.csv")

STRAT_COLOR = {
    "Baseline (augmentasyonsuz)": "#64748B",
    "+SMOTE": "#10B981",
    "+ICCS-ω": "#3B82F6",
    "+Sentetik": "#F59E0B",
}


def _missing_layout():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Augmentasyon Bulguları", className="page-title"),
            html.Div("SMOTE / ICCS-ω / Sentetik strateji karşılaştırması", className="page-subtitle")]),
        html.Div(className="warning-box", children=[
            _icon("mdi:alert-outline", 40, "#F59E0B"),
            html.Div("Augmentasyon Verileri Bulunamadı", className="warning-title"),
            html.Div([
                "Bulgular henüz üretilmemiş. Lütfen önce ",
                html.Code("notebooks/13_sentetik_augmentasyon_ablasyonu.ipynb"), " ve ",
                html.Code("notebooks/14_augmentasyon_karsilastirma.ipynb"),
                " notebook'larını çalıştırın."
            ], className="warning-body"),
        ]),
    ])


def get_augmentation_layout():
    if not os.path.exists(AUG_CSV):
        return _missing_layout()
    aug = pd.read_csv(AUG_CSV)

    fig_auc = go.Figure()
    for strat, color in STRAT_COLOR.items():
        sub = aug[aug["Strateji"] == strat]
        if sub.empty:
            continue
        fig_auc.add_trace(go.Bar(name=strat, x=sub["Model"], y=sub["AUC_PR"],
                                 marker_color=color,
                                 text=[f"{v:.3f}" for v in sub["AUC_PR"]], textposition="outside",
                                 textfont=dict(size=9)))
    fig_auc.update_layout(**PLT_LAYOUT, height=380, barmode="group",
                          yaxis_title="AUC-PR", yaxis_range=[0.7, 1.02],
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

    fig_pr = go.Figure()
    for strat, color in STRAT_COLOR.items():
        sub = aug[aug["Strateji"] == strat]
        if sub.empty:
            continue
        fig_pr.add_trace(go.Scatter(
            x=sub["Recall"], y=sub["Precision"], mode="markers+text",
            text=sub["Model"], textposition="top center", textfont=dict(size=8, color="#475569"),
            marker=dict(size=12, color=color, opacity=0.85), name=strat))
    fig_pr.update_layout(**PLT_LAYOUT, height=380,
                         xaxis_title="Recall (duyarlılık)", yaxis_title="Precision (kesinlik)",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

    ks_section = []
    ks_mean = None
    if os.path.exists(KS_CSV):
        ks = pd.read_csv(KS_CSV).sort_values("KS_mesafe", ascending=True)
        ks_mean = ks["KS_mesafe"].mean()
        worst = ks.iloc[-1]
        ks_clr = ["#EF4444" if v >= 0.45 else "#F59E0B" if v >= 0.30 else "#10B981" for v in ks["KS_mesafe"]]
        fig_ks = go.Figure(go.Bar(
            y=ks["Özellik"], x=ks["KS_mesafe"], orientation="h", marker_color=ks_clr,
            text=[f"{v:.2f}" for v in ks["KS_mesafe"]], textposition="outside",
            textfont=dict(size=9, color="#475569")))
        fig_ks.update_layout(**PLT_LAYOUT, height=520,
                             title="Sentetik-Gerçek Dağılım Açığı (KS mesafesi, özellik başına)",
                             xaxis_title="KS mesafesi (0 = birebir, 1 = tamamen farklı)")
        ks_section = [
            html.Div(className="panel mb-4", children=[
                dcc.Graph(figure=fig_ks, config={"displayModeBar": False})]),
        ]

    ablation_section = []
    figs = []
    if os.path.exists(FULL_CSV):
        full = pd.read_csv(FULL_CSV)
        fig_full = go.Figure()
        for model in full["Model"].unique():
            sub = full[full["Model"] == model].sort_values("Sentetik")
            fig_full.add_trace(go.Scatter(x=sub["Sentetik"], y=sub["AUC_PR"], mode="lines+markers", name=model))
        fig_full.update_layout(**PLT_LAYOUT, height=340, title="Tam-Veri: Sentetik Ekleme vs AUC-PR",
                               xaxis_title="Eklenen sentetik segment", yaxis_title="AUC-PR")
        figs.append(fig_full)
    if os.path.exists(LOW_CSV):
        low = pd.read_csv(LOW_CSV)
        xcol = "Gerçek %" if "Gerçek %" in low.columns else low.columns[0]
        fig_low = go.Figure(go.Bar(
            x=low[xcol].astype(str), y=low["ΔAUC_PR"],
            marker_color=["#10B981" if v >= 0 else "#EF4444" for v in low["ΔAUC_PR"]],
            text=[f"{v:+.3f}" for v in low["ΔAUC_PR"]], textposition="outside"))
        fig_low.update_layout(**PLT_LAYOUT, height=340, title="Az-Veri: +1000 Sentetik ΔAUC-PR",
                              xaxis_title="Gerçek veri oranı (%)", yaxis_title="ΔAUC-PR")
        figs.append(fig_low)
    if figs:
        ablation_section = [dbc.Row([
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=f, config={"displayModeBar": False})]),
                    md=6) for f in figs
        ], className="mb-4 g-3")]

    cards = stat_strip([
        ("Augmentasyon Etkisi", "Nötr", "AUC-PR ~sabit", "yellow"),
        ("Ortalama KS Açığı", f"{ks_mean:.2f}" if ks_mean is not None else "N/A", "sentetik vs gerçek", "red"),
        ("Kesinlik (ICCS-ω)", "Artar", "yanlış alarmı azaltır", "blue"),
        ("Duyarlılık (SMOTE)", "Artar", "kaçırmayı azaltır", "green"),
    ])

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Augmentasyon Bulguları", className="page-title"),
            html.Div("SMOTE / ICCS-ω / Sentetik · gerçek Ψ üzerinde etki ve nedeni", className="page-subtitle")]),
        cards,
        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[_icon("mdi:chart-bar", 16),
                         " AUC-PR Strateji Karşılaştırması (model bazında)"]),
                dcc.Graph(figure=fig_auc, config={"displayModeBar": False})]), md=6),
            dbc.Col(html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[_icon("mdi:chart-scatter-plot", 16),
                         " Kesinlik-Duyarlılık Dengesi (asıl etki burada)"]),
                dcc.Graph(figure=fig_pr, config={"displayModeBar": False})]), md=6),
        ], className="mb-4 g-3"),
        *ks_section,
        *ablation_section,
        html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[
                _icon("mdi:table-large", 16), " Strateji Karşılaştırma Tablosu"]),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in aug.columns],
                data=aug.round(3).to_dict("records"), page_size=16, sort_action="native",
                style_header={"backgroundColor": "#EEF2F8", "color": "#64748B", "fontWeight": "600",
                              "border": "1px solid #E2E8F0", "fontSize": "11px"},
                style_cell={"backgroundColor": "#FFFFFF", "color": "#1E293B", "border": "1px solid #E2E8F0",
                            "fontFamily": "IBM Plex Sans", "fontSize": "12px", "padding": "8px"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#F4F6FB"}],
            ),
        ]),
    ])
