import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from utils.ui import PLT_LAYOUT, icon, metric_card, stat_strip

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ABLATION_PKL = os.path.join(ROOT, "models", "ablation_results.pkl")

def get_ablation_layout():
    if not os.path.exists(ABLATION_PKL):
        return html.Div([
            html.Div(className="page-header", children=[
                html.Div("ABLASYON ANALİZİ", style={"fontSize": "9px", "letterSpacing": "3px", "color": "#94A3B8"}),
                html.Div("UYDU TELEMETRİ ANOMALİ TESPİTİ > ABLASYON ANALİZİ", style={"fontSize": "11px", "color": "#94A3B8"})
            ]),
            html.Div(className="warning-box", children=[
                icon("mdi:alert-outline", 40, "#F59E0B"),
                html.Div("Ablasyon Verileri Bulunamadı", className="warning-title"),
                html.P([
                    "Ablasyon analizi henüz çalıştırılmamış.",
                    html.Br(),
                    "Ablasyon analizi için ", html.Code("notebooks/08_ablation_study.ipynb"),
                    " notebook'unu baştan sona çalıştırın."
                ], className="warning-body")
            ])
        ])
        
    try:
        data = joblib.load(ABLATION_PKL)
    except Exception as e:
        return html.Div(f"Hata: {e}")
        
    models = list(data['baseline'].keys())
    best_model = models[0] if models else "Unknown"
    
    baseline_auc = data['baseline'].get(best_model, {}).get('AUC-ROC', 0)
    
    best_set = data.get('best_set', {})
    crit_feats = best_set.get('critical', [])
    crit_feat_name = crit_feats[0] if crit_feats else "Bulunamadi"
    min_feat_count = best_set.get('optimal_count', 19)
    
    try:
        df_cum = pd.DataFrame(data['cumulative'])
        best_cum_auc = df_cum['AUC-ROC'].max()
        max_drop_pct = ((baseline_auc - best_cum_auc) / max(baseline_auc, 0.0001)) * 100
    except:
        max_drop_pct = 0.0

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between"}, children=[
                html.Div("ABLATION STUDY", style={"fontSize": "16px", "letterSpacing": "3px", "color": "#94A3B8", "fontWeight": "bold"}),
                html.Div("UYDU TELEMETRİ ANOMALİ TESPİTİ > ABLATION STUDY", style={"fontSize": "11px", "color": "#94A3B8"})
            ])
        ]),
        
        stat_strip([
            ("Baseline AUC-ROC", f"{baseline_auc:.3f}", None, "blue"),
            ("En Kritik Özellik", crit_feat_name, None, "red"),
            ("Yeterli Özellik", f"{min_feat_count} Adet", None, "green"),
            ("Performans Değişimi", f"%{abs(max_drop_pct):.2f}", None, "cyan"),
        ]),
        
        dcc.Tabs(id="ablation-tabs", value="tab-single", className="custom-tabs", children=[
            dcc.Tab(label="Tekil Özellik Etkisi", value="tab-single", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Kümülatif Performans", value="tab-cum", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Özellik Grubu", value="tab-group", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Bağımlılık Matrisi", value="tab-heat", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Önerilen Özellik Seti", value="tab-best", className="tab", selected_className="tab--selected"),
        ]),
        
        html.Div(id="ablation-tab-content", style={"marginTop": "20px"})
    ])

def register_ablation_callbacks(app):
    @app.callback(
        Output("ablation-tab-content", "children"),
        Input("ablation-tabs", "value")
    )
    def render_ablation_tab(tab):
        if not os.path.exists(ABLATION_PKL):
            return html.Div()
        try:
            data = joblib.load(ABLATION_PKL)
        except:
            return html.Div()
            
        if tab == "tab-single":
            models = list(data['baseline'].keys())
            return html.Div(className="panel", children=[
                html.Div("Model Seçimi:", style={"marginBottom": "10px"}),
                dcc.Dropdown(id="abl-model-dropdown", options=[{"label": m, "value": m} for m in models],
                             value=models[0] if models else None, className="custom-dropdown", clearable=False),
                dcc.Graph(id="abl-single-graph", config={"displayModeBar": False}),
                html.Div(id="abl-single-text", style={"marginTop": "10px", "color": "#475569"})
            ])
            
        elif tab == "tab-cum":
            df_cum = pd.DataFrame(data['cumulative'])
            fig = px.line(df_cum, x='num_features', y='AUC-ROC', color='model', markers=True)
            best_feat_count = data.get('best_set', {}).get('optimal_count', 19)
            fig.add_vline(x=best_feat_count, line_dash="dash", line_color="cyan")
            fig.update_layout(**PLT_LAYOUT, title="Özellik Sayısına Göre Model Performansı")
            return html.Div(className="panel", children=[
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                html.Div(f"{best_feat_count} özellik ile maksimum performans elde ediliyor.", style={"marginTop": "10px", "color": "#0284C7", "fontWeight": "bold"})
            ])
            
        elif tab == "tab-group":
            group_rows = []
            for g_name, g_res in data.get('groups', {}).items():
                for m_name, metrics in g_res.items():
                    group_rows.append({'Group': g_name, 'Model': m_name, 'AUC-ROC': metrics['AUC-ROC']})
            df_grp = pd.DataFrame(group_rows)
            fig = px.bar(df_grp, x='Group', y='AUC-ROC', color='Model', barmode='group')
            fig.update_layout(**PLT_LAYOUT, title="Zaman Penceresi Grubu Karşılaştırması", yaxis_range=[0.5, 1.0])
            
            best_group = "N/A"
            if len(df_grp) > 0:
                best_group = df_grp.groupby('Group')['AUC-ROC'].mean().idxmax()
                
            return html.Div(className="panel", children=[
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                html.Div(f"En iyi grup kombinasyonu: {best_group}", style={"marginTop": "10px", "color": "#0284C7", "fontWeight": "bold"})
            ])
            
        elif tab == "tab-heat":
            df_single = pd.DataFrame(data['single_removal'])
            heat_data = df_single.pivot(index='removed_feature', columns='model', values='AUC-ROC')
            fig = px.imshow(heat_data, text_auto=".3f", 
                            color_continuous_scale=[[0, "#F4F6FB"], [0.5, "#003A5C"], [1, "#0284C7"]], 
                            aspect='auto')
            fig.update_layout(**PLT_LAYOUT, title="Model - Özellik Bağımlılık Matrisi")
            return html.Div(className="panel", children=[
                dcc.Graph(figure=fig, config={"displayModeBar": False})
            ])
            
        elif tab == "tab-best":
            best_set = data.get('best_set', {})
            crit = best_set.get('critical', [])
            remov = best_set.get('removable', [])
            
            crit_items = [html.Div([icon("mdi:check-circle", 16, "#10B981"), html.Span(f" {f} (Katkısı yüksek)", style={"marginLeft": "8px"})], style={"marginBottom": "5px"}) for f in crit]
            remov_items = [html.Div([icon("mdi:alert-circle", 16, "#F59E0B"), html.Span(f" {f} (Çıkarılabilir)", style={"marginLeft": "8px"})], style={"marginBottom": "5px"}) for f in remov]
            
            labels = ['Kritik', 'Çıkarılabilir', 'Diğer']
            sizes = [len(crit), len(remov), max(19 - len(crit) - len(remov), 0)]
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=.4, 
                                             marker_colors=["#10B981", "#F59E0B", "#3B82F6"])])
            fig_pie.update_layout(**PLT_LAYOUT)
            fig_pie.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            
            return dbc.Row([
                dbc.Col([
                    html.Div(className="panel mb-3", children=[
                        html.Div("ÖNERİLEN ÖZELLİK SETİ", style={"fontWeight": "bold", "color": "#10B981", "marginBottom": "10px"}),
                        html.Div(crit_items)
                    ]),
                    html.Div(className="panel", children=[
                        html.Div("ÇIKARILABİLİR ÖZELLİKLER", style={"fontWeight": "bold", "color": "#F59E0B", "marginBottom": "10px"}),
                        html.Div(remov_items)
                    ])
                ], md=6),
                dbc.Col([
                    html.Div(className="panel mb-3", children=[
                        dcc.Graph(figure=fig_pie, config={"displayModeBar": False})
                    ]),
                    html.Div(className="panel", style={"borderLeft": "2px solid #0284C7", "backgroundColor": "rgba(0,200,255,0.04)"}, children=[
                        html.Div("SONUÇ", style={"fontWeight": "bold", "color": "#0284C7", "marginBottom": "10px"}),
                        html.Div(f"Yapılan Ablasyon analizine göre, toplam {best_set.get('optimal_count', 19)} özellik ile maksimum anomali tespit verimi alınmaktadır. Geri kalan özellikler sistemden çıkartılarak operasyonel yük azaltılabilir.")
                    ])
                ], md=6)
            ], className="g-3")
            
        return html.Div()

    @app.callback(
        Output("abl-single-graph", "figure"),
        Output("abl-single-text", "children"),
        Input("abl-model-dropdown", "value")
    )
    def update_single_graph(model_name):
        if not model_name or not os.path.exists(ABLATION_PKL):
            return go.Figure(), ""
            
        try:
            data = joblib.load(ABLATION_PKL)
            df_single = pd.DataFrame(data['single_removal'])
            df_model = df_single[df_single['model'] == model_name].copy()
            df_model = df_model.sort_values('delta_auc')
            
            fig = px.bar(df_model, x='delta_auc', y='removed_feature', orientation='h',
                         color='delta_auc', color_continuous_scale=['red', 'gray', 'green'])
            fig.add_vline(x=0, line_dash="dash", line_color="white")
            fig.update_layout(**PLT_LAYOUT)
            
            fig.update_traces(hovertemplate="Bu özellik çıkarılınca AUC %{x:.4f} kadar değişti.")
            
            if len(df_model) > 0:
                crit = df_model.iloc[0]['removed_feature']
                useless = df_model.iloc[-1]['removed_feature']
                text = f"En kritik özellik: {crit} | En gereksiz özellik: {useless}"
            else:
                text = ""
                
            return fig, text
        except Exception as e:
            print("Callback error:", e)
            return go.Figure(), ""
