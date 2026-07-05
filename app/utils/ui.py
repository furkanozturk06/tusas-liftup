from dash import html
from dash_iconify import DashIconify

PLT_LAYOUT = dict(
    template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    font=dict(family="IBM Plex Sans", color="#475569"),
    margin=dict(l=40, r=20, t=40, b=30),
)


def icon(name, size=18, color=None):
    return DashIconify(icon=name, width=size, color=color or "#64748B")


def metric_card(ic, value, label, color="blue", footer=None):
    children = [
        html.Div(icon(ic, 20), className="metric-icon"),
        html.Div(str(value), className="metric-value"),
        html.Div(label, className="metric-label"),
    ]
    if footer:
        children.append(html.Div(footer, className="metric-card-footer"))
    return html.Div(className=f"metric-card {color}", children=children)


_STAT_ACCENT = {"blue": "#2563EB", "red": "#EF4444", "green": "#10B981",
                "cyan": "#0891B2", "yellow": "#D97706", "purple": "#7C3AED", "slate": "#1E293B"}


def stat_strip(items):
    row = []
    for i, it in enumerate(items):
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else None
        acc = it[3] if len(it) > 3 else None
        accent = acc if (acc and str(acc).startswith("#")) else _STAT_ACCENT.get(acc, "#1E293B")
        if i:
            row.append(html.Div(style={"width": "1px", "alignSelf": "stretch",
                                       "background": "#E2E8F0", "margin": "6px 0"}))
        row.append(html.Div(style={"flex": "1", "padding": "2px 18px", "minWidth": "0"}, children=[
            html.Div(label, style={"fontSize": "10px", "letterSpacing": "1.2px", "color": "#94A3B8",
                                   "fontWeight": "600", "textTransform": "uppercase", "marginBottom": "4px"}),
            html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "7px"}, children=[
                html.Span(str(value), style={"fontSize": "20px", "fontWeight": "700", "color": accent}),
                html.Span(sub or "", style={"fontSize": "11px", "color": "#64748B"}),
            ]),
        ]))
    return html.Div(className="panel", style={"display": "flex", "alignItems": "center",
                                              "padding": "14px 4px", "marginBottom": "20px"}, children=row)
