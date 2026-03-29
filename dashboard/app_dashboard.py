"""
============================================================
LinkedIn Jobs Intelligence System
STEP 6 — DASHBOARD: Interactive Analytics
============================================================
- Interactive dashboard using Plotly Dash
- Live filtering (Location, Category)
- KPI Cards (Total Jobs, Median Salary, Top Skill)
- Charts (Skills Bar, Salary Distribution, Category Pie)

Usage:
    python dashboard/app_dashboard.py

Output:
    A dash server at http://localhost:8050
============================================================
"""

import logging
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
FEATURES_CSV  = Path("data/processed/jobs_features.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── Dash App ───────────────────────────────────────────────
def run_dashboard():
    if not FEATURES_CSV.exists():
        log.error(f"  Missing features data: {FEATURES_CSV}. Run full pipeline first.")
        return

    df = pd.read_csv(FEATURES_CSV)
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    # 1. Content Components
    header = html.Div([
        html.H1("LinkedIn Jobs Intelligence Dashboard", className="text-center my-4"),
        html.Hr(style={"borderColor": "#444"}),
    ])

    # Filters
    filters = dbc.Card([
        dbc.CardBody([
            html.H5("🔎 Filters", className="card-title"),
            html.Label("Select Category:"),
            dcc.Dropdown(
                id="cat-filter",
                options=[{"label": c, "value": c} for c in sorted(df['category'].unique())],
                value=None,
                placeholder="All Categories",
                style={"color": "#333"}
            ),
            html.Br(),
            html.Label("Select Location:"),
            dcc.Dropdown(
                id="loc-filter",
                options=[{"label": l, "value": l} for l in sorted(df['location'].unique()) if isinstance(l, str)],
                value=None,
                placeholder="Global",
                style={"color": "#333"}
            ),
        ])
    ], className="mb-4")

    # KPIs
    kpis = html.Div(id="kpi-container")

    # Charts
    charts = dbc.Row([
        dbc.Col(dcc.Graph(id="skill-chart"), width=6),
        dbc.Col(dcc.Graph(id="salary-chart"), width=6),
    ])

    # Layout
    app.layout = dbc.Container([
        header,
        dbc.Row([
            dbc.Col(filters, width=3),
            dbc.Col([
                kpis,
                charts
            ], width=9)
        ])
    ], fluid=True)

    # 2. Callbacks
    @app.callback(
        [Output("kpi-container", "children"),
         Output("skill-chart", "figure"),
         Output("salary-chart", "figure")],
        [Input("cat-filter", "value"),
         Input("loc-filter", "value")]
    )
    def update_dashboard(cat, loc):
        filtered = df.copy()
        if cat: filtered = filtered[filtered["category"] == cat]
        if loc: filtered = filtered[filtered["location"] == loc]

        # KPIs
        total_jobs = f"{len(filtered):,}"
        med_salary = f"${filtered['salary_numeric'].median():,.0f}" if len(filtered) > 0 else "$0"
        
        kpi_row = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Jobs"), html.H2(total_jobs)]), color="primary", inverse=True)),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Median Salary"), html.H2(med_salary)]), color="success", inverse=True)),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Analysis Status"), html.H2("LIVE")]), color="info", inverse=True)),
        ], className="mb-4")

        # Skill Chart
        skills = filtered['skills'].dropna().str.split(", ").explode()
        skill_df = skills.value_counts().head(15).reset_index()
        skill_df.columns = ["Skill", "Count"]

        skill_fig = px.bar(
            skill_df, x="Count", y="Skill", orientation="h",
            title="🔥 Top 15 In-Demand Skills",
            template="plotly_dark", color="Count", color_continuous_scale="Blues"
        ).update_layout(yaxis={'categoryorder':'total ascending'})

        # Salary Chart
        sal_fig = px.histogram(
            filtered, x="salary_numeric", nbins=20,
            title="💰 Salary Distribution",
            template="plotly_dark", color_discrete_sequence=["#28A745"]
        )

        return kpi_row, skill_fig, sal_fig

    log.info("\n✅  Dashboard server started → http://localhost:8050")
    app.run_server(debug=True, port=8050)

if __name__ == "__main__":
    run_dashboard()
