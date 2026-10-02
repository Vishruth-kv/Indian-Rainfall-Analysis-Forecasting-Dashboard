import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from prophet.plot import plot_plotly
from dash import Dash, html, dcc, Input, Output, State
from pathlib import Path
import sys


import pandas as pd

data_path = r"C:\Rainfall_Analysis\data\rainfall_area-wt_India_1901-2015.xlsx"
rainfall_data = pd.read_excel(data_path)
print(rainfall_data.head())

# ---------------- Load Data ----------------
data_dir = Path(r"C:\Rainfall_Analysis\data\rainfall_area-wt_India_1901-2015.xlsx")
candidates = list(data_dir.glob(".xlsx")) + list(data_dir.glob(".csv"))
'''if not candidates:
    raise FileNotFoundError(f"No rainfall data found in {data_dir}")'''
data_file = candidates[0]
print(f"📁 Loading rainfall data from: {data_file}")

rainfall_data = pd.read_excel(data_file) if data_file.suffix.lower() == ".xlsx" else pd.read_csv(data_file)
rainfall_data.columns = [c.strip() for c in rainfall_data.columns]

# Fix columns
if 'YEAR' not in rainfall_data.columns:
    alt = next((c for c in rainfall_data.columns if 'year' in c.lower()), None)
    rainfall_data.rename(columns={alt: 'YEAR'}, inplace=True)
if 'ANNUAL' not in rainfall_data.columns:
    rainfall_data['ANNUAL'] = rainfall_data.select_dtypes('number').sum(axis=1)

rainfall_data['YEAR'] = pd.to_numeric(rainfall_data['YEAR'], errors='coerce')
rainfall_data.dropna(subset=['YEAR'], inplace=True)
rainfall_data['YEAR'] = rainfall_data['YEAR'].astype(int)

# Monthly columns
monthly_expected = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
monthly_display = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
col_map = {c.upper(): c for c in rainfall_data.columns}
monthly_columns = [col_map.get(m, m) for m in monthly_expected if m in col_map]

# ---------------- Build Figures ----------------
def build_figures(template):
    annual = rainfall_data[['YEAR', 'ANNUAL']].dropna()
    fig_annual = go.Figure([
        go.Scatter(x=annual['YEAR'], y=annual['ANNUAL'], mode='lines', name='Annual'),
        go.Scatter(x=annual['YEAR'], y=[annual['ANNUAL'].mean()] * len(annual),
                   mode='lines', name='Mean', line=dict(dash='dash'))
    ])
    fig_annual.update_layout(title='Annual Rainfall Trend', template=template)

    monthly_avg = rainfall_data[monthly_columns].mean()
    fig_monthly = px.bar(x=monthly_display, y=monthly_avg.values,
                         labels={'x': 'Month', 'y': 'Rainfall (mm)'},
                         title='Average Monthly Rainfall',
                         color=monthly_avg.values, color_continuous_scale='Blues',
                         template=template)

    rainfall_data['10YR AVG'] = rainfall_data['ANNUAL'].rolling(10).mean()
    fig_climate = go.Figure([
        go.Scatter(x=rainfall_data['YEAR'], y=rainfall_data['ANNUAL'], mode='lines', name='Annual'),
        go.Scatter(x=rainfall_data['YEAR'], y=rainfall_data['10YR AVG'], mode='lines', name='10-Year Avg')
    ])
    fig_climate.update_layout(title='Climate Change Impact (10-Year Rolling Avg)', template=template)

    rainfall_data['DATE'] = pd.to_datetime(rainfall_data['YEAR'], format='%Y')
    prophet_data = rainfall_data[['DATE', 'ANNUAL']].rename(columns={'DATE': 'ds', 'ANNUAL': 'y'})
    model = Prophet()
    model.fit(prophet_data)
    future = model.make_future_dataframe(periods=20, freq='Y')
    forecast = model.predict(future)
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(title='Rainfall Forecast (Prophet)', template=template)

    features = rainfall_data[monthly_columns + ['ANNUAL']].fillna(0)
    scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    rainfall_data['Cluster'] = kmeans.fit_predict(scaled)
    labels = {0: 'Dry', 1: 'Normal', 2: 'Wet'}
    rainfall_data['Category'] = rainfall_data['Cluster'].map(labels)
    fig_cluster = px.scatter(rainfall_data, x='YEAR', y='ANNUAL', color='Category',
                             title='Rainfall Clustering (Dry / Normal / Wet)', template=template)

    return {
        'Annual Rainfall Trend': fig_annual,
        'Average Monthly Rainfall': fig_monthly,
        'Climate Change (Rolling Avg)': fig_climate,
        'Rainfall Forecast (Prophet)': fig_forecast,
        'Rainfall Clusters': fig_cluster
    }

figs_light = build_figures('plotly_white')
figs_dark = build_figures('plotly_dark')

# ---------------- Dash App ----------------
app = Dash(_name_)
app.title = "Rainfall Dashboard"

years = sorted(rainfall_data['YEAR'].unique())

# ---------------- Theme Styling ----------------
def get_colors(dark):
    return {
        "bg": "#0f172a" if dark else "#f8fafc",
        "text": "#E2E8F0" if dark else "#1E293B",
        "header": "#1e293b" if dark else "#0072ff",
        "secondary": "#00bcd4",
    }

# ---------------- Layout ----------------
app.layout = html.Div([
    dcc.Store(id='theme-store', storage_type='local'),
    dcc.Location(id='url', refresh=False),

    # Welcome Screen
    html.Div(id='welcome-screen', children=[
        html.Div([
            html.H1("🌦 Indian Rainfall Analysis Dashboard", id='welcome-title'),
            html.P("Explore India's rainfall patterns, trends, and forecasts (1901–2015)",
                   id='welcome-text'),
            html.Button("Start Analysis 🌈", id='start-btn', n_clicks=0,
                        style={'marginTop': '30px', 'padding': '12px 30px',
                               'fontSize': '18px', 'background': '#00bcd4',
                               'color': 'white', 'border': 'none', 'borderRadius': '10px',
                               'cursor': 'pointer'})
        ], style={'textAlign': 'center'})
    ], style={'height': '100vh', 'display': 'flex', 'alignItems': 'center',
              'justifyContent': 'center', 'flexDirection': 'column',
              'background': 'linear-gradient(135deg,#0072ff,#00c6ff)',
              'transition': 'all 0.3s ease'}),

    # Dashboard
    html.Div(id='dashboard-screen', style={'display': 'none', 'transition': 'all 0.3s ease'}, children=[
        html.Div(id='header', style={'display': 'flex', 'justifyContent': 'space-between',
                                     'alignItems': 'center', 'padding': '16px 24px',
                                     'borderRadius': '8px', 'transition': 'all 0.3s ease'}, children=[
            html.H2("📊 Rainfall Dashboard", id='header-title',
                    style={'margin': '0', 'transition': 'color 0.3s ease'}),
            html.Button("🌙", id='theme-toggle', n_clicks=0,
                        style={'fontSize': '22px', 'background': 'none', 'border': 'none',
                               'cursor': 'pointer', 'transition': 'all 0.3s ease'})
        ]),
        html.Div(id='body', style={'padding': '25px', 'minHeight': '100vh',
                                   'transition': 'background 0.3s ease, color 0.3s ease'}, children=[
            html.Div([
                html.Div([
                    html.Label("View Mode", id='view-label'),
                    dcc.RadioItems(id='view-mode',
                                   options=[{'label': 'All Years', 'value': 'all'},
                                            {'label': 'Single Year', 'value': 'single'}],
                                   value='all', inline=True)
                ], style={'marginRight': '25px'}),
                html.Div([
                    html.Label("Select Visualization", id='viz-label'),
                    dcc.Dropdown(id='figure-select',
                                 options=[{'label': k, 'value': k} for k in figs_light.keys()],
                                 value='Annual Rainfall Trend', clearable=False,
                                 style={'width': '300px'})
                ])
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px',
                      'alignItems': 'center', 'marginBottom': '15px'}),

            html.Div(id='year-row', style={'display': 'none', 'marginBottom': '15px'}, children=[
                html.Label("Select Year", id='year-label',
                           style={'marginRight': '10px'}),
                dcc.Dropdown(id='year-dropdown',
                             options=[{'label': str(y), 'value': y} for y in years],
                             value=years[0], clearable=False, style={'width': '160px'})
            ]),

            dcc.Graph(id='graph-display', style={'height': '78vh'})
        ])
    ])
])

# ---------------- Callbacks ----------------
@app.callback(
    Output('welcome-screen', 'style'),
    Output('dashboard-screen', 'style'),
    Output('welcome-title', 'style'),
    Output('welcome-text', 'style'),
    Input('start-btn', 'n_clicks'),
    State('theme-store', 'data')
)
def show_dashboard(n_clicks, theme_data):
    dark = theme_data and theme_data.get('theme') == 'dark'
    bg_gradient = 'linear-gradient(135deg,#0f172a,#1e293b)' if dark else 'linear-gradient(135deg,#0072ff,#00c6ff)'
    text_color = '#E2E8F0' if dark else '#1E293B'

    welcome_style = {
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'flexDirection': 'column',
        'height': '100vh',
        'background': bg_gradient,
        'transition': 'all 0.3s ease'
    }
    title_style = {'color': text_color, 'fontSize': '42px', 'transition': 'color 0.3s ease'}
    text_style = {'color': text_color, 'fontSize': '18px', 'transition': 'color 0.3s ease'}

    if n_clicks and n_clicks > 0:
        return {'display': 'none'}, {'display': 'block'}, title_style, text_style

    return welcome_style, {'display': 'none'}, title_style, text_style

@app.callback(Output('year-row', 'style'), Input('view-mode', 'value'))
def toggle_year(view):
    return {'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'} if view == 'single' else {'display': 'none'}

@app.callback(Output('theme-store', 'data'), Output('theme-toggle', 'children'),
              Input('theme-toggle', 'n_clicks'), State('theme-store', 'data'))
def toggle_theme(n_clicks, data):
    theme = data.get('theme', 'light') if data else 'light'
    if n_clicks:
        theme = 'dark' if theme == 'light' else 'light'
    return {'theme': theme}, ('☀' if theme == 'dark' else '🌙')

@app.callback(
    Output('graph-display', 'figure'),
    Output('header', 'style'),
    Output('body', 'style'),
    Output('header-title', 'style'),
    Input('figure-select', 'value'),
    Input('view-mode', 'value'),
    Input('year-dropdown', 'value'),
    Input('theme-store', 'data')
)
def update_figure(selected, view, year, theme_data):
    dark = theme_data and theme_data.get('theme') == 'dark'
    figs = figs_dark if dark else figs_light
    colors = get_colors(dark)
    template = 'plotly_dark' if dark else 'plotly_white'

    header_style = {'display': 'flex', 'justifyContent': 'space-between',
                    'alignItems': 'center', 'padding': '16px 24px',
                    'background': colors['header'], 'color': 'white',
                    'borderRadius': '8px', 'transition': 'all 0.3s ease'}
    body_style = {'background': colors['bg'], 'color': colors['text'],
                  'padding': '25px', 'minHeight': '100vh',
                  'transition': 'background 0.3s ease, color 0.3s ease'}
    text_style = {'color': colors['text'], 'margin': '0', 'transition': 'color 0.3s ease'}

    if view == 'all':
        return figs[selected], header_style, body_style, text_style

    row = rainfall_data[rainfall_data['YEAR'] == year]
    if row.empty:
        row = rainfall_data.iloc[[0]]
    yd = row.iloc[0]
    vals = [float(yd.get(m, 0)) for m in monthly_columns]
    fig = px.bar(x=monthly_display, y=vals,
                 labels={'x': 'Month', 'y': 'Rainfall (mm)'},
                 title=f'Monthly Rainfall - {year}',
                 color=vals, color_continuous_scale='Blues', template=template)
    return fig, header_style, body_style, text_style

# ---------------- Run ----------------
if _name_ == '_main_':
    print("🚀 Rainfall Dashboard (Dark/Light Mode + Smooth Transitions)")
    app.run(debug=True, port=8050)