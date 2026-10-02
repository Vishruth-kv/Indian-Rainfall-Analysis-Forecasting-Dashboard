import os
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
from dash import Dash
# ---------------- Load Data ----------------
# Use the script directory so this works locally and on Render
# ---------------- Load Data ----------------
# Use the specific Excel file path instead of auto-search
data_file = Path(r"C:\Rainfall_Analysis\data\rainfall_area-wt_India_1925-2025.xlsx")  # ✅ changed path

if not data_file.exists():
    raise FileNotFoundError(f"Data file not found: {data_file}")
print(f"📁 Loading rainfall data from: {data_file}")

# Load the file (supports .xlsx or .csv)
if data_file.suffix.lower() == ".xlsx":
    rainfall_data = pd.read_excel(data_file)
else:
    rainfall_data = pd.read_csv(data_file)

rainfall_data.columns = [c.strip() for c in rainfall_data.columns]

if 'YEAR' not in rainfall_data.columns:
    alt = next((c for c in rainfall_data.columns if 'year' in c.lower()), None)
    if alt:
        rainfall_data.rename(columns={alt: 'YEAR'}, inplace=True)
if 'ANNUAL' not in rainfall_data.columns:
    # sum numeric columns except YEAR (safe fallback)
    numeric_cols = [c for c in rainfall_data.select_dtypes('number').columns if c.upper() not in ('YEAR',)]
    if numeric_cols:
        rainfall_data['ANNUAL'] = rainfall_data[numeric_cols].sum(axis=1)
    else:
        rainfall_data['ANNUAL'] = 0

rainfall_data['YEAR'] = pd.to_numeric(rainfall_data['YEAR'], errors='coerce')
rainfall_data.dropna(subset=['YEAR'], inplace=True)
rainfall_data['YEAR'] = rainfall_data['YEAR'].astype(int)

# Monthly columns
monthly_expected = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
monthly_display = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
col_map = {c.upper(): c for c in rainfall_data.columns}
monthly_columns = [col_map.get(m, m) for m in monthly_expected if m in col_map]

# Safety: if monthly columns missing, fallback to numeric cols except YEAR/ANNUAL
if not monthly_columns:
    numeric_cols = [c for c in rainfall_data.select_dtypes('number').columns if c.upper() not in ('YEAR','ANNUAL')]
    monthly_columns = numeric_cols[:12]
    monthly_display = [c for c in monthly_columns]

# ---------------- Build Figures ----------------
def build_figures(template):
    # Annual trend (lines)
    annual = rainfall_data[['YEAR', 'ANNUAL']].dropna()
    fig_annual = go.Figure([
        go.Scatter(x=annual['YEAR'], y=annual['ANNUAL'], mode='lines', name='Annual'),
        go.Scatter(x=annual['YEAR'], y=[annual['ANNUAL'].mean()] * len(annual),
                   mode='lines', name='Mean', line=dict(dash='dash'))
    ])
    fig_annual.update_layout(title='Annual Rainfall Trend', template=template)

    # Average monthly rainfall (bar)
    monthly_avg = rainfall_data[monthly_columns].mean() if monthly_columns else pd.Series([0]*len(monthly_display))
    fig_monthly = px.bar(x=monthly_display, y=monthly_avg.values,
                         labels={'x': 'Month', 'y': 'Rainfall (mm)'},
                         title='Average Monthly Rainfall',
                         color=monthly_avg.values, color_continuous_scale='Blues',
                         template=template)

    # 10-Year rolling average (climate)
    rainfall_data['10YR AVG'] = rainfall_data['ANNUAL'].rolling(10).mean()
    fig_climate = go.Figure([
        go.Scatter(x=rainfall_data['YEAR'], y=rainfall_data['ANNUAL'], mode='lines', name='Annual'),
        go.Scatter(x=rainfall_data['YEAR'], y=rainfall_data['10YR AVG'], mode='lines', name='10-Year Avg')
    ])
    fig_climate.update_layout(title='Climate Change Impact (10-Year Rolling Avg)', template=template)

    # Prophet forecast
    try:
        rainfall_data['DATE'] = pd.to_datetime(rainfall_data['YEAR'], format='%Y')
        prophet_data = rainfall_data[['DATE', 'ANNUAL']].rename(columns={'DATE': 'ds', 'ANNUAL': 'y'})
        model = Prophet()
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=20, freq='Y')
        forecast = model.predict(future)
        fig_forecast = plot_plotly(model, forecast)
        fig_forecast.update_layout(title='Rainfall Forecast (Prophet)', template=template)
    except Exception as e:
        # If Prophet fails (rarely on some envs), create a placeholder figure
        fig_forecast = go.Figure()
        fig_forecast.add_annotation(text=f"Forecast unavailable: {e}", showarrow=False)
        fig_forecast.update_layout(title='Rainfall Forecast (Prophet)', template=template)

    # Clustering (KMeans)
    features = rainfall_data[monthly_columns + ['ANNUAL']].fillna(0) if monthly_columns else rainfall_data[['ANNUAL']].fillna(0)
    try:
        scaled = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42)
        rainfall_data['Cluster'] = kmeans.fit_predict(scaled)
        labels = {0: 'Dry', 1: 'Normal', 2: 'Wet'}
        rainfall_data['Category'] = rainfall_data['Cluster'].map(labels)
        fig_cluster = px.scatter(rainfall_data, x='YEAR', y='ANNUAL', color='Category',
                                 title='Rainfall Clustering (Dry / Normal / Wet)', template=template)
    except Exception as e:
        fig_cluster = go.Figure()
        fig_cluster.add_annotation(text=f"Clustering unavailable: {e}", showarrow=False)
        fig_cluster.update_layout(title='Rainfall Clustering', template=template)

    # --- Monthly Boxplot (distribution across years)
    try:
        df_melt = rainfall_data[['YEAR'] + monthly_columns].melt(id_vars='YEAR', var_name='Month', value_name='Rainfall')
        month_order = monthly_columns
        fig_box = px.box(df_melt, x='Month', y='Rainfall', category_orders={'Month': month_order},
                         labels={'Month': 'Month', 'Rainfall': 'Rainfall (mm)'},
                         title='Monthly Rainfall Distribution (boxplot)', template=template)
    except Exception as e:
        fig_box = go.Figure()
        fig_box.add_annotation(text=f"Boxplot unavailable: {e}", showarrow=False)
        fig_box.update_layout(title='Monthly Rainfall Distribution', template=template)

    # --- Cumulative Annual Rainfall
    try:
        cum = annual.copy()
        cum['CUM_SUM'] = cum['ANNUAL'].cumsum()
        fig_cum = go.Figure([
            go.Scatter(x=cum['YEAR'], y=cum['CUM_SUM'], mode='lines', name='Cumulative Sum'),
            go.Bar(x=cum['YEAR'], y=cum['ANNUAL'], name='Annual', opacity=0.3)
        ])
        fig_cum.update_layout(title='Cumulative Annual Rainfall (and yearly bars)', template=template,
                              yaxis_title='Cumulative Rainfall (mm)')
    except Exception as e:
        fig_cum = go.Figure()
        fig_cum.add_annotation(text=f"Cumulative plot unavailable: {e}", showarrow=False)
        fig_cum.update_layout(title='Cumulative Annual Rainfall', template=template)

    return {
        'Annual Rainfall Trend': fig_annual,
        'Average Monthly Rainfall': fig_monthly,
        'Climate Change (Rolling Avg)': fig_climate,
        'Rainfall Forecast (Prophet)': fig_forecast,
        'Rainfall Clusters': fig_cluster,
        'Monthly Distribution (Boxplot)': fig_box,
        'Cumulative Annual Rainfall': fig_cum
    }

figs_light = build_figures('plotly_white')
figs_dark = build_figures('plotly_dark')

# ---------------- Dash App ----------------

app = Dash(__name__)

# ✅ CSS fix for dropdown appearances (both light and dark look)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        /* Dropdown theme classes */
        .dropdown-dark .Select-control,
        .dropdown-dark .css-1n8s9fs-control {
            background-color: #ffffff !important;
            color: #0b1220 !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }
        .dropdown-dark .Select-placeholder,
        .dropdown-dark .Select-value,
        .dropdown-dark .css-1wa3eu0-placeholder,
        .dropdown-dark .css-1uccc91-singleValue {
            color: #0b1220 !important;
        }
        .dropdown-dark .Select-control.is-open,
        .dropdown-dark .Select-control:focus,
        .dropdown-dark .css-1n8s9fs-control:focus,
        .dropdown-dark .css-1n8s9fs-control.css-1n8s9fs--is-open {
            background-color: #0f172a !important;
            color: #E2E8F0 !important;
            border-color: #1e293b !important;
        }
        .dropdown-dark .Select-menu-outer,
        .dropdown-dark .css-1n8s9fs-menu,
        .dropdown-dark .Select-option,
        .dropdown-dark .css-1n8s9fs-option {
            background-color: #0f172a !important;
            color: #E2E8F0 !important;
        }
        .dropdown-dark .Select-option.is-focused,
        .dropdown-dark .css-1n8s9fs-option:hover {
            background-color: #162033 !important;
            color: #ffffff !important;
        }

        .dropdown-light .Select-control,
        .dropdown-light .css-1n8s9fs-control {
            background-color: #ffffff !important;
            color: #0b1220 !important;
            border: 1px solid rgba(0,0,0,0.08) !important;
        }
        .dropdown-light .Select-menu-outer,
        .dropdown-light .css-1n8s9fs-menu {
            background-color: #ffffff !important;
            color: #0b1220 !important;
        }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

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
            html.P("Explore India's rainfall patterns, trends, and forecasts (1925–2025)", id='welcome-text'),
            html.Button("Start Analysis 🌈", id='start-btn', n_clicks=0,
                        style={'marginTop': '30px', 'padding': '12px 30px', 'fontSize': '18px',
                               'background': '#00bcd4', 'color': 'white', 'border': 'none',
                               'borderRadius': '10px', 'cursor': 'pointer'})
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
            html.H2("📊 Rainfall Dashboard", id='header-title', style={'margin': '0', 'transition': 'color 0.3s ease'}),
            html.Button("🌙", id='theme-toggle', n_clicks=0,
                        style={'fontSize': '22px', 'background': 'none', 'border': 'none', 'cursor': 'pointer',
                               'transition': 'all 0.3s ease'})
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
                                 style={'width': '320px'},
                                 className='dropdown-light')
                ])
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px',
                      'alignItems': 'center', 'marginBottom': '15px'}),

            html.Div(id='year-row', style={'display': 'none', 'marginBottom': '15px'}, children=[
                html.Label("Select Year", id='year-label', style={'marginRight': '10px'}),
                dcc.Dropdown(id='year-dropdown',
                             options=[{'label': str(y), 'value': y} for y in years],
                             value=years[0], clearable=False, style={'width': '160px'},
                             className='dropdown-light')
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
    # show a sun icon when currently dark (to indicate clicking will switch to light), otherwise moon
    return {'theme': theme}, ('☀' if theme == 'dark' else '🌙')

@app.callback(
    Output('graph-display', 'figure'),
    Output('header', 'style'),
    Output('body', 'style'),
    Output('header-title', 'style'),
    Output('figure-select', 'className'),
    Output('year-dropdown', 'className'),
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

    dropdown_class = 'dropdown-dark' if dark else 'dropdown-light'

    # ALL YEARS: simple — return the chosen saved figure
    if view == 'all':
        return figs[selected], header_style, body_style, text_style, dropdown_class, dropdown_class

    # SINGLE YEAR: produce a figure appropriate for the selected visualization
    # Find the row for the selected year (safe fallback to first row)
    row = rainfall_data[rainfall_data['YEAR'] == year]
    if row.empty:
        row = rainfall_data.iloc[[0]]
        year = int(row.iloc[0]['YEAR'])
    yd = row.iloc[0]

    # Case A: Monthly-specific visualizations -> show that year's monthly bars / overlay on boxplot
    if selected == 'Average Monthly Rainfall':
        vals = [float(yd.get(m, 0)) for m in monthly_columns] if monthly_columns else [0] * len(monthly_display)
        fig = px.bar(x=monthly_display, y=vals,
                     labels={'x': 'Month', 'y': 'Rainfall (mm)'},
                     title=f'Average Monthly Rainfall — {year}',
                     color=vals, color_continuous_scale='Blues', template=template)
        return fig, header_style, body_style, text_style, dropdown_class, dropdown_class

    if selected == 'Monthly Distribution (Boxplot)':
        # base boxplot across years, then overlay this year's monthly points
        try:
            df_melt = rainfall_data[['YEAR'] + monthly_columns].melt(id_vars='YEAR', var_name='Month', value_name='Rainfall')
            month_order = monthly_columns
            fig = px.box(df_melt, x='Month', y='Rainfall', category_orders={'Month': month_order},
                         labels={'Month': 'Month', 'Rainfall': 'Rainfall (mm)'},
                         title=f'Monthly Rainfall Distribution (with {year} overlaid)', template=template)
            # overlay the selected year's monthly points
            vals = [float(yd.get(m, None) or 0) for m in monthly_columns]
            fig.add_trace(go.Scatter(x=monthly_columns, y=vals, mode='markers+lines', name=str(year),
                                     marker=dict(size=10, symbol='diamond'), hoverinfo='y'))
            # if display names differ, update x-axis ticktext
            fig.update_layout(xaxis={'tickmode':'array', 'tickvals': monthly_columns, 'ticktext': monthly_display})
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Boxplot unavailable: {e}", showarrow=False)
            fig.update_layout(title='Monthly Rainfall Distribution', template=template)
        return fig, header_style, body_style, text_style, dropdown_class, dropdown_class

    # Case B: Year-series visualizations -> take the saved figure and highlight the year
    base_fig = figs.get(selected, None)
    if base_fig is None:
        # fallback: show simple monthly bar for year
        vals = [float(yd.get(m, 0)) for m in monthly_columns] if monthly_columns else [0] * len(monthly_display)
        fig = px.bar(x=monthly_display, y=vals, labels={'x': 'Month', 'y': 'Rainfall (mm)'},
                     title=f'{selected} — {year}', color=vals, color_continuous_scale='Blues', template=template)
        return fig, header_style, body_style, text_style, dropdown_class, dropdown_class

    # copy figure so we don't mutate cached figs
    fig = go.Figure(base_fig)

    # add vertical highlight line at the selected year (when x axis is YEAR)
    try:
        fig.add_vline(x=year, line_width=2, line_dash="dash", line_color="crimson", opacity=0.9)
    except Exception:
        # not all figures have numeric-year x-axis — ignore errors silently
        pass

    # add annotation for the year with the annual value if available
    annual_val = None
    try:
        annual_val = float(yd.get('ANNUAL', None) or 0)
        fig.add_annotation(x=year, y=annual_val,
                           text=f"{year}: {annual_val:.1f}",
                           showarrow=True, arrowhead=2, ax=0, ay=-40,
                           bgcolor="rgba(255,255,255,0.8)" if not dark else "rgba(11,17,32,0.85)")
    except Exception:
        # ignore if annual not applicable for this visualization
        pass

    # For clustering or scatter-like plots, also add a highlighted marker for the year
    try:
        if 'Cluster' in rainfall_data.columns and annual_val is not None:
            fig.add_trace(go.Scatter(x=[year], y=[annual_val], mode='markers', marker=dict(size=12, color='crimson'),
                                     name=f"Selected {year}", hoverinfo='y'))
    except Exception:
        pass

    # Ensure template is consistent with theme
    fig.update_layout(template=template, title=f"{selected} — highlighted {year}")

    return fig, header_style, body_style, text_style, dropdown_class, dropdown_class

# ---------------- Run ----------------
if __name__ == '__main__':

    print("🚀 Rainfall Dashboard (Final dark mode dropdown fix)")
    # Use PORT env var (Render provides it); fallback to 8050 for local runs
    port = int(os.environ.get("PORT", 8050))
    # BIND TO LOOPBACK (127.0.0.1) so console shows http://127.0.0.1:8050/ and the app is only reachable locally.
    app.run(debug=True, host='127.0.0.1', port=port)
#done