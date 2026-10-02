import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from prophet.plot import plot_plotly
from dash import Dash, html, dcc, Input, Output, State

# --------------------- LOAD DATA ---------------------
rainfall_data = pd.read_excel(r"C:\Rainfall_Analysis\data\rainfall_data.xlsx")

monthly_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
seasonal_columns = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']

# -------- FIGURE 1: Annual Rainfall Trend --------
annual_rainfall = rainfall_data[['YEAR', 'ANNUAL']]
fig_annual = go.Figure()
fig_annual.add_trace(go.Scatter(x=annual_rainfall['YEAR'], y=annual_rainfall['ANNUAL'],
                                mode='lines', name='Annual Rainfall'))
fig_annual.add_trace(go.Scatter(x=annual_rainfall['YEAR'],
                                y=[annual_rainfall['ANNUAL'].mean()] * len(annual_rainfall),
                                mode='lines', name='Mean', line=dict(dash='dash')))
fig_annual.update_layout(title='Trend in Annual Rainfall')

# -------- FIGURE 2: Monthly Average Rainfall --------
monthly_avg = rainfall_data[monthly_columns].mean()
fig_monthly = px.bar(x=monthly_avg.index, y=monthly_avg.values,
                     labels={'x': 'Month', 'y': 'Rainfall (mm)'},
                     title='Average Monthly Rainfall',
                     color=monthly_avg.values, color_continuous_scale='Blues')

# -------- FIGURE 3: Seasonal Rainfall --------
seasonal_avg = rainfall_data[seasonal_columns].mean()
fig_seasonal = px.bar(x=seasonal_avg.index, y=seasonal_avg.values,
                      title='Seasonal Rainfall Distribution',
                      color=seasonal_avg.values, color_continuous_scale='Tealgrn')

# -------- FIGURE 4: 10-Year Rolling Average --------
rainfall_data['10-Year Rolling Avg'] = rainfall_data['ANNUAL'].rolling(10).mean()
fig_climate = go.Figure()
fig_climate.add_trace(go.Scatter(x=rainfall_data['YEAR'], y=rainfall_data['ANNUAL'],
                                 mode='lines', name='Annual'))
fig_climate.add_trace(go.Scatter(x=rainfall_data['YEAR'], y=rainfall_data['10-Year Rolling Avg'],
                                 mode='lines', name='10-Year Avg'))
fig_climate.update_layout(title='Climate Change Impact (10-Year Rolling Avg)')

# -------- FIGURE 5: Prophet Forecast --------
rainfall_data['DATE'] = pd.to_datetime(rainfall_data['YEAR'], format='%Y')
prophet_data = rainfall_data[['DATE', 'ANNUAL']].rename(columns={'DATE': 'ds', 'ANNUAL': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future = prophet_model.make_future_dataframe(periods=20, freq='YE')
forecast = prophet_model.predict(future)
fig_forecast = plot_plotly(prophet_model, forecast)
fig_forecast.update_layout(title='Rainfall Forecast Using Prophet')

# -------- FIGURE 6: Clustering --------
features = rainfall_data[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'ANNUAL']]
scaled = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
rainfall_data['Cluster'] = kmeans.fit_predict(scaled)
labels = {0: 'Dry', 1: 'Normal', 2: 'Wet'}
rainfall_data['Category'] = rainfall_data['Cluster'].map(labels)
fig_cluster = px.scatter(rainfall_data, x='YEAR', y='ANNUAL', color='Category',
                         title='Clustering of Years by Rainfall Pattern')

# -------- Dropdown Figures --------
figures = {
    'Annual Rainfall Trend': fig_annual,
    'Average Monthly Rainfall': fig_monthly,
    'Seasonal Rainfall': fig_seasonal,
    'Climate Change (Rolling Avg)': fig_climate,
    'Rainfall Forecast (Prophet)': fig_forecast,
    'Rainfall Clusters': fig_cluster
}

# ---------------- DASH APP ----------------
app = Dash(__name__)

# -------- Layout with Welcome Screen --------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    # --- Welcome Screen ---
    html.Div(id='welcome-screen', children=[
        html.H1("🌦️ Indian Rainfall Analysis Dashboard", 
                style={'fontSize': '42px', 'color': 'white', 'textAlign': 'center'}),
        html.P("Discover patterns, trends, and forecasts of India's rainfall data.",
               style={'fontSize': '20px', 'color': '#e0f7fa', 'textAlign': 'center'}),
        html.Button("Start Analysis ", id='start-btn', n_clicks=0,
                    style={'fontSize': '20px', 'padding': '15px 40px',
                           'borderRadius': '15px', 'backgroundColor': '#00bcd4',
                           'color': 'white', 'border': 'none', 'cursor': 'pointer',
                           'marginTop': '40px'}),
    ], style={
        'background': 'linear-gradient(to right, #0072ff, #00c6ff)',
        'height': '100vh',
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'justifyContent': 'center',
        'transition': 'opacity 1s ease-in-out'
    }),

    # --- Dashboard Screen ---
    html.Div(id='dashboard-screen', style={'display': 'none', 'padding': '20px'}, children=[
        html.H1("📊 Rainfall Analysis Dashboard", style={'textAlign': 'center', 'color': '#003366'}),
        html.Label("Select Visualization:", style={'fontSize': '18px', 'marginLeft': '10px'}),
        dcc.Dropdown(
            id='figure-select',
            options=[{'label': k, 'value': k} for k in figures.keys()],
            value='Annual Rainfall Trend',
            clearable=False,
            style={'width': '60%', 'margin': '20px auto'}
        ),
        dcc.Graph(id='graph-display', style={'height': '75vh'})
    ])
])

# -------- Callback to switch screens --------
@app.callback(
    Output('welcome-screen', 'style'),
    Output('dashboard-screen', 'style'),
    Input('start-btn', 'n_clicks')
)
def switch_screens(n_clicks):
    if n_clicks > 0:
        return {'display': 'none'}, {'display': 'block'}
    return {'display': 'flex',
            'background': 'linear-gradient(to right, #0072ff, #00c6ff)',
            'height': '100vh',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'center'}, {'display': 'none'}

# -------- Callback for graph update --------
@app.callback(
    Output('graph-display', 'figure'),
    Input('figure-select', 'value')
)
def update_graph(selected):
    return figures[selected]


if __name__ == '__main__':
    app.run(debug=True)
