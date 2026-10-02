import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- Configuration and Data Loading ---

# Set up the page layout
st.set_page_config(
    page_title="Rainfall Analysis Web App",
    layout="wide"
)

# @st.cache_data is used to cache the data so it only loads once, speeding up the app
@st.cache_data
def load_data():
    """
    Loads the rainfall data.
    
    NOTE: The original hardcoded path has been removed.
    You MUST place your 'rainfall_data.xlsx' file in the same directory
    as this app.py file for the try-block to work, or use a st.file_uploader.
    
    A mock DataFrame is created as a fallback for demonstration purposes.
    """
    try:
        # --- ATTEMPT TO LOAD USER'S DATA ---
        rainfall_data = pd.read_excel(r"C:\Rainfall_Analysis\data\rainfall_data.xlsx")
        st.sidebar.success("✅ Successfully loaded rainfall_data.xlsx")
        
    except FileNotFoundError:
        st.sidebar.error("⚠ 'C:\Rainfall_Analysis\data\rainfall_data.xlsx' not found. Using MOCK DATA for demonstration.")
        # --- MOCK DATA FOR DEMONSTRATION ---
        years = range(1901, 2016)
        n_years = len(years)
        data = {
            'YEAR': list(years),
            'ANNUAL': np.random.normal(1200, 150, n_years),
            'JAN': np.random.normal(20, 5, n_years),
            'FEB': np.random.normal(15, 5, n_years),
            'MAR': np.random.normal(30, 10, n_years),
            'APR': np.random.normal(40, 10, n_years),
            'MAY': np.random.normal(100, 30, n_years),
            'JUN': np.random.normal(300, 50, n_years),
            'JUL': np.random.normal(350, 50, n_years),
            'AUG': np.random.normal(300, 50, n_years),
            'SEP': np.random.normal(200, 40, n_years),
            'OCT': np.random.normal(100, 30, n_years),
            'NOV': np.random.normal(40, 10, n_years),
            'DEC': np.random.normal(25, 5, n_years),
        }
        rainfall_data = pd.DataFrame(data)
        rainfall_data['ANNUAL'] = rainfall_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].sum(axis=1)

    return rainfall_data.round(2)

# Load the data
rainfall_data = load_data()
MIN_YEAR = int(rainfall_data['YEAR'].min())
MAX_YEAR = int(rainfall_data['YEAR'].max())
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# --- Sidebar for User Selections ---
st.sidebar.title("Filter Options")

# 1. Analysis Type Selector
analysis_type = st.sidebar.radio(
    "Select Analysis Scope:",
    ("Full Analysis", "Yearly Trend Only", "Monthly Distribution Only"),
    help="Choose the type of plot you want to generate."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Time Range Selection")

# 2. Year Range Selector (Year to Year)
start_year, end_year = st.sidebar.slider(
    "Select Year Range:",
    MIN_YEAR, MAX_YEAR, (MIN_YEAR, MAX_YEAR)
)

# 3. Month Range Selector (Month to Month)
start_month, end_month = st.sidebar.select_slider(
    'Select Month Range:',
    options=MONTHS,
    value=('JAN', 'DEC'),
    help="Filters which months are included in the Monthly Distribution chart."
)
start_month_idx = MONTHS.index(start_month)
end_month_idx = MONTHS.index(end_month)
selected_monthly_columns = MONTHS[start_month_idx : end_month_idx + 1]

# --- Main App Title and Data Filtering ---
st.title("🌧 Interactive Rainfall Analysis")
st.markdown(f"*Data Range:* {MIN_YEAR} - {MAX_YEAR} | *Filtered Range:* {start_year} - {end_year}")
st.markdown("---")

# Data Filtering based on Year Slider
filtered_data = rainfall_data[
    (rainfall_data['YEAR'] >= start_year) &
    (rainfall_data['YEAR'] <= end_year)
]

# --- 1. Annual Rainfall Trend Plot (Yearly Analysis) ---
if analysis_type in ["Full Analysis", "Yearly Trend Only"]:
    st.header("Annual Rainfall Trend Over Time")
    
    if not filtered_data.empty:
        annual_rainfall = filtered_data[['YEAR', 'ANNUAL']]
        mean_rainfall = annual_rainfall['ANNUAL'].mean()

        # Plotly code adapted from your original script
        fig_annual = go.Figure()
        fig_annual.add_trace(go.Scatter(
            x=annual_rainfall['YEAR'],
            y=annual_rainfall['ANNUAL'],
            mode='lines',
            name='Annual Rainfall',
            line=dict(color='blue', width=2),
            opacity=0.7
        ))
        fig_annual.add_trace(go.Scatter(
            x=annual_rainfall['YEAR'],
            y=[mean_rainfall] * len(annual_rainfall),
            mode='lines',
            name=f'Mean Rainfall ({mean_rainfall:.2f} mm)',
            line=dict(color='red', dash='dash')
        ))
        
        fig_annual.update_layout(
            title=f'Trend in Annual Rainfall in India ({start_year}-{end_year})',
            xaxis_title='Year',
            yaxis_title='Rainfall (mm)',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig_annual, use_container_width=True)
    else:
        st.warning("No data found for the selected year range.")

st.markdown("---")

# --- 2. Monthly Rainfall Distribution Plot (Monthly Analysis) ---
if analysis_type in ["Full Analysis", "Monthly Distribution Only"]:
    st.header(f"Average Monthly Rainfall Distribution")
    
    if not filtered_data.empty:
        # Calculate the average rainfall only for the selected months and selected year range
        monthly_avg = filtered_data[selected_monthly_columns].mean()
        total_mean_rainfall = monthly_avg.mean()

        # Plotly Express code adapted from your original script
        fig_monthly = px.bar(
            x=monthly_avg.index,
            y=monthly_avg.values,
            labels={'x': 'Month', 'y': 'Average Rainfall (mm)'},
            title=f'Average Monthly Rainfall: {start_month} to {end_month} ({start_year}-{end_year})',
            text=monthly_avg.values.round(2)
        )
        
        fig_monthly.add_hline(
            y=total_mean_rainfall,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean Monthly Rainfall ({total_mean_rainfall:.2f} mm)",
            annotation_position="top right"
        )
        fig_monthly.update_traces(marker_color='skyblue', marker_line_color='black', marker_line_width=1)
        fig_monthly.update_layout(template='plotly_white', height=500)
        
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.warning("No data found for the selected year range.")

st.markdown("---")

# --- Export/Download Section ---
st.header("Export Options")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Download Filtered Data")
    # Convert filtered data to CSV for download
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f'rainfall_data_{start_year}to{end_year}.csv',
        mime='text/csv',
        help="Downloads the data table used for the current analysis."
    )

with col2:
    st.subheader("Export to PDF / Print")
    st.info("The simplest and most reliable way to export the entire dashboard (including all charts) to a PDF document is by using your web browser's print function.")
    st.markdown("1. Press **Ctrl + P** (Windows/Linux) or **Cmd + P** (Mac).")
    st.markdown("2. In the print dialog, change the *Destination* or *Printer* to **Save as PDF**.")