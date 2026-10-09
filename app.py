import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="Geo-Alpha Predictor", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for a polished, professional UI
st.markdown("""
<style>
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style the metric cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #2E2E3E;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom Headers */
    h1 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -1px;
    }
    h3 {
        color: #A0A0B0;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.title("🌍 Geopolitical Alpha Predictor")
st.markdown("### AI-driven macro forecasting and risk analysis for global equities.")
st.write("---")

# 2. Sidebar Controls
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942206.png", width=60) # Placeholder finance icon
    st.title("Dashboard Controls")
    st.write("Configure the asset and timeline below:")
    ticker = st.text_input("Stock Ticker", value="SPY").upper()
    benchmark = st.selectbox("Historical Window", ["1y", "2y", "5y", "max"], index=2)
    
    st.markdown("---")
    st.markdown("**About this Engine:**")
    st.markdown("This Random Forest model calculates real-time technicals and merges them with a Geopolitical Risk (GPR) index to predict 3-month asset direction.")

# 3. Load Model and Macro Data
@st.cache_resource
def load_model():
    return joblib.load('geo_alpha_model.joblib')

@st.cache_data
def load_macro_data():
    df = pd.read_csv('month_1_master_dataset.csv', index_col='Date', parse_dates=True)
    return df[['GPR_Index']]

try:
    model = load_model()
    gpr_data = load_macro_data()
except FileNotFoundError:
    st.error("Missing files. Ensure 'month_1_master_dataset.csv' and 'geo_alpha_model.joblib' are uploaded to GitHub.")
    st.stop()

# 4. Fetch Live Market Data
with st.spinner(f"Aggregating live market data for {ticker}..."):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=benchmark)
        
        if hist.empty:
            st.error(f"Could not find market data for ticker '{ticker}'.")
            st.stop()
            
        hist['Daily_Return'] = hist['Close'].pct_change()
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist = hist.dropna()
        
        hist.index = hist.index.tz_localize(None)
        merged_data = hist.join(gpr_data, how='left')
        merged_data['GPR_Index'] = merged_data['GPR_Index'].ffill().bfill()
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# 5. Prediction & Confidence
features = ['Daily_Return', 'SMA_20', 'SMA_50', 'GPR_Index']
latest_data = merged_data.iloc[-1]
X_latest = latest_data[features].values.reshape(1, -1)

prediction = model.predict(X_latest)[0]
probabilities = model.predict_proba(X_latest)[0]
confidence = probabilities[prediction] * 100

prediction_label = "Bullish (Higher)" if prediction == 1 else "Bearish (Lower)"
status_color = "normal" if prediction == 1 else "inverse"

# 6. Top Metrics Summary Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${latest_data['Close']:.2f}", f"{latest_data['Daily_Return']*100:.2f}%")
col2.metric("GPR Index Level", f"{latest_data['GPR_Index']:.1f}")
col3.metric("3-Month Forecast", prediction_label, delta_color=status_color)
col4.metric("AI Confidence", f"{confidence:.1f}%")

st.write("---")

# 7. Visualizations: Price vs GPR & Feature Importance
tab1, tab2 = st.tabs(["📊 Asset vs Risk Trajectory", "🧠 AI Model Explainability"])

with tab1:
    fig_trajectory = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Sleeker lines, fill beneath the stock price
    fig_trajectory.add_trace(
        go.Scatter(x=merged_data.index, y=merged_data['Close'], name=f'{ticker} Price', 
                   line=dict(color='#00F0FF', width=2), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.05)'),
        secondary_y=False,
    )
    fig_trajectory.add_trace(
        go.Scatter(x=merged_data.index, y=merged_data['GPR_Index'], name='GPR Index', 
                   line=dict(color='#FF3366', dash='dot', width=2)),
        secondary_y=True,
    )
    
    # Transparent backgrounds and clean grids
    fig_trajectory.update_layout(
        height=500, hovermode='x unified', margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_trajectory.update_xaxes(showgrid=False)
    fig_trajectory.update_yaxes(title_text=f"{ticker} Price ($)", showgrid=True, gridcolor='rgba(255,255,255,0.1)', secondary_y=False)
    fig_trajectory.update_yaxes(title_text="Geopolitical Risk Index", showgrid=False, secondary_y=True)
    
    st.plotly_chart(fig_trajectory, use_container_width=True)

with tab2:
    st.markdown("#### Random Forest Feature Weights")
    st.write("Understanding the driving factors behind the current forecast:")
    
    importance_df = pd.DataFrame({
        'Feature': ['Daily Return', '20-Day SMA', '50-Day SMA', 'Geopolitical Risk Index'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_importance = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#00F0FF',
        marker_line_color='#FFFFFF',
        marker_line_width=1,
        opacity=0.8
    ))
    
    fig_importance.update_layout(
        height=350, margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Weight in Decision Trees"
    )
    fig_importance.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig_importance, use_container_width=True)
