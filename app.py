import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# 1. Page Configuration & Custom CSS for Permanent Sidebar
st.set_page_config(page_title="Geo-Alpha Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Hide Streamlit default branding & header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Lock the sidebar open by hiding the collapse button */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Sleek container styling */
    div[data-testid="metric-container"] {
        background-color: #161622;
        border: 1px solid #2A2A3C;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# 2. Session State Login Gate
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1>🔐 Geo-Alpha Terminal</h1>
            <p style="color: #A0A0B0;">Institutional-Grade Geopolitical Macro Forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Terminal ID", placeholder="e.g., trader_01")
            password = st.text_input("Access Key", type="password", placeholder="••••••••")
            submit = st.form_submit_button("Initialize Terminal Session", use_container_width=True)
            
            if submit:
                if username and password:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Please enter a valid Terminal ID and Access Key.")
    st.stop()

# 3. Main Dashboard (Loads after successful login)
st.title("🌍 Geo-Alpha Intelligence Terminal")
st.markdown("### Real-time macro risk assessment, technical profiling, and machine learning asset forecasting.")
st.write("---")

# 4. Permanent Sidebar Controls
with st.sidebar:
    st.title("🎛️ Terminal Controls")
    st.write("Configure your macro inspection parameters:")
    ticker = st.text_input("Target Ticker", value="SPY").upper()
    benchmark = st.selectbox("Historical Window", ["1y", "2y", "5y", "max"], index=2)
    
    st.markdown("---")
    st.markdown("### 📡 System Telemetry")
    st.markdown("**Engine:** Random Forest (v2.4)")
    st.markdown("**Data Feed:** Yahoo Finance + GPR Index")
    st.markdown("**Status:** <span style='color:#00C853;'>● PERMANENT ONLINE</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("Terminate Session", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# 5. Load Model and Macro Data
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

# 6. Fetch Live Market Data
with st.spinner(f"Establishing secure feed for {ticker}..."):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=benchmark)
        
        if hist.empty:
            st.error(f"Could not aggregate market telemetry for ticker '{ticker}'.")
            st.stop()
            
        hist['Daily_Return'] = hist['Close'].pct_change()
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist = hist.dropna()
        
        hist.index = hist.index.tz_localize(None)
        merged_data = hist.join(gpr_data, how='left')
        merged_data['GPR_Index'] = merged_data['GPR_Index'].ffill().bfill()
        
    except Exception as e:
        st.error(f"Data aggregation fault: {e}")
        st.stop()

# 7. Prediction & Confidence Calculations
features = ['Daily_Return', 'SMA_20', 'SMA_50', 'GPR_Index']
latest_data = merged_data.iloc[-1]
X_latest = latest_data[features].values.reshape(1, -1)

prediction = model.predict(X_latest)[0]
probabilities = model.predict_proba(X_latest)[0]
confidence = probabilities[prediction] * 100

prediction_label = "BULLISH (Upward Momentum)" if prediction == 1 else "BEARISH (Downward Pressure)"
status_color = "normal" if prediction == 1 else "inverse"

# 8. Top Metrics Summary Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Asset Price", f"${latest_data['Close']:.2f}", f"{latest_data['Daily_Return']*100:.2f}%")
col2.metric("Macro Risk Index (GPR)", f"{latest_data['GPR_Index']:.1f}")
col3.metric("3-Month AI Outlook", prediction_label, delta_color=status_color)
col4.metric("Model Confidence Score", f"{confidence:.1f}%")

st.write("---")

# 9. Immersive Editorial Macro Briefing
st.markdown("### 📋 Executive Macro Intelligence Briefing")
if prediction == 1:
    st.success(f"**Market Assessment for {ticker}:** The algorithmic matrix indicates favorable conditions over the next 3 months. Despite lingering global geopolitical frictions reflected in the GPR index, technical price strength (moving averages) is overriding macroeconomic volatility, pointing toward continued capital inflows.")
else:
    st.warning(f"**Market Assessment for {ticker}:** Caution advised. The model detects vulnerability in current price action relative to prevailing geopolitical risk metrics. Defensive positioning or capital preservation is favored over the upcoming 90-day window.")

st.write("")

# 10. Interactive Tabs for Charts & Explainability
tab1, tab2 = st.tabs(["📈 Asset vs. Geopolitical Risk Trajectory", "🔬 Model Attribution & Explainability"])

with tab1:
    fig_trajectory = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_trajectory.add_trace(
        go.Scatter(x=merged_data.index, y=merged_data['Close'], name=f'{ticker} Price', 
                   line=dict(color='#00F0FF', width=2), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.04)'),
        secondary_y=False,
    )
    fig_trajectory.add_trace(
        go.Scatter(x=merged_data.index, y=merged_data['GPR_Index'], name='Geopolitical Risk Index', 
                   line=dict(color='#FF3366', dash='dot', width=2)),
        secondary_y=True,
    )
    
    fig_trajectory.update_layout(
        height=480, hovermode='x unified', margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_trajectory.update_xaxes(showgrid=False)
    fig_trajectory.update_yaxes(title_text=f"{ticker} Valuation ($)", showgrid=True, gridcolor='rgba(255,255,255,0.08)', secondary_y=False)
    fig_trajectory.update_yaxes(title_text="Macro Risk Score", showgrid=False, secondary_y=True)
    
    st.plotly_chart(fig_trajectory, use_container_width=True)

with tab2:
    st.markdown("#### Random Forest Decision Vector Breakdown")
    st.write("The chart below quantifies the weight assigned to each predictive variable during the decision-making process:")
    
    importance_df = pd.DataFrame({
        'Feature': ['Daily Return Velocity', '20-Day Simple Moving Average', '50-Day Simple Moving Average', 'Geopolitical Risk Index (GPR)'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_importance = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#00F0FF',
        marker_line_color='#FFFFFF',
        marker_line_width=1,
        opacity=0.85
    ))
    
    fig_importance.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Relative Decision Weight"
    )
    fig_importance.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
    
    st.plotly_chart(fig_importance, use_container_width=True)
