import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# 1. Page Configuration
st.set_page_config(page_title="Dynamic Geo-Alpha Predictor", layout="wide")
st.title("Interactive Geopolitical Alpha Predictor")
st.write("Analyze how geopolitical risk impacts equities and inspect machine-learning forecasts.")

# 2. Sidebar Controls
st.sidebar.header("Dashboard Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., SPY, AAPL, TSLA, NVDA)", value="SPY").upper()
benchmark = st.sidebar.selectbox("Historical Window", ["1y", "2y", "5y", "max"], index=2)

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
with st.spinner(f"Fetching live market data for {ticker}..."):
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
col3.metric("3-Month Forecast", prediction_label)
col4.metric("Model Confidence", f"{confidence:.1f}%")

st.divider()

# 7. Visualizations: Price vs GPR & Feature Importance
tab1, tab2 = st.tabs(["Asset vs Risk Trajectory", "Model Explainability"])

with tab1:
    fig_trajectory = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trajectory.add_trace(
        go.Scatter(x=merged_data.index, y=merged_data['Close'], name=f'{ticker} Price', line=dict(color='#2962FF')),
        secondary_y=False,
    )
    fig_trajectory.add_trace(
        go.Scatter(x=merged_data.index, y=merged_data['GPR_Index'], name='GPR Index', line=dict(color='#FF6D00', dash='dot')),
        secondary_y=True,
    )
    fig_trajectory.update_layout(height=480, hovermode='x unified', margin=dict(l=20, r=20, t=20, b=20))
    fig_trajectory.update_yaxes(title_text=f"{ticker} Price ($)", secondary_y=False)
    fig_trajectory.update_yaxes(title_text="Geopolitical Risk Index", secondary_y=True)
    st.plotly_chart(fig_trajectory, use_container_width=True)

with tab2:
    st.markdown("### Random Forest Feature Importance")
    st.write("Relative weight each indicator holds across all decision trees:")
    
    importance_df = pd.DataFrame({
        'Feature': ['Daily Return', '20-Day SMA', '50-Day SMA', 'Geopolitical Risk Index'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_importance = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#00C853'
    ))
    fig_importance.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Weight")
    st.plotly_chart(fig_importance, use_container_width=True)
