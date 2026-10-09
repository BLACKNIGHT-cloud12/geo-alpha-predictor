import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# 1. Page Configuration
st.set_page_config(page_title="Dynamic Geo-Alpha Predictor", layout="wide")
st.title("Interactive Geopolitical Alpha Predictor")
st.write("Analyze how geopolitical risk impacts different stocks and forecast their 3-month trajectory.")

# 2. Sidebar for User Input
st.sidebar.header("Dashboard Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., SPY, AAPL, TSLA, NVDA)", value="SPY").upper()

# 3. Load Static Model and Macro Data (GPR)
@st.cache_resource
def load_model():
    return joblib.load('geo_alpha_model.joblib')

@st.cache_data
def load_macro_data():
    # Load just the GPR index from your master dataset
    df = pd.read_csv('month_1_master_dataset.csv', index_col='Date', parse_dates=True)
    return df[['GPR_Index']]

try:
    model = load_model()
    gpr_data = load_macro_data()
except FileNotFoundError:
    st.error("Missing files. Ensure 'month_1_master_dataset.csv' and 'geo_alpha_model.joblib' are uploaded.")
    st.stop()

# 4. Fetch Live Stock Data
with st.spinner(f"Fetching live market data for {ticker}..."):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        
        if hist.empty:
            st.error(f"Could not find data for ticker '{ticker}'. Please try a valid Yahoo Finance ticker.")
            st.stop()
            
        # Calculate technical features on the fly
        hist['Daily_Return'] = hist['Close'].pct_change()
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist = hist.dropna()
        
        # Make dates timezone naive to match your GPR data
        hist.index = hist.index.tz_localize(None)
        
        # Merge live stock data with static GPR Data
        merged_data = hist.join(gpr_data, how='left')
        merged_data['GPR_Index'] = merged_data['GPR_Index'].ffill().bfill() # Fill missing daily risk values
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# 5. Process Latest Data for Prediction
features = ['Daily_Return', 'SMA_20', 'SMA_50', 'GPR_Index']
latest_data = merged_data.iloc[-1]
X_latest = latest_data[features].values.reshape(1, -1)

prediction = model.predict(X_latest)[0]
prediction_text = "Higher (Bullish)" if prediction == 1 else "Lower (Bearish)"
status_color = "green" if prediction == 1 else "red"

# 6. Display Forecast Dashboard
st.subheader(f"Current 3-Month Forecast: {ticker}")
st.markdown(f"**Predicted Asset Direction:** :{status_color}[{prediction_text}]")
st.write(f"Forecast generated using market data up to: **{merged_data.index[-1].date()}**")
st.write(f"Current GPR Index Level: **{latest_data['GPR_Index']:.2f}**")
st.write(f"Current Price: **${latest_data['Close']:.2f}**")

# 7. Visualize Asset Price vs. Geopolitical Risk
st.subheader(f"Historical Trajectory: {ticker} vs. Geopolitical Risk")

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=merged_data.index, y=merged_data['Close'], name=f'{ticker} Price', line=dict(color='blue')),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=merged_data.index, y=merged_data['GPR_Index'], name='GPR Index', line=dict(color='orange'), opacity=0.6),
    secondary_y=True,
)

fig.update_layout(height=500, hovermode='x unified')
fig.update_yaxes(title_text=f"{ticker} Price ($)", secondary_y=False)
fig.update_yaxes(title_text="Geopolitical Risk Index", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)
