import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Page Configuration
st.set_page_config(page_title="Geo-Alpha Predictor", layout="wide")
st.title("Geopolitical Macro Alpha Predictor")
st.write("A machine learning pipeline forecasting 3-month asset direction based on geopolitical risk.")

# 2. Load Data and Model
@st.cache_data
def load_data():
    return pd.read_csv('month_1_master_dataset.csv', index_col='Date', parse_dates=True)

@st.cache_resource
def load_model():
    return joblib.load('geo_alpha_model.joblib')

try:
    df = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("Missing files. Ensure 'month_1_master_dataset.csv' and 'geo_alpha_model.joblib' are in the same folder.")
    st.stop()

# 3. Process Latest Data for Prediction
features = ['Daily_Return', 'SMA_20', 'SMA_50', 'GPR_Index']
latest_data = df.iloc[-1]
X_latest = latest_data[features].values.reshape(1, -1)

prediction = model.predict(X_latest)[0]
prediction_text = "Higher (Bullish)" if prediction == 1 else "Lower (Bearish)"
status_color = "green" if prediction == 1 else "red"

# 4. Display Forecast Dashboard
st.subheader("Current 3-Month Forecast")
st.markdown(f"**Predicted Asset Direction:** :{status_color}[{prediction_text}]")
st.write(f"Forecast generated using data up to: **{df.index[-1].date()}**")
st.write(f"Current GPR Index Level: **{latest_data['GPR_Index']:.2f}**")

# 5. Visualize Asset Price vs. Geopolitical Risk
st.subheader("Historical Trajectory: Asset vs. Geopolitical Risk")

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=df.index, y=df['Close'], name='Asset Price (Close)', line=dict(color='blue')),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df.index, y=df['GPR_Index'], name='GPR Index', line=dict(color='orange'), opacity=0.6),
    secondary_y=True,
)

fig.update_layout(height=500, hovermode='x unified')
fig.update_yaxes(title_text="Asset Price", secondary_y=False)
fig.update_yaxes(title_text="Geopolitical Risk Index", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)
