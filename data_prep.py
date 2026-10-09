import pandas as pd
import yfinance as yf
import requests
import io
import warnings
warnings.filterwarnings('ignore')

# --- Step 1: The Target Asset Data ---
print("Downloading SPY data...")
df_asset = yf.download('SPY', start='2010-01-01')
if isinstance(df_asset.columns, pd.MultiIndex):
    df_asset.columns = df_asset.columns.get_level_values(0)

df_asset['Daily_Return'] = df_asset['Close'].pct_change()
df_asset['SMA_20'] = df_asset['Close'].rolling(window=20).mean()
df_asset['SMA_50'] = df_asset['Close'].rolling(window=50).mean()
df_asset.dropna(inplace=True)
df_asset.index = pd.to_datetime(df_asset.index).tz_localize(None)

# --- Step 2: The Geopolitical Signal ---
print("Downloading GPR data...")
url = 'https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
df_gpr_raw = pd.read_excel(io.BytesIO(response.content))

df_gpr = df_gpr_raw[['month', 'GPR']].copy()
df_gpr.rename(columns={'month': 'Date', 'GPR': 'GPR_Index'}, inplace=True)
df_gpr['Date'] = pd.to_datetime(df_gpr['Date'])
df_gpr.set_index('Date', inplace=True)
df_gpr = df_gpr[df_gpr.index >= '2009-11-01']

# --- Step 3: Merging & Exporting ---
print("Aligning time-series data...")
df_gpr.index = df_gpr.index + pd.offsets.MonthBegin(1)
df_gpr_daily = df_gpr.resample('D').ffill()

master_df = df_asset.join(df_gpr_daily, how='left')
master_df['GPR_Index'] = master_df['GPR_Index'].ffill()
master_df.dropna(subset=['GPR_Index'], inplace=True)

master_df.to_csv('month_1_master_dataset.csv')
print("\nSuccess! 'month_1_master_dataset.csv' has been created in your folder.")