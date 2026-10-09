import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("1. Loading dataset...")
df = pd.read_csv('month_1_master_dataset.csv', index_col='Date', parse_dates=True)

print("2. Creating the 3-month predictive target...")
# 63 trading days is approximately 3 months
forecast_out = 63 

# Target: 1 if future price is higher, 0 if lower
df['Future_Close'] = df['Close'].shift(-forecast_out)
df['Target'] = np.where(df['Future_Close'] > df['Close'], 1, 0)

# Drop the last 63 days since their future outcome is still unknown
df.dropna(subset=['Future_Close'], inplace=True)

# Define features
features = ['Daily_Return', 'SMA_20', 'SMA_50', 'GPR_Index']
X = df[features]
y = df['Target']

print("3. Performing strict chronological train/test split...")
# 80% training data, 20% testing data
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("4. Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to disk for the Streamlit app
model_filename = 'geo_alpha_model.joblib'
joblib.dump(model, model_filename)
print(f"\nSuccess! Model saved as '{model_filename}'")