import pandas as pd
import os
import numpy as np

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "predictions_log.csv")
DATA_DRIFT_PATH = os.path.join(DATA_DIR, "predictions_log_drift_data.csv")
PRED_DRIFT_PATH = os.path.join(DATA_DIR, "predictions_log_drift_predictions.csv")

# Load original log data
df = pd.read_csv(INPUT_PATH)

# Simulate data drift (NO DRIFT)
"""df_data_drift = df.copy()
df_data_drift["age"] = df_data_drift["age"] * 1.5
df_data_drift["rm"] = np.random.normal(loc=5, scale=0.5, size=len(df_data_drift))
df_data_drift.drop(columns=["timestamp"], inplace=True)
df_data_drift.to_csv(DATA_DRIFT_PATH, index=False)
"""

# Simulate data drift (DRIFT)
df_data_drift = df.copy()

# Alter numeric features with extreme values
df_data_drift["age"] = np.random.uniform(100, 200, size=len(df_data_drift))  # well beyond original range
df_data_drift["rm"] = np.random.normal(loc=12, scale=1.5, size=len(df_data_drift))  # shift mean + high var
df_data_drift["lstat"] = np.random.beta(2, 5, size=len(df_data_drift)) * 100  # new distribution
df_data_drift["crim"] = np.random.exponential(scale=20, size=len(df_data_drift))  # exponential noise
df_data_drift["nox"] = np.random.uniform(0.8, 1.0, size=len(df_data_drift))  # force right edge
df_data_drift["dis"] = np.random.normal(loc=0.5, scale=0.2, size=len(df_data_drift))  # compact left shift

# Drop non-feature columns
df_data_drift.drop(columns=["timestamp"], inplace=True)

# Save drifted features
df_data_drift.to_csv(DATA_DRIFT_PATH, index=False)

# Simulate prediction drift
df_pred_drift = df.copy()
df_pred_drift["prediction"] = df_pred_drift["prediction"] + np.random.uniform(8, 12, size=len(df_pred_drift))
df_pred_drift = df_pred_drift[["prediction"]]

# Save drifted predictions
df_pred_drift.to_csv(PRED_DRIFT_PATH, index=False)

print("Simulated drift files saved to /data")
