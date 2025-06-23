import pandas as pd
import os
import numpy as np

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "predictions_log.csv")
DATA_DRIFT_PATH = os.path.join(DATA_DIR, "predictions_log_drift_data.csv")
PRED_DRIFT_PATH = os.path.join(DATA_DIR, "predictions_log_drift_predictions.csv")

# Load original predictions log
df = pd.read_csv(INPUT_PATH)

# -------- Simulate Data Drift --------
df_data_drift = df.copy()
# Example: amplify AGE and alter RM
df_data_drift["AGE"] = df_data_drift["AGE"] * 1.5
df_data_drift["RM"] = np.random.normal(loc=4.5, scale=0.5, size=len(df_data_drift))
df_data_drift.to_csv(DATA_DRIFT_PATH, index=False)

# -------- Simulate Prediction Drift --------
df_pred_drift = df.copy()
# Example: shift predictions up artificially
df_pred_drift["prediction"] = df_pred_drift["prediction"] + np.random.uniform(5, 10, size=len(df_pred_drift))
df_pred_drift.to_csv(PRED_DRIFT_PATH, index=False)

print("Simulated drift files saved to /data:")
print(f"- {os.path.basename(DATA_DRIFT_PATH)}")
print(f"- {os.path.basename(PRED_DRIFT_PATH)}")