import os
import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_FEATURES_PATH = os.path.join(BASE_DIR, "data", "BostonHousing.csv")
CURRENT_FEATURES_PATH = os.path.join(BASE_DIR, "data", "predictions_log_drift_data.csv")
CURRENT_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "predictions_log_drift_predictions.csv")
OUTPUT_REPORT_PATH = os.path.join(BASE_DIR, "data", "evidently_drift_report.html")

# Load datasets
reference_data = pd.read_csv(REFERENCE_FEATURES_PATH)
current_data = pd.read_csv(CURRENT_FEATURES_PATH)
current_preds = pd.read_csv(CURRENT_PREDICTIONS_PATH)

# Drop columns not present in the reference data
for col in ["timestamp", "prediction"]:
    if col in current_data.columns:
        current_data = current_data.drop(columns=[col])

# Ensure reference_data contains only input features
reference_data = reference_data.drop(columns=["medv"])

# Ensure current_data contains only same input features (exclude prediction)
current_data = current_data[reference_data.columns]

# Run and save the report
report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=reference_data, current_data=current_data)

# Export to HTML
result.save_html(OUTPUT_REPORT_PATH)
print(f"Drift report saved at: {OUTPUT_REPORT_PATH}")

# Parse JSON snapshot to detect drift
data_dict = json.loads(result.json())
drift_info = next(
    (m for m in data_dict.get("metrics", [])
     if m.get("type") == "DataDriftTable"),
    None
)

drift_detected = False
if drift_info:
    n_drifted = drift_info["result"].get("number_of_drifted_features", 0)
    drift_detected = n_drifted > 0
    print(f"Drifted features: {n_drifted}")

print(f"Drift detected? {drift_detected}")

# Trigger retraining if drift is detected
if drift_detected:
    print("Drift detected — triggering retraining...")
    exit_code = os.system("python src/retrain_model.py")
    if exit_code == 0:
        print("Retraining completed successfully.")
    else:
        print("Retraining process failed.")
else:
    print("No drift detected — model is up to date.")



