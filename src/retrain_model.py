import os
import shutil
from datetime import datetime
import pandas as pd
import joblib
import mlflow
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "data", "BostonHousing.csv")
DRIFT_DATA_PATH = os.path.join(BASE_DIR, "data", "predictions_log_drift_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
original_data = pd.read_csv(ORIGINAL_DATA_PATH)
drift_data = pd.read_csv(DRIFT_DATA_PATH)

# Check for NaNs in drift_data (log only)
print("\n[INFO] Checking for NaN in drift_data:")
print(drift_data.isnull().sum())

# Ensure drift_data does not contain extra columns
drift_data = drift_data.drop(columns=[col for col in ["timestamp", "medv", "prediction"] if col in drift_data.columns])

# Prepare training data
original_features = original_data.drop(columns=["medv"])
original_target = original_data["medv"]

# Synthetic targets for drifted data
synthetic_target = original_target.sample(n=len(drift_data), replace=True, random_state=42).reset_index(drop=True)

# Combine original and drifted data
X = pd.concat([original_features, drift_data], ignore_index=True)
y = pd.concat([original_target, synthetic_target], ignore_index=True)

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
#rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n[INFO] Retrained GradientBoostingRegressor Evaluation:")
print(f"RMSE    : {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Generate timestamped model filename
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_filename = f"gradient_boosting_model_{timestamp}.joblib"
model_path = os.path.join(MODELS_DIR, model_filename)

# Save model artifact
joblib.dump(model, model_path)
print(f"[INFO] Model saved to: {model_path}")

# Update 'latest' pointer (symlink or copy on Windows)
latest_model_path = os.path.join(MODELS_DIR, "gradient_boosting_model_latest.joblib")
try:
    if os.name == "nt":
        shutil.copyfile(model_path, latest_model_path)
    else:
        if os.path.islink(latest_model_path):
            os.remove(latest_model_path)
        os.symlink(model_path, latest_model_path)
    print(f"[INFO] Updated latest model pointer: {latest_model_path}")
except Exception as e:
    print(f"[WARN] Could not update latest model symlink: {e}")

# Register run in MLflow (local tracking)
mlflow.set_experiment("gradient_boosting_housing")

with mlflow.start_run(run_name=f"retrain_{timestamp}") as run:
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    mlflow.log_artifact(model_path, artifact_path="model")

print("[INFO] MLflow tracking completed.")
