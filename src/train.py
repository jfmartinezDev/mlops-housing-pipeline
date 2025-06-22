import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import mlflow
import mlflow.sklearn

# Start an MLflow experiment run
with mlflow.start_run(run_name="GradientBoosting_BostonHousing"):

    # Load dataset
    df = pd.read_csv("data/BostonHousing.csv")

    # Separate features and target
    X = df.drop("medv", axis=1)
    y = df["medv"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define and train model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # Log parameters and metrics to MLflow
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Save and log the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/gradient_boosting_model.joblib"
    joblib.dump(model, model_path)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model saved locally to: {model_path}")

