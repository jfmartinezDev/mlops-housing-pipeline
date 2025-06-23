from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Boston Housing Price Prediction API")

# Define request body using Pydantic
class HousingFeatures(BaseModel):
    # Define the 13 features used in the Boston Housing dataset
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

# Get absolute path to the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gradient_boosting_model.joblib")
LOG_PATH = os.path.join(BASE_DIR, "data", "predictions_log.csv")

# Load the trained model
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Boston Housing Price Prediction API is running."}

@app.post("/predict")
def predict_price(features: HousingFeatures):
    # Convert features to NumPy array for model input
    input_data = np.array([[ 
        features.CRIM, features.ZN, features.INDUS, features.CHAS, features.NOX,
        features.RM, features.AGE, features.DIS, features.RAD, features.TAX,
        features.PTRATIO, features.B, features.LSTAT
    ]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Create log record
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **features.dict(),
        "prediction": prediction
    }

    # Convert to DataFrame
    log_df = pd.DataFrame([record])

    # Append to CSV or create it
    if not os.path.isfile(LOG_PATH):
        log_df.to_csv(LOG_PATH, index=False)
    else:
        log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)

    return {"predicted_price": round(prediction, 2)}
