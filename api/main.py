import os
import joblib
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np

app = FastAPI()

class HousingFeatures(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int
    nox: float
    rm: float
    age: float
    dis: float
    rad: int
    tax: float
    ptratio: float
    b: float
    lstat: float

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gradient_boosting_model_latest.joblib")
LOG_PATH = os.path.join(BASE_DIR, "data", "predictions_log.csv")

# Print diagnostic info
print(f"[*] BASE_DIR: {BASE_DIR}")
print(f"[*] MODEL_PATH: {MODEL_PATH}")
print(f"[*] LOG_PATH: {LOG_PATH}")

# Load model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("[+] Model loaded successfully.")
except Exception as e:
    print(f"[!] Error loading model:\n{traceback.format_exc()}")
    model = None

# Root healthcheck endpoint
@app.get("/")
def root():
    return {"message": "Boston Housing Price Prediction API is running."}

# Prediction endpoint
@app.post("/predict")
def predict_price(features: HousingFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available.")

    try:
        input_data = np.array([
            features.crim, features.zn, features.indus, features.chas,
            features.nox, features.rm, features.age, features.dis,
            features.rad, features.tax, features.ptratio, features.b,
            features.lstat
        ]).reshape(1, -1)

        print(f"[INFO] Input received: {input_data.tolist()}")

        prediction = model.predict(input_data)[0]
        print(f"[INFO] Prediction: {prediction}")

        # Logging to CSV
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            **features.dict(),
            "prediction": prediction
        }

        log_df = pd.DataFrame([record])
        if os.path.isfile(LOG_PATH):
            log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
        else:
            log_df.to_csv(LOG_PATH, index=False)

        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        print(f"[!] Prediction error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

