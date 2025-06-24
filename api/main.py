from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = FastAPI(title="Boston Housing Price Prediction API")

# Define the input features using lowercase names
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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gradient_boosting_model_latest.joblib")
LOG_PATH = os.path.join(BASE_DIR, "data", "predictions_log.csv")

model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Boston Housing Price Prediction API is running."}

@app.post("/predict")
def predict_price(features: HousingFeatures):
    input_data = np.array([
        features.crim, features.zn, features.indus, features.chas,
        features.nox, features.rm, features.age, features.dis,
        features.rad, features.tax, features.ptratio, features.b,
        features.lstat
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]

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
