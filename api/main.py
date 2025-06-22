from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

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

# Load the trained model
model = joblib.load("models/gradient_boosting_model.joblib")

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

    return {"predicted_price": round(prediction, 2)}
