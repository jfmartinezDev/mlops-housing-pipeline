import requests
import pandas as pd
import time
import random

# Load dataset with real housing data
df = pd.read_csv("data/BostonHousing.csv")

# Define API endpoint (local)
API_URL = "http://127.0.0.1:8000/predict"

# Select a random sample of 100 rows to simulate predictions
sample_df = df.sample(n=100, random_state=42)

# Rename columns to match the Pydantic model expected by the API
sample_df.columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "medv"
]

# Drop the target column since it is not required for prediction
features_df = sample_df.drop(columns=["medv"])

# Loop through each row and send a POST request to the API
for i, row in features_df.iterrows():
    payload = row.to_dict()
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            print(f"Prediction {i + 1}: {response.json()}")
        else:
            print(f"Error {i + 1}: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

    # Wait between requests to avoid overloading the API
    time.sleep(random.uniform(0.3, 0.7))
