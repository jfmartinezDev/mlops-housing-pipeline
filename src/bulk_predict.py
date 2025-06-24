import requests
import pandas as pd
import time
import random

df = pd.read_csv("data/BostonHousing.csv")

API_URL = "http://127.0.0.1:8000/predict"

sample_df = df.sample(n=100, random_state=42)

features_df = sample_df.drop(columns=["medv"])

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

    time.sleep(random.uniform(0.3, 0.7))
