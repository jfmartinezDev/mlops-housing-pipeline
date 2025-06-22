from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={
        "CRIM": 0.1, "ZN": 25.0, "INDUS": 5.0, "CHAS": 0, "NOX": 0.5,
        "RM": 6.0, "AGE": 60.0, "DIS": 4.0, "RAD": 4.0, "TAX": 300.0,
        "PTRATIO": 15.0, "B": 390.0, "LSTAT": 5.0
    })
    assert response.status_code == 200
    assert "predicted_price" in response.json()
