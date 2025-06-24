# Housing Price Prediction - MLOps Pipeline

This repository implements a complete MLOps pipeline for predicting housing prices using the Boston Housing dataset. It covers model training, API deployment (locally, Docker, and Azure App Service), monitoring with MLflow and Evidently, drift detection, and automated retraining logic.

---

## 📁 Project Structure

```
mlops-housing-pipeline/
├── api/
│   ├── main.py                  # FastAPI API entry point
│   └── test_main.py             # Pytest unit tests
├── data/
│   ├── BostonHousing.csv        # Training dataset
│   ├── predictions_log.csv      # Inference logs
│   ├── predictions_log_drift_data.csv
│   ├── predictions_log_drift_predictions.csv
│   └── evidently_drift_report.html
├── models/
│   ├── gradient_boosting_model.joblib
│   ├── gradient_boosting_model_latest.joblib
│   └── gradient_boosting_model_YYYYMMDD_HHMMSS.joblib
├── src/
│   ├── train.py                 # Train the model
│   ├── bulk_predict.py          # Run batch predictions
│   ├── detect_drift.py          # Drift detection using Evidently
│   ├── simulate_drift.py        # Generate synthetic drift data
│   ├── retrain_model.py         # Retrain model automatically
│   └── download_dataset.py      # Download dataset (if required)
├── mlruns/                      # MLflow logs and artifacts
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── README.md
```

---

## ✅ Requirements

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate         # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 Key Dependencies

The project relies on the following main Python packages:

| Package        | Version   | Description                                      |
|----------------|-----------|--------------------------------------------------|
| fastapi        | 0.115.13  | Web framework for building the REST API         |
| joblib         | 1.5.1     | Model serialization and persistence             |
| mlflow         | 3.1.0     | End-to-end machine learning lifecycle tracking  |
| mlflow_skinny  | 3.1.0     | Lightweight MLflow client for deployment        |
| numpy          | 2.3.1     | Numerical computations and array operations     |
| pandas         | 2.3.0     | Data manipulation and preprocessing             |
| pydantic       | 2.11.7    | Data validation and parsing for FastAPI models  |
| requests       | 2.32.4    | HTTP requests handling                          |
| scikit-learn   | 1.7.0     | Model training and evaluation                   |
| evidently      | 0.7.8     | Model monitoring and drift detection            |


---

## 🧠 Train the Model

```bash
python src/train.py
```

This will:

- Load `BostonHousing.csv`
- Train and evaluate a `GradientBoostingRegressor`
- Log artifacts with MLflow
- Save models to `models/`

---

## 🚀 Run the API Locally

```bash
uvicorn api.main:app --reload
```

Visit Swagger UI at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Run API Tests

```bash
pytest api/test_main.py
```

---

## 🐳 Run with Docker

### 1. Build Docker Image

```bash
docker build -t housing-price-predictor-api .
```

### 2. Run the Container

```bash
docker run -d -p 8000:8000 --name housing-api \
  housing-price-predictor-api \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Stop and Remove

```bash
docker stop housing-api
docker rm housing-api
```

---

## ☁️ Deployment on Azure App Service

### Steps:

1. Push the image to Docker Hub:
   ```bash
   docker tag housing-price-predictor-api jfma8925/housing-price-predictor-api:latest
   docker push jfma8925/housing-price-predictor-api:latest
   ```

2. In Azure Portal > App Service > Deployment Center:
   - **Container Registry**: Docker Hub
   - **Image name**: `jfma8925/housing-price-predictor-api:latest`
   - **Startup command**:
     ```bash
     uvicorn api.main:app --host 0.0.0.0 --port 8000
     ```

---

## ☸️ Deployment on Azure Kubernetes Service (AKS)

> For production-grade deployments

1. Push Docker image (same as above)
2. Create a Kubernetes deployment YAML:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: housing-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: housing-api
  template:
    metadata:
      labels:
        app: housing-api
    spec:
      containers:
      - name: housing-api
        image: jfma8925/housing-price-predictor-api:latest
        ports:
        - containerPort: 8000
```

3. Apply it:

```bash
kubectl apply -f deployment.yaml
kubectl expose deployment housing-api --type=LoadBalancer --port=80 --target-port=8000
```

---

## 📊 Monitoring and Drift Detection

### Launch MLflow UI:

```bash
mlflow ui
```

Visit: [http://localhost:5000](http://localhost:5000)

### Run Drift Detection:

```bash
python src/detect_drift.py
```

Drift report saved in: `data/evidently_drift_report.html`

---

## 🔁 Retraining Strategy

Retrain if drift is detected or periodically (weekly/monthly).

### To Retrain:

```bash
python src/retrain_model.py
```

This will:
- Load updated dataset
- Train new model
- Replace `gradient_boosting_model_latest.joblib`

You can also automate this via CI/CD or a scheduled Azure Function.

---

## 📎 License

This project is part of a technical assessment and not intended for production.