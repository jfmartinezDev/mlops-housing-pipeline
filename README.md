# Housing Price Prediction - MLOps Pipeline

This repository implements a complete **MLOps pipeline** for predicting housing prices using the **Boston Housing Dataset**. It covers training, local/Docker/Azure API deployment, ML monitoring (MLflow + Evidently), automated drift detection and retraining, version control, and prediction logging.

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

## ✅ Requirements

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate         # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 Key Dependencies

| Package        | Version   | Description                                      |
|----------------|-----------|--------------------------------------------------|
| fastapi        | 0.115.13  | Web framework for building the REST API         |
| joblib         | 1.5.1     | Model serialization and persistence             |
| mlflow         | 3.1.0     | ML lifecycle tracking and experiment logging    |
| evidently      | 0.7.8     | Model monitoring and drift detection            |
| numpy          | 2.3.1     | Numerical computations                          |
| pandas         | 2.3.0     | Data manipulation and preprocessing             |
| scikit-learn   | 1.7.0     | Model training and evaluation                   |
| pydantic       | 2.11.7    | Data validation in FastAPI                      |

## 🧠 Train the Model

```bash
python src/train.py
```

- Loads `BostonHousing.csv`
- Trains and evaluates a `GradientBoostingRegressor`
- Saves models to `models/`
- Logs to MLflow

## 🚀 Run the API Locally

```bash
uvicorn api.main:app --reload
```

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 🧪 Run API Tests

```bash
pytest api/test_main.py
```

## 🐳 Run with Docker

### 1. Build Docker Image

```bash
docker build -t housing-price-predictor-api .
```

### 2. Run the Container

```bash
docker run -d -p 8000:8000 --name housing-api   housing-price-predictor-api   uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Stop and Remove

```bash
docker stop housing-api
docker rm housing-api
```

## Deployment on Azure App Service

### 1. Push to Docker Hub:

```bash
docker tag housing-price-predictor-api <your-dockerhub-username>/housing-price-predictor-api:latest
docker push <your-dockerhub-username>/housing-price-predictor-api:latest
```

### 2. Azure Portal Configuration

- **Container Registry**: Docker Hub
- **Image Name**: `<your-dockerhub-username>/housing-price-predictor-api:latest`
- **Startup Command**:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Restart App Service after update


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
        image: <your-dockerhub-username>/housing-price-predictor-api:latest
        ports:
        - containerPort: 8000
```

3. Apply it:

```bash
kubectl apply -f deployment.yaml
kubectl expose deployment housing-api --type=LoadBalancer --port=80 --target-port=8000
```

## 📊 Monitoring and Drift Detection

### Run MLflow UI:

```bash
mlflow ui
```

Access: [http://localhost:5000](http://localhost:5000)

### Run Drift Detection:

```bash
python src/detect_drift.py
```

Output: `data/evidently_drift_report.html`

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
- Saves timestamped backup model
- Updates Docker image if needed

You can also automate this via CI/CD or a scheduled Azure Function.

---


## 🧾 Inference Logging

- API logs every prediction to `data/predictions_log.csv` with timestamp, inputs and prediction.
- Useful for drift detection and auditing.

## 🧯 Troubleshooting

- ❗ **500 Internal Server Error on Azure?**
  - Ensure the model file exists in `/app/models/`
  - Check correct filename: `gradient_boosting_model_latest.joblib`
  - Restart the App Service from Azure Portal

- ❗ **Model Input Shape Error?**
  - Ensure you rebuilt the model after feature changes:
  ```bash
  python src/train.py
  ```

- ❗ **Docker Container Won’t Start?**
  - Run logs: `docker logs housing-api`
  - Ensure path to model is correct in `main.py`

## License

This project is part of a technical assessment and not intended for production.

## 👨‍💻 Author

**José Francisco Martínez Amaya**  
- Docker Hub: [jfma8925](https://hub.docker.com/repository/docker/jfma8925/housing-price-predictor-api/)  
- GitHub: [github.com/jfma8925](https://github.com/jfmartinezDev/mlops-housing-pipeline)