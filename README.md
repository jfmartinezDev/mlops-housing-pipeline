# Housing Price Prediction API

This project implements a complete ML pipeline that trains a regression model on the Boston Housing dataset and exposes it via a FastAPI-based RESTful API. The pipeline includes data loading, model training, saving, logging with MLflow, containerization with Docker, local testing and cloud deployment in Azure.

## Project Structure

```
mlops-housing-pipeline/
├── api/
│   ├── main.py              # FastAPI app
│   └── test_main.py         # pytest test for the API
├── data/
│   └── BostonHousing.csv    # Dataset used for training
├── models/
│   └── gradient_boosting_model.joblib  # Saved model
├── Dockerfile               # Docker image definition
├── .dockerignore            # Docker ignore config
├── requirements.txt         # Project dependencies
├── src/
│   └── train.py             # Script to train and save the model
├── mlruns/                  # MLflow experiment logs
└── README.md                # This file
```

## Requirements

Install the dependencies in a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies (from `requirements.txt`):

```
fastapi
uvicorn
numpy
pandas
scikit-learn
pydantic
mlflow
joblib
httpx
pytest
```

Ensure `mlflow` is properly configured with local backend and artifact store if you plan to explore MLflow UI:

```bash
mlflow ui
```

## Train the Model

To train the model and generate `gradient_boosting_model.joblib`:

```bash
python src/train.py
```

This script:

* Loads the dataset from `data/BostonHousing.csv`
* Splits it into train/test sets
* Trains a `GradientBoostingRegressor`
* Evaluates MAE and R²
* Saves the trained model into `models/`
* Logs parameters, metrics, and the model using MLflow

## Run the API Locally

Start the FastAPI server locally with:

```bash
cd api
uvicorn api.main:app --reload
```

Access Swagger UI:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Test the API

You can test the API with `curl` or `pytest`:

### 1. Using cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"crim": 0.2, "zn": 0, "indus": 8.14, "chas": 0, "nox": 0.538, "rm": 5.56, "age": 85.4, "dis": 2.45, "rad": 4, "tax": 307, "ptratio": 21.0, "b": 396.9, "lstat": 14.1}'
```

### 2. Using pytest

Make sure you're in the root directory and run:

```bash
pytest api/test_main.py
```

## Build and Run with Docker

### Step 1: Build the Docker image

```bash
docker build -t housing-price-predictor-api .
```

### Step 2: Run the Docker container

```bash
docker run -d -p 8000:8000 --name housing-api housing-price-predictor-api uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Access the API at [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 3: Stop and remove the container

```bash
docker stop housing-api
docker rm housing-api
```

## Deployment to Azure App Service

To deploy on Azure App Service:

1. Push the Docker image to DockerHub or connect your GitHub repository.
2. In Azure, set the startup command as:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

3. Ensure ports, environment variables and paths are properly configured.

## Monitoring

This project uses MLflow to log training metrics and models. To view them:

```bash
mlflow ui
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.


## License

This project is part of a technical assessment and is not intended for production use.