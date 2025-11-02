# MLOps Homework 3: Airflow + MLflow + FastAPI

## 📖 Project Overview
This project demonstrates an end-to-end MLOps pipeline using:
- **Airflow** for orchestrating model training workflows.
- **MLflow** for experiment tracking and model registry.
- **FastAPI** for serving predictions and managing model versions dynamically.

The system trains multiple models, registers them in MLflow, and exposes REST API endpoints for prediction and version control.

---

🚀 Setup Instructions

1. Clone the Repository
```
git clone git@github.com:MattBecker92/mlops_Homework3.git
cd mlops-homework3
```
2. Create and Activate Virtual Environment
```
python3 -m venv venvsource venv/bin/activateShow more lines
```
3. Install Dependencies
```
pip install -r requirements.txt
```

---
⚙️ Components
Airflow

DAG: dags/train_model.py
Trains models and logs them to MLflow.

MLflow

Tracks experiments and stores models.
Accessible at http://localhost:5000.

FastAPI

Serves predictions using MLflow models.
Endpoints:

GET /health – Check API health and current model URI.
POST /predict – Predict Iris species from input samples.
POST /set-version – Switch the model version being served.
GET /current-version – View the currently active model version.
GET /generate-and-predict – Generate random test data and return predictions.

---
▶️ Running the API
Start the FastAPI server:
```
./fast_api.sh
```
Access Swagger UI at:
```
http://localhost:8000/docs
```
---

🖼 System Diagram
!System Diagram
Explanation:

Airflow DAG triggers training and logs models to MLflow.
MLflow stores models and provides version control.
FastAPI loads models from MLflow and serves predictions to users.
Users interact with the API via HTTP requests.

---

✅ Example Usage
Predict Endpoint:
```

curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "samples": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
  ]
}'
```
