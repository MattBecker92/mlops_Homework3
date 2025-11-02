
import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import random

# ---- Configuration ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load initial model
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(model_uri)

# Track current model and version
current_model = model
current_model_version = MODEL_VERSION

# ---- Pydantic Schemas ----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5}
                    ]
                }
            ]
        }
    }

IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]
    class_label: List[str]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1], "class_label": ["setosa", "versicolor"]}
            ]
        }
    }

# ---- FastAPI App ----
app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species and manage model versions.",
    version="1.0.0",
)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "model_uri": f"models:/{MODEL_NAME}/{current_model_version}"}

@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    df = pd.DataFrame([sample.dict() for sample in req.samples])
    preds = current_model.predict(df)
    class_ids = [int(p) for p in preds]
    class_labels = [IRIS_LABELS[i] for i in class_ids]
    return PredictResponse(class_id=class_ids, class_label=class_labels)

@app.post("/set-version", tags=["model"])
def set_version(version: str):
    global current_model, current_model_version
    try:
        new_uri = f"models:/{MODEL_NAME}/{version}"
        new_model = mlflow.pyfunc.load_model(new_uri)
        current_model = new_model
        current_model_version = version
        return {"message": f"Model version {version} is now being served."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model version {version}: {str(e)}")

@app.get("/current-version", tags=["model"])
def current_version():
    return {"current_model_version": current_model_version}

@app.get("/generate-and-predict", tags=["testing"])
def generate_and_predict(n: int = 5):
    samples = []
    for _ in range(n):
        sample = {
            "sepal_length": round(random.uniform(4.3, 7.9), 1),
            "sepal_width": round(random.uniform(2.0, 4.4), 1),
            "petal_length": round(random.uniform(1.0, 6.9), 1),
            "petal_width": round(random.uniform(0.1, 2.5), 1)
        }
        samples.append(sample)

    df = pd.DataFrame(samples)
    preds = current_model.predict(df)
    class_ids = [int(p) for p in preds]
    class_labels = [IRIS_LABELS[i] for i in class_ids]

    return {
        "samples": samples,
        "predictions": {
            "class_id": class_ids,
            "class_label": class_labels
        }
    }
