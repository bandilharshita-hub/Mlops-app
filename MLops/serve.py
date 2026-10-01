import os
import glob
from pathlib import Path
import joblib
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

# Features definition
FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]

# Model path definitions
LOCAL_MODEL_PKL = BASE_DIR / "model.pkl"
LOCAL_MLFLOW_MODEL_DIR = BASE_DIR / "model"

MODEL_URI = "local_model"
model = None

# Search dynamically inside mlartifacts if standard paths don't exist
mlartifacts_path = BASE_DIR / "mlartifacts"
found_mlflow_uri = None
if mlartifacts_path.exists():
    artifact_dirs = glob.glob(str(mlartifacts_path / "**" / "artifacts"), recursive=True)
    if artifact_dirs:
        found_mlflow_uri = artifact_dirs[0]

# Load model locally using fallback options
if LOCAL_MODEL_PKL.exists():
    model = joblib.load(LOCAL_MODEL_PKL)
    MODEL_URI = str(LOCAL_MODEL_PKL)
elif LOCAL_MLFLOW_MODEL_DIR.exists():
    model = mlflow.sklearn.load_model(str(LOCAL_MLFLOW_MODEL_DIR))
    MODEL_URI = str(LOCAL_MLFLOW_MODEL_DIR)
elif found_mlflow_uri:
    model = mlflow.sklearn.load_model(found_mlflow_uri)
    MODEL_URI = found_mlflow_uri
else:
    raise FileNotFoundError(
        "No model file found! Ensure 'model.pkl', 'model/' folder, or 'mlartifacts/' is correctly copied in the container."
    )

app = FastAPI(title="House Price Predictor")

class HouseFeatures(BaseModel):
    sqft: float = Field(..., gt=0, le=20000)
    bedrooms: int = Field(..., gt=0, le=20)
    bathrooms: int = Field(..., gt=0, le=200)
    age_years: int = Field(..., ge=0, le=100)
    garage: int = Field(..., ge=0, le=10)
    location_score: float = Field(..., ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_URI}

@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.model_dump()], columns=FEATURES)
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/")
def frontend():
    return FileResponse(BASE_DIR / "static" / "index.html")