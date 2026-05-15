from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import RiskInput, PredictionResponse
from api.ml_service import predict_risk

app = FastAPI(
    title="Risk Profiler API",
    description="AI-powered credit scoring risk calculator for UMKM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Risk Profiler API is running"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: RiskInput):
    result = predict_risk(data)
    return result