"""
FastAPI app that serves the trained churn model.

Run:
    uvicorn app:app --reload
Docs:
    http://127.0.0.1:8000/docs
"""
import json
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

import config

# ---- Request schema, built from config.py so it always matches training ----
_fields = {name: (Optional[float], None) for name in config.NUMERIC_FEATURES}
_fields.update({name: (Optional[str], None) for name in config.CATEGORICAL_FEATURES})

CustomerFeatures = create_model(
    "CustomerFeatures",
    __doc__="Customer attributes. Missing values are allowed and imputed.",
    **_fields,
)

CustomerFeatures.model_config["json_schema_extra"] = {
    "example": {
        "tenure_months": 5,
        "monthly_charges": 85.5,
        "total_charges": 427.5,
        "num_support_tickets": 3,
        "avg_monthly_usage_gb": 60.0,
        "late_payments": 2,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check",
        "internet_service": "fiber",
        "has_tech_support": "no",
        "paperless_billing": "yes",
    }
}


class Prediction(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    risk_level: str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


# ---- Model loading ---------------------------------------------------------
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {config.MODEL_PATH}. Run `python train.py` first."
        )
    state["model"] = joblib.load(config.MODEL_PATH)
    state["metadata"] = json.loads(config.METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(
    title="Customer Churn API",
    description="Predicts the probability that a customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)


def _risk_level(p: float) -> str:
    for cutoff, label in config.RISK_BANDS:
        if p >= cutoff:
            return label
    return "low"


def _predict(records: list[dict]) -> list[Prediction]:
    df = pd.DataFrame(records, columns=config.FEATURES)
    # Pydantic gives None for missing values; convert to NaN for the imputers.
    df[config.NUMERIC_FEATURES] = df[config.NUMERIC_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    df[config.CATEGORICAL_FEATURES] = df[config.CATEGORICAL_FEATURES].astype(object).fillna(float("nan"))
    try:
        probs = state["model"].predict_proba(df)[:, 1]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")
    threshold = state["metadata"]["threshold"]
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            churn_prediction=bool(p >= threshold),
            risk_level=_risk_level(float(p)),
        )
        for p in probs
    ]


# ---- Endpoints ------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    return state["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(customer: CustomerFeatures):
    return _predict([customer.model_dump()])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    return BatchResponse(predictions=_predict([c.model_dump() for c in request.customers]))
