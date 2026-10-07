"""
Phishing URL Detector - FastAPI backend.

Run locally from the backend/ folder:
    uvicorn main:app --reload

Environment variables:
    ALLOWED_ORIGINS   Comma-separated list of allowed frontend origins.
                      Default: http://localhost:5173,http://127.0.0.1:5173
                      Example (production):
                      ALLOWED_ORIGINS=https://your-app.vercel.app
"""

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from features import FEATURE_NAMES, feature_vector, get_signals, normalize_url

MODEL_PATH = Path(__file__).parent / "model" / "phishing_model.joblib"
MAX_URL_LENGTH = 2048

# Probability bands. Because the model was trained on a limited dataset, the
# middle band is reported as "uncertain" instead of forcing a confident verdict.
LOW_BELOW = 0.35
HIGH_FROM = 0.65

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    if bundle["features"] != FEATURE_NAMES:
        raise RuntimeError(
            "Model features do not match features.py. Re-run: python ml/train.py"
        )
    state["model"] = bundle["model"]
    state["model_name"] = bundle["name"]
    yield
    state.clear()


app = FastAPI(title="Phishing URL Detector", version="2.0.0", lifespan=lifespan)

origins = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class URLRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=MAX_URL_LENGTH)

    @field_validator("url")
    @classmethod
    def must_look_like_url(cls, value: str) -> str:
        value = value.strip()
        if re.search(r"\s", value):
            raise ValueError("URL must not contain spaces")
        host = normalize_url(value).split("/", 1)[0].split("?", 1)[0]
        host = host.rsplit("@", 1)[-1].split(":", 1)[0]
        if "." not in host:
            raise ValueError("Enter a valid URL, for example https://example.com/page")
        return value


class PredictionResponse(BaseModel):
    url: str
    label: str            # "phishing" | "legitimate"  (probability >= 0.5)
    risk_level: str       # "low" | "uncertain" | "high"
    probability: float    # model's phishing-class probability, 0..1
    reasons: list[str]    # rule-based signals found in the URL text
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: URLRequest):
    try:
        row = pd.DataFrame([feature_vector(request.url)], columns=FEATURE_NAMES)
        probability = float(state["model"].predict_proba(row)[0][1])
    except Exception:
        raise HTTPException(status_code=422, detail="Could not analyze this URL.")

    if probability >= HIGH_FROM:
        risk_level = "high"
    elif probability < LOW_BELOW:
        risk_level = "low"
    else:
        risk_level = "uncertain"

    return PredictionResponse(
        url=request.url,
        label="phishing" if probability >= 0.5 else "legitimate",
        risk_level=risk_level,
        probability=round(probability, 4),
        reasons=get_signals(request.url),
        model=state["model_name"],
    )
