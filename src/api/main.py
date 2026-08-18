"""API FastAPI para predição de propensão de compra."""

import pickle
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models/best_model.pkl")

app = FastAPI(title="Clickstream Purchase Prediction API")
_model = None


class PredictRequest(BaseModel):
    """Payload de entrada: features do clickstream na ordem esperada pelo modelo."""

    features: list[float]


class PredictResponse(BaseModel):
    """Resposta da predição."""

    prediction: int
    probability: float
    label: str


def _load_model():
    """Carrega o modelo treinado (lazy loading)."""
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    """Endpoint de health check usado pelo Render."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Prediz a propensão de compra a partir das features do clickstream."""
    model = _load_model()
    X = [request.features]
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return PredictResponse(
        prediction=prediction,
        probability=round(probability, 4),
        label="compra" if prediction == 1 else "sem_compra",
    )
