"""API FastAPI para predição de propensão de compra."""

import logging
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

MODEL_PATH = Path("models/best_model.pkl")
EXPECTED_N_FEATURES = 11

logger = logging.getLogger(__name__)

app = FastAPI(title="Clickstream Purchase Prediction API")
_model = None


class PredictRequest(BaseModel):
    """Payload de entrada: features do clickstream na ordem esperada pelo modelo."""

    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_length(cls, value: list[float]) -> list[float]:
        """Garante que a quantidade de features corresponde ao esperado pelo modelo."""
        if len(value) != EXPECTED_N_FEATURES:
            raise ValueError(
                f"Esperado {EXPECTED_N_FEATURES} features, recebido {len(value)}."
            )
        return value


class PredictResponse(BaseModel):
    """Resposta da predição."""

    prediction: int
    probability: float
    label: str


def _load_model():
    """Carrega o modelo treinado (lazy loading)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Modelo não encontrado em {MODEL_PATH}. "
                    "Execute o export do modelo antes de servir a API."
                ),
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    """Endpoint de health check usado pelo Render."""
    return {"status": "ok", "model_loaded": str(_model is not None)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Prediz a propensão de compra a partir das features do clickstream."""
    model = _load_model()
    X = [request.features]
    try:
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
    except Exception as exc:
        logger.exception("Falha ao executar predição")
        raise HTTPException(status_code=400, detail=f"Erro ao processar features: {exc}") from exc
    return PredictResponse(
        prediction=prediction,
        probability=round(probability, 4),
        label="compra" if prediction == 1 else "sem_compra",
    )
