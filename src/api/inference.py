"""Endpoint de inferência para predição de propensão de compra."""

import json
import pickle
from pathlib import Path

MODEL_PATH = Path("models/best_model.pkl")
_model = None


def _load_model():
    """Carrega o modelo treinado (lazy loading)."""
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def _parse_features(body: dict) -> list[list[float]]:
    """Extrai features do payload JSON."""
    return [body.get("features", [])]


def _build_response(status: int, body: dict) -> dict:
    """Constrói resposta HTTP padronizada."""
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event: dict, context=None) -> dict:
    """Handler Lambda para predição de compra.

    Payload esperado:
        {"features": [month, day, click_order, country, main_category,
                      colour, photo_location, model_photography, price,
                      price_above_avg, page_number]}

    Retorna:
        {"prediction": 0|1, "probability": float}
    """
    try:
        body = json.loads(event.get("body", "{}"))
        features = _parse_features(body)
        model = _load_model()

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        return _build_response(200, {
            "prediction": prediction,
            "probability": round(probability, 4),
            "label": "compra" if prediction == 1 else "sem_compra",
        })
    except Exception as e:
        return _build_response(500, {"error": str(e)})
