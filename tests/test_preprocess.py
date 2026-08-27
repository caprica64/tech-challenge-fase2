import json
import pickle

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api import inference
from src.api.main import app
from src.config import settings
from src.data.preprocess import (
    create_target_variable,
    engineer_features,
    run_preprocessing,
    split_features_and_target,
)


@pytest.fixture
def sample_clickstream() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2008, 2008, 2008, 2008, 2008],
            "month": [4, 4, 4, 4, 4],
            "day": [1, 1, 1, 1, 2],
            "order": [1, 2, 3, 1, 1],
            "country": [29, 29, 29, 15, 15],
            "session ID": [1, 1, 1, 2, 2],
            "page 1 (main category)": [1, 2, 4, 1, 3],
            "page 2 (clothing model)": ["A1", "B2", "C3", "A1", "B2"],
            "colour": [1, 2, 3, 1, 5],
            "location": [1, 2, 3, 4, 5],
            "model photography": [1, 2, 1, 1, 2],
            "price": [28, 35, 50, 42, 30],
            "price 2": [1, 2, 1, 2, 1],
            "page": [1, 1, 2, 1, 1],
        }
    )


@pytest.fixture
def trained_model(tmp_path):
    X = [[float(i), float(i + 1), float(i + 2), float(i + 3), float(i + 4), float(i + 5), float(i + 6), float(i + 7), float(i + 8), float(i + 9), float(i + 10)] for i in range(60)]
    y = [0 if i % 2 == 0 else 1 for i in range(60)]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    model.fit(X, y)
    model_path = tmp_path / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path


def test_create_target_variable(sample_clickstream: pd.DataFrame) -> None:
    result = create_target_variable(sample_clickstream)

    assert "purchased" in result.columns
    assert 4 not in result["page 1 (main category)"].values

    session_1 = result[result["session ID"] == 1]
    session_2 = result[result["session ID"] == 2]
    assert (session_1["purchased"] == 1).all()
    assert (session_2["purchased"] == 0).all()


def test_engineer_features(sample_clickstream: pd.DataFrame) -> None:
    df_with_target = create_target_variable(sample_clickstream)
    result = engineer_features(df_with_target)

    assert "year" not in result.columns
    assert "session ID" not in result.columns
    assert "click_order" in result.columns
    assert "main_category" in result.columns
    assert "purchased" in result.columns


def test_split_features_and_target(sample_clickstream: pd.DataFrame) -> None:
    df_with_target = create_target_variable(sample_clickstream)
    df_features = engineer_features(df_with_target)
    features, target = split_features_and_target(df_features)

    assert "purchased" not in features.columns
    assert target.name == "purchased"
    assert len(features) == len(target)


def test_run_preprocessing_creates_processed_files(tmp_path, monkeypatch) -> None:
    rows = []
    for session_id in range(1, 11):
        for offset in range(2):
            row = {
                "year": 2008,
                "month": 4,
                "day": session_id,
                "order": offset + 1,
                "country": 29,
                "session ID": session_id,
                "page 1 (main category)": 4 if session_id % 2 == 0 else 1,
                "page 2 (clothing model)": f"M{session_id}{offset}",
                "colour": 1 + offset,
                "location": 1 + offset,
                "model photography": 1,
                "price": 20 + session_id + offset,
                "price 2": 1 if offset == 0 else 2,
                "page": 1,
            }
            rows.append(row)

    raw_path = tmp_path / "clickstream.csv"
    processed_dir = tmp_path / "processed"
    object.__setattr__(settings, "data_processed_path", processed_dir)
    pd.DataFrame(rows).to_csv(raw_path, sep=";", index=False)

    result = run_preprocessing(raw_path)

    assert set(result) == {"X_train", "X_test", "y_train", "y_test", "scaler"}
    assert result["X_train"].shape[1] == result["X_test"].shape[1]
    assert len(result["X_train"]) > 0
    assert len(result["X_test"]) > 0
    assert processed_dir.exists()
    assert (processed_dir / "X_train.csv").exists()
    assert (processed_dir / "y_train.csv").exists()


def test_fastapi_prediction_returns_valid_response(trained_model, monkeypatch):
    import src.api.main as api_main

    monkeypatch.setattr(api_main, "MODEL_PATH", trained_model)
    api_main._model = None

    response = TestClient(api_main.app).post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"prediction", "probability", "label"}
    assert payload["prediction"] in {0, 1}
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["label"] in {"compra", "sem_compra"}


def test_fastapi_prediction_rejects_invalid_feature_count(trained_model, monkeypatch):
    import src.api.main as api_main

    monkeypatch.setattr(api_main, "MODEL_PATH", trained_model)
    api_main._model = None

    response = TestClient(api_main.app).post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
    )

    assert response.status_code == 422


def test_inference_handler_validates_payload_and_returns_prediction(trained_model, monkeypatch):
    monkeypatch.setattr(inference, "MODEL_PATH", trained_model)
    inference._model = None

    valid_response = inference.handler({"body": json.dumps({"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]})})
    invalid_response = inference.handler({"body": json.dumps({"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})})

    assert valid_response["statusCode"] == 200
    valid_payload = json.loads(valid_response["body"])
    assert valid_payload["prediction"] in {0, 1}
    assert 0.0 <= valid_payload["probability"] <= 1.0

    assert invalid_response["statusCode"] == 400
    invalid_payload = json.loads(invalid_response["body"])
    assert "features" in invalid_payload["error"].lower()
