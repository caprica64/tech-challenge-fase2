"""Módulo para exportar o melhor modelo treinado para deploy."""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings

logger = logging.getLogger(__name__)


def _train_best_pipeline() -> Pipeline:
    """Treina pipeline completo (scaler + modelo) para export."""
    path = settings.data_processed_path
    X_train = pd.read_csv(path / "X_train.csv")
    y_train = pd.read_csv(path / "y_train.csv").squeeze()

    # Dados já estão escalados, retreinar com dados brutos
    from src.data.preprocess import (
        create_target_variable,
        engineer_features,
        load_raw_data,
        split_features_and_target,
    )

    df = load_raw_data()
    df = create_target_variable(df)
    df = engineer_features(df)
    features, target = split_features_and_target(df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.1, subsample=0.8,
            random_state=settings.random_seed,
        )),
    ])
    pipeline.fit(features, target)
    return pipeline


def export_model(output_path: Path | None = None) -> Path:
    """Exporta pipeline (scaler + modelo) como pickle."""
    path = output_path or Path("models/best_model.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = _train_best_pipeline()
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info("Modelo exportado em: %s", path)
    return path


if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()
    export_model()
