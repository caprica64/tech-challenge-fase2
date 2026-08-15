"""Módulo para exportar o melhor modelo treinado para deploy."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.config import settings


def _train_best_model() -> GradientBoostingClassifier:
    """Treina o GradientBoosting (melhor modelo) para export."""
    path = settings.data_processed_path
    X_train = pd.read_csv(path / "X_train.csv")
    y_train = pd.read_csv(path / "y_train.csv").squeeze()

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=5,
        learning_rate=0.1, subsample=0.8,
        random_state=settings.random_seed,
    )
    model.fit(X_train, y_train)
    return model


def export_model(output_path: Path | None = None) -> Path:
    """Exporta o melhor modelo como pickle para inferência."""
    path = output_path or Path("models/best_model.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)

    model = _train_best_model()
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modelo exportado em: {path}")
    return path


if __name__ == "__main__":
    export_model()
