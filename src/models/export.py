"""Módulo para exportar o melhor modelo treinado para deploy."""

import json
import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.train import get_model_configs

METRICS_PATH = Path("metrics.json")


def _load_best_model_name(metrics_path: Path = METRICS_PATH) -> str | None:
    """Lê o metrics.json (gerado por train_all_models) e retorna o nome do melhor modelo.

    Retorna None se o arquivo não existir ou não contiver a chave, permitindo
    fallback para um modelo padrão.
    """
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        results = json.load(f)
    return results.get("_best_model")


def _build_model(name: str | None):
    """Instancia o estimador correspondente ao nome informado.

    Se `name` for None ou não reconhecido, usa o GradientBoosting como fallback
    (mesmo comportamento anterior), mantendo o export funcional mesmo sem
    um treinamento prévio.
    """
    configs = {c["name"]: c["model"] for c in get_model_configs()}
    if name in configs:
        return configs[name]
    print(f"Aviso: modelo '{name}' não encontrado, usando GradientBoosting como fallback.")
    return configs["GradientBoosting"]


def _train_best_pipeline() -> Pipeline:
    """Treina pipeline completo (scaler + modelo vencedor) para export."""
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

    best_name = _load_best_model_name()
    model = _build_model(best_name)
    print(f"Exportando modelo: {best_name or 'GradientBoosting (fallback)'}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
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

    print(f"Modelo exportado em: {path}")
    return path


if __name__ == "__main__":
    export_model()
