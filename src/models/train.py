"""Módulo de treinamento de múltiplos modelos com tracking via MLflow."""

import json
import logging
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import settings

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "clickstream-purchase-model"


def load_processed_data() -> dict:
    """Carrega os dados pré-processados."""
    path = settings.data_processed_path
    return {
        "X_train": pd.read_csv(path / "X_train.csv"),
        "X_test": pd.read_csv(path / "X_test.csv"),
        "y_train": pd.read_csv(path / "y_train.csv").squeeze(),
        "y_test": pd.read_csv(path / "y_test.csv").squeeze(),
    }


def _build_logistic_regression(seed: int) -> dict:
    """Cria config da Regressão Logística."""
    params = {"max_iter": 1000, "solver": "lbfgs"}
    return {
        "name": "LogisticRegression",
        "model": LogisticRegression(random_state=seed, **params),
        "params": params,
    }


def _build_decision_tree(seed: int) -> dict:
    """Cria config da Árvore de Decisão."""
    params = {"max_depth": 10, "min_samples_split": 10}
    return {
        "name": "DecisionTree",
        "model": DecisionTreeClassifier(random_state=seed, **params),
        "params": params,
    }


def _build_random_forest(seed: int) -> dict:
    """Cria config do Random Forest."""
    params = {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5}
    return {
        "name": "RandomForest",
        "model": RandomForestClassifier(random_state=seed, n_jobs=-1, **params),
        "params": params,
    }


def _build_gradient_boosting(seed: int) -> dict:
    """Cria config do Gradient Boosting."""
    params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8}
    return {
        "name": "GradientBoosting",
        "model": GradientBoostingClassifier(random_state=seed, **params),
        "params": params,
    }


def _build_mlp(seed: int) -> dict:
    """Cria config da MLP Neural Network."""
    params = {
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        "max_iter": 500,
        "early_stopping": True,
    }
    return {
        "name": "MLP_NeuralNetwork",
        "model": MLPClassifier(random_state=seed, **params),
        "params": {**params, "hidden_layer_sizes": "(128,64,32)"},
    }


def get_model_configs() -> list[dict]:
    """Retorna lista de modelos para comparação."""
    seed = settings.random_seed
    return [
        _build_logistic_regression(seed),
        _build_decision_tree(seed),
        _build_random_forest(seed),
        _build_gradient_boosting(seed),
        _build_mlp(seed),
    ]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calcula métricas de avaliação."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def _log_to_mlflow(model, name: str, params: dict, metrics: dict) -> None:
    """Registra parâmetros, métricas e modelo (como artefato da run) no MLflow."""
    mlflow.log_param("model_type", name)
    mlflow.log_param("random_seed", settings.random_seed)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")


def train_single_model(config: dict, data: dict) -> dict[str, float]:
    """Treina um modelo e loga parâmetros/métricas/artefato no MLflow."""
    name, model, params = config["name"], config["model"], config["params"]
    with mlflow.start_run(run_name=name) as run:
        start = time.time()
        model.fit(data["X_train"], data["y_train"])
        elapsed = time.time() - start
        metrics = evaluate_model(model, data["X_test"], data["y_test"])
        metrics["training_time_seconds"] = round(elapsed, 2)
        _log_to_mlflow(model, name, params, metrics)
        metrics["run_id"] = run.info.run_id
        auc, f1 = metrics["roc_auc"], metrics["f1_score"]
        logger.info("  %-20s | AUC=%.4f | F1=%.4f | %.1fs", name, auc, f1, elapsed)
    return metrics


def find_best_model(results: dict[str, dict]) -> str:
    """Identifica o melhor modelo por ROC AUC."""
    return max(results, key=lambda n: results[n]["roc_auc"])


def promote_best_model(name: str, run_id: str, roc_auc: float) -> None:
    """Registra o melhor modelo no Model Registry e o promove para Production."""
    client = MlflowClient()
    model_version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        description=f"Melhor modelo: {name} (ROC AUC={roc_auc:.4f})",
    )
    logger.info("'%s' registrado como '%s' v%s -> Production", name, REGISTERED_MODEL_NAME, model_version.version)


def train_all_models() -> dict[str, dict]:
    """Treina todos os modelos, compara, registra o melhor e salva métricas."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    data = load_processed_data()
    results: dict[str, dict] = {}

    logger.info("TREINO — %d amostras", len(data['X_train']) + len(data['X_test']))

    for config in get_model_configs():
        results[config["name"]] = train_single_model(config, data)

    best = find_best_model(results)
    logger.info("MELHOR: %s (AUC=%.4f)", best, results[best]['roc_auc'])
    promote_best_model(best, results[best]["run_id"], results[best]["roc_auc"])
    results["_best_model"] = best

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()
    train_all_models()
