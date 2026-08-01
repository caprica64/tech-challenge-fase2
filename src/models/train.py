"""Módulo de treinamento de múltiplos modelos com tracking via MLflow."""

import json
import time

import mlflow
import mlflow.sklearn
import pandas as pd
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


def load_processed_data() -> dict:
    """Carrega os dados pré-processados do diretório configurado.

    Returns:
        Dicionário com X_train, X_test, y_train e y_test.
    """
    processed_path = settings.data_processed_path
    return {
        "X_train": pd.read_csv(processed_path / "X_train.csv"),
        "X_test": pd.read_csv(processed_path / "X_test.csv"),
        "y_train": pd.read_csv(processed_path / "y_train.csv").squeeze(),
        "y_test": pd.read_csv(processed_path / "y_test.csv").squeeze(),
    }


def get_model_configs() -> list[dict]:
    """Retorna a lista de modelos com seus hiperparâmetros para comparação.

    Returns:
        Lista de dicionários com nome, instância e parâmetros de cada modelo.
    """
    seed = settings.random_seed

    return [
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
            "params": {
                "max_iter": 1000,
                "solver": "lbfgs",
            },
        },
        {
            "name": "DecisionTree",
            "model": DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                random_state=seed,
            ),
            "params": {
                "max_depth": 10,
                "min_samples_split": 10,
            },
        },
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=seed,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
            },
        },
        {
            "name": "GradientBoosting",
            "model": GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=seed,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
            },
        },
        {
            "name": "MLP_NeuralNetwork",
            "model": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                random_state=seed,
            ),
            "params": {
                "hidden_layer_sizes": "(128, 64, 32)",
                "activation": "relu",
                "max_iter": 500,
                "early_stopping": True,
            },
        },
    ]


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calcula métricas de avaliação do modelo.

    Args:
        model: Modelo treinado (qualquer estimator sklearn).
        X_test: Features de teste.
        y_test: Labels reais de teste.

    Returns:
        Dicionário com métricas calculadas.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_single_model(model_config: dict, data: dict) -> dict[str, float]:
    """Treina um modelo individual e registra no MLflow.

    Args:
        model_config: Dicionário com name, model e params.
        data: Dicionário com X_train, X_test, y_train, y_test.

    Returns:
        Dicionário de métricas do modelo.
    """
    model_name = model_config["name"]
    model = model_config["model"]
    params = model_config["params"]

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("random_seed", settings.random_seed)
        mlflow.log_param("dataset_size", len(data["X_train"]) + len(data["X_test"]))
        mlflow.log_params(params)

        start_time = time.time()
        model.fit(data["X_train"], data["y_train"])
        training_time = time.time() - start_time

        metrics = evaluate_model(model, data["X_test"], data["y_test"])
        metrics["training_time_seconds"] = round(training_time, 2)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"clickstream-{model_name.lower()}",
        )

        print(
            f"  {model_name:<20} | ROC AUC={metrics['roc_auc']:.4f} "
            f"| F1={metrics['f1_score']:.4f} "
            f"| Tempo={training_time:.1f}s"
        )

    return metrics


def find_best_model(results: dict[str, dict]) -> str:
    """Identifica o melhor modelo baseado no ROC AUC.

    Args:
        results: Dicionário {nome_modelo: métricas}.

    Returns:
        Nome do melhor modelo.
    """
    return max(results, key=lambda name: results[name]["roc_auc"])


def train_all_models() -> dict[str, dict]:
    """Treina todos os modelos, compara e salva resultados.

    Returns:
        Dicionário com métricas de cada modelo.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    data = load_processed_data()
    model_configs = get_model_configs()
    results: dict[str, dict] = {}

    print("=" * 65)
    print("TREINAMENTO DE MODELOS — Propensão de Compra (Clickstream)")
    print(f"Dataset: {len(data['X_train']) + len(data['X_test'])} amostras")
    print("=" * 65)

    for config in model_configs:
        metrics = train_single_model(config, data)
        results[config["name"]] = metrics

    best_model_name = find_best_model(results)
    results["_best_model"] = best_model_name

    print("\n" + "=" * 65)
    print(
        f"MELHOR MODELO: {best_model_name} "
        f"(ROC AUC = {results[best_model_name]['roc_auc']:.4f})"
    )
    print("=" * 65)

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_all_models()
