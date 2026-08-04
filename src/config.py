"""Módulo de configuração centralizada do projeto."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Configurações globais do projeto carregadas via variáveis de ambiente."""

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_raw_path: Path = Path(os.getenv("DATA_RAW_PATH", "data/raw/clickstream_shop.csv"))
    data_processed_path: Path = Path(os.getenv("DATA_PROCESSED_PATH", "data/processed/"))

    # MLflow
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "clickstream-purchase-prediction"
    )

    # Modelo
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))


settings = Settings()
