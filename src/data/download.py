"""Módulo para download do dataset Clickstream Data for Online Shopping (UCI)."""

import logging
import urllib.request
import zipfile
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/553/" "clickstream+data+for+online+shopping.zip"
)
_EXTRACTED_NAME = "e-shop clothing 2008.csv"
_DESC_NAME = "e-shop clothing 2008 data description.txt"


def _download_zip(zip_path: Path) -> None:
    """Baixa o ZIP do UCI."""
    logger.info("Baixando dataset de: %s", DATASET_URL)
    urllib.request.urlretrieve(DATASET_URL, zip_path)


def _extract_and_rename(zip_path: Path, dest: Path) -> None:
    """Extrai ZIP, renomeia CSV e limpa temporários."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest.parent)
    (dest.parent / _EXTRACTED_NAME).rename(dest)
    zip_path.unlink(missing_ok=True)
    (dest.parent / _DESC_NAME).unlink(missing_ok=True)


def download_dataset(output_path: Path | None = None) -> Path:
    """Baixa e extrai o dataset do UCI (165.474 registros)."""
    path = output_path or settings.data_raw_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        logger.info("Dataset já existe em: %s", path)
        return path
    zip_path = path.parent / "clickstream.zip"
    _download_zip(zip_path)
    _extract_and_rename(zip_path, path)
    logger.info("Dataset salvo em: %s", path)
    return path


if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()
    download_dataset()
