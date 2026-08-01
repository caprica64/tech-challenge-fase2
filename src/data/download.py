"""Módulo para download do dataset Clickstream Data for Online Shopping (UCI)."""

import urllib.request
import zipfile
from pathlib import Path

from src.config import settings

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/553/"
    "clickstream+data+for+online+shopping.zip"
)


def download_dataset(output_path: Path | None = None) -> Path:
    """Baixa e extrai o dataset do UCI Machine Learning Repository.

    O dataset contém 165.474 registros de clickstream de uma loja
    online de roupas para gestantes (2008), com 14 features.

    Args:
        output_path: Caminho onde salvar o CSV final. Usa padrão se None.

    Returns:
        Caminho do arquivo CSV extraído.
    """
    path = output_path or settings.data_raw_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"Dataset já existe em: {path}")
        return path

    zip_path = path.parent / "clickstream.zip"

    print(f"Baixando dataset de: {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    print("Extraindo arquivo ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path.parent)

    # Renomear para o nome padrão
    extracted_file = path.parent / "e-shop clothing 2008.csv"
    extracted_file.rename(path)

    # Limpar arquivos temporários
    zip_path.unlink()
    desc_file = path.parent / "e-shop clothing 2008 data description.txt"
    if desc_file.exists():
        desc_file.unlink()

    print(f"Dataset salvo em: {path} (165.474 registros)")
    return path


if __name__ == "__main__":
    download_dataset()
