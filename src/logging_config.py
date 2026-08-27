"""Configuração centralizada de logging para o projeto."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configura o logging raiz com formato padronizado e saída para stdout.

    Deve ser chamado uma única vez no ponto de entrada de cada script CLI
    (bloco ``if __name__ == "__main__"``). A API FastAPI não precisa chamar
    esta função — o uvicorn já configura o logging ao iniciar.

    Args:
        level: Nível mínimo de log (padrão: INFO).
    """
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
