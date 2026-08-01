FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copiar arquivos de dependência primeiro (cache de camadas)
COPY pyproject.toml poetry.lock ./

# Instalar dependências de produção
RUN poetry install --only=main --no-root

# Copiar código fonte
COPY src/ ./src/
COPY dvc.yaml ./
COPY configs/ ./configs/
COPY README.md ./

# Instalar o pacote do projeto
RUN poetry install --only=main

CMD ["python", "-m", "src.models.train"]
