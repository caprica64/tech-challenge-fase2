# === Stage 1: Instalar dependências ===
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

# Camada de dependências (cache enquanto pyproject/lock não mudam)
COPY pyproject.toml poetry.lock ./
RUN poetry install --only=main --no-root

# === Stage 2: Imagem final enxuta ===
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copiar virtualenv do builder
COPY --from=builder /app/.venv .venv

# Copiar código fonte (camada separada — muda com frequência)
COPY src/ ./src/
COPY configs/ ./configs/
COPY dvc.yaml README.md ./

# Usuário não-root para segurança
RUN useradd --create-home appuser
USER appuser

CMD ["python", "-m", "src.models.train"]
