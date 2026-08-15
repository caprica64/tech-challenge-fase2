# Tech Challenge Fase 2 — Propensão de Compra em E-commerce

Sistema preditivo para identificar a propensão de compra de um usuário baseado em seu comportamento de navegação, utilizando o dataset **Clickstream Data for Online Shopping** (UCI, 165k registros).

## Arquitetura do Projeto

```
tech-challenge-fase2/
├── src/
│   ├── config.py              # Configurações centralizadas via .env
│   ├── data/
│   │   ├── download.py        # Download do dataset UCI
│   │   └── preprocess.py      # Feature engineering + split + scale
│   └── models/
│       └── train.py           # Treino comparativo (5 modelos) + MLflow
├── tests/
│   └── test_preprocess.py     # Testes unitários
├── data/
│   ├── raw/                   # Dataset bruto (versionado via DVC)
│   └── processed/             # Dados processados (gerado pelo pipeline)
├── models/                    # Artefatos de modelos
├── configs/
│   └── model_params.yaml      # Hiperparâmetros documentados
├── results/
│   └── experiment_results.md  # Resultados dos experimentos
├── pyproject.toml             # Dependências (Poetry)
├── poetry.lock                # Lock file
├── Dockerfile                 # Containerização
├── docker-compose.yml         # Orquestração (app + MLflow)
├── dvc.yaml                   # Pipeline DVC (preprocess → train)
├── .env.example               # Template de variáveis de ambiente
├── .gitignore
└── .dockerignore
```

## Pré-requisitos

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) >= 1.8
- [Docker](https://docs.docker.com/get-docker/) e Docker Compose
- [DVC](https://dvc.org/doc/install) >= 3.0
- Credenciais AWS configuradas (para `dvc pull` do S3)

## Setup Rápido

### 1. Clonar e instalar dependências

```bash
git clone <url-do-repositorio>
cd tech-challenge-fase2
poetry install
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Editar .env com suas credenciais e configurações
```

### 3. Obter o dataset

```bash
# Opção A: puxar do S3 via DVC (requer credenciais AWS)
dvc pull

# Opção B: baixar direto do UCI Machine Learning Repository
poetry run python -m src.data.download
```

### 4. Subir o MLflow server

```bash
docker compose up mlflow -d
# Acessar UI em http://localhost:5000
```

### 5. Executar o pipeline completo

```bash
poetry run dvc repro
```

O `dvc repro` executa automaticamente os dois estágios do pipeline:

1. **preprocess** — carrega dados brutos, cria variável target, engenharia de features, split estratificado (80/20) e StandardScaler.
2. **train** — treina 5 modelos, loga parâmetros/métricas no MLflow, registra o melhor no Model Registry e salva `metrics.json`.

## Execução via Docker

```bash
# Build e execução completa (app + MLflow)
docker compose up --build

# Apenas o MLflow UI
docker compose up mlflow -d
```

## Pipeline DVC

```bash
# Reproduzir pipeline
dvc repro

# Verificar se há mudanças pendentes
dvc status

# Enviar dados para o remote S3
dvc push

# Baixar dados do remote S3
dvc pull
```

## Modelos Comparados

| Modelo | ROC AUC | F1-Score | Tempo |
|---|---|---|---|
| Logistic Regression | 0.7256 | 0.5311 | 0.2s |
| Decision Tree | 0.7816 | 0.6599 | 0.1s |
| Random Forest | 0.7926 | 0.6690 | 1.9s |
| **Gradient Boosting** | **0.8038** | **0.6812** | **20.6s** |
| MLP Neural Network | 0.7726 | 0.6396 | 14.0s |

**Melhor modelo:** Gradient Boosting (ROC AUC = 0.8038)

## MLflow

O tracking de experimentos registra para cada modelo:
- **Parâmetros:** hiperparâmetros específicos + random_seed + dataset_size
- **Métricas:** accuracy, precision, recall, f1_score, roc_auc, training_time_seconds
- **Artefatos:** modelo serializado registrado no MLflow Model Registry

UI: http://localhost:5000

## Testes

```bash
poetry run pytest -v
```

## Lint

```bash
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/
```

## Deploy do Endpoint de Inferência (Opcional)

O projeto inclui um endpoint de inferência serverless via **AWS Lambda** com container image. Custo praticamente zero para demonstração.

### Pré-requisitos

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) instalado
- Docker instalado e rodando
- Credenciais AWS configuradas (IAM User ou Role)

### Configuração de Credenciais AWS

**Opção recomendada: IAM Role (via SSO ou instance profile)**

```bash
aws configure sso
# Seguir o fluxo de autenticação SSO
```

**Opção alternativa: IAM User com Access Keys**

```bash
aws configure
# Informar AWS_ACCESS_KEY_ID e AWS_SECRET_ACCESS_KEY
```

O IAM User/Role precisa das seguintes permissões:
- `ecr:CreateRepository`, `ecr:GetAuthorizationToken`, `ecr:PutImage`
- `lambda:CreateFunction`, `lambda:UpdateFunctionCode`, `lambda:CreateFunctionUrlConfig`
- `iam:PassRole` (para associar a execution role ao Lambda)

### Criar a Execution Role do Lambda

Antes do primeiro deploy, crie a role que o Lambda usará:

```bash
aws iam create-role \
  --role-name lambda-execution-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### Executar o Deploy

```bash
./scripts/deploy_inference.sh
```

O script automatiza: exportar modelo → build da imagem → push para ECR → criar/atualizar Lambda → criar Function URL.

### Obter o Endpoint

Ao final do script, a URL é exibida. Para consultar depois:

```bash
aws lambda get-function-url-config \
  --function-name clickstream-purchase-prediction \
  --query FunctionUrl \
  --output text
```

### Testar o Endpoint

```bash
curl -X POST <URL_DO_LAMBDA> \
  -H 'Content-Type: application/json' \
  -d '{"features": [4, 15, 3, 29, 1, 2, 3, 1, 35, 1, 2]}'
```

Resposta esperada:

```json
{"prediction": 1, "probability": 0.7234, "label": "compra"}
```

As features correspondem a: month, day, click_order, country, main_category, colour, photo_location, model_photography, price, price_above_avg, page_number.

### Exemplos de Teste

```bash
# Teste 1: Sem compra — Alemão, primeiro click, produto barato
curl -X POST <URL> -H 'Content-Type: application/json' \
  -d '{"features": [4, 3, 1, 16, 3, 5, 1, 2, 18, 2, 1]}'
# → {"prediction": 0, "probability": 0.1992, "label": "sem_compra"}

# Teste 2: Sem compra — Francês, navegação média, preço alto
curl -X POST <URL> -H 'Content-Type: application/json' \
  -d '{"features": [7, 20, 4, 15, 2, 1, 5, 1, 55, 1, 2]}'
# → {"prediction": 0, "probability": 0.1161, "label": "sem_compra"}

# Teste 3: Compra — Polonês, navegação longa, produto caro
curl -X POST <URL> -H 'Content-Type: application/json' \
  -d '{"features": [6, 15, 8, 29, 1, 2, 3, 1, 85, 1, 3]}'
# → {"prediction": 1, "probability": 0.616, "label": "compra"}

# Teste 4: Compra — Polonês, muitos clicks, calças baratas, página 5
curl -X POST <URL> -H 'Content-Type: application/json' \
  -d '{"features": [8, 28, 15, 29, 1, 9, 6, 2, 15, 2, 5]}'
# → {"prediction": 1, "probability": 0.9446, "label": "compra"}
```

Para testar localmente (sem deploy):

```bash
poetry run python -m src.models.export
poetry run python -c "
import json
from src.api.inference import handler
event = {'body': json.dumps({'features': [6, 15, 8, 29, 1, 2, 3, 1, 85, 1, 3]})}
print(json.loads(handler(event)['body']))
"
```

## Dataset

- **Nome:** Clickstream Data for Online Shopping
- **Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/553)
- **Registros:** 165.474 (brutos) → 126.727 (após feature engineering)
- **Target:** Sessão resultou em visita à página de compra (classificação binária)
- **Licença:** CC BY 4.0

## Tecnologias

| Ferramenta | Uso |
|---|---|
| Scikit-Learn | Pré-processamento e modelos de classificação |
| MLflow | Tracking de experimentos e Model Registry |
| DVC | Versionamento de dados e pipeline reprodutível |
| Poetry | Gerenciamento de dependências |
| Docker | Containerização do ambiente |
| Ruff | Linting e formatação |

## Grupo

- [Nome 1] — RM XXXXX
- [Nome 2] — RM XXXXX
- [Nome 3] — RM XXXXX

## Licença

Projeto acadêmico — FIAP Pós Tech 10MLET
