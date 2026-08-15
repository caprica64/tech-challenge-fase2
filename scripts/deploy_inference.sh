#!/bin/bash
# Deploy do endpoint de inferência no AWS Lambda (Container Image)
# Pré-requisitos: AWS CLI configurado, Docker instalado

set -e

REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="tech-challenge-inference"
FUNCTION_NAME="clickstream-purchase-prediction"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"

echo "=== 1. Exportar modelo ==="
poetry run python -m src.models.export

echo "=== 2. Criar repositório ECR ==="
aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION} 2>/dev/null || true

echo "=== 3. Login no ECR ==="
aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "=== 4. Build da imagem ==="
docker build -f Dockerfile.inference -t ${REPO_NAME} .

echo "=== 5. Tag e push ==="
docker tag ${REPO_NAME}:latest ${IMAGE_URI}
docker push ${IMAGE_URI}

echo "=== 6. Criar/Atualizar Lambda ==="
aws lambda create-function \
  --function-name ${FUNCTION_NAME} \
  --package-type Image \
  --code ImageUri=${IMAGE_URI} \
  --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role \
  --timeout 30 \
  --memory-size 512 \
  --region ${REGION} 2>/dev/null || \
aws lambda update-function-code \
  --function-name ${FUNCTION_NAME} \
  --image-uri ${IMAGE_URI} \
  --region ${REGION}

echo "=== 7. Criar Function URL (pública) ==="
aws lambda create-function-url-config \
  --function-name ${FUNCTION_NAME} \
  --auth-type NONE \
  --region ${REGION} 2>/dev/null || true

URL=$(aws lambda get-function-url-config \
  --function-name ${FUNCTION_NAME} \
  --query FunctionUrl --output text \
  --region ${REGION})

echo ""
echo "=== DEPLOY COMPLETO ==="
echo "Endpoint: ${URL}"
echo ""
echo "Teste:"
echo "curl -X POST ${URL} -H 'Content-Type: application/json' \\"
echo "  -d '{\"features\": [4, 15, 3, 29, 1, 2, 3, 1, 35, 1, 2]}'"
