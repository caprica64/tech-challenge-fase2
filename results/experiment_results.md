# Resultados dos Experimentos

## Dataset
- **Nome:** Clickstream Data for Online Shopping (UCI)
- **Registros brutos:** 165.474
- **Registros após feature engineering:** 126.727
- **Split:** 101.381 treino / 25.346 teste (80/20, estratificado)
- **Proporção target positivo:** 44.89%

## Comparação de Modelos

| Modelo | ROC AUC | F1-Score | Accuracy | Precision | Recall | Tempo (s) |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.7256 | 0.5311 | — | — | — | 0.2 |
| DecisionTree | 0.7816 | 0.6599 | — | — | — | 0.1 |
| RandomForest | 0.7926 | 0.6690 | — | — | — | 1.9 |
| **GradientBoosting** | **0.8038** | **0.6812** | — | — | — | **20.6** |
| MLP_NeuralNetwork | 0.7726 | 0.6396 | — | — | — | 14.0 |

## Melhor Modelo
- **GradientBoosting** com ROC AUC = 0.8038
- Parâmetros: n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8
- Registrado no MLflow Model Registry como `clickstream-gradientboosting`

## Seed
- random_state = 42 (fixado para reprodutibilidade)
