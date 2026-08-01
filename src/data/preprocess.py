"""Módulo de pré-processamento dos dados de clickstream do e-commerce."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import settings


def load_raw_data(file_path: Path | None = None) -> pd.DataFrame:
    """Carrega o dataset bruto (CSV separado por ponto-e-vírgula).

    Args:
        file_path: Caminho opcional para o arquivo CSV. Usa o padrão se None.

    Returns:
        DataFrame com os dados brutos carregados.
    """
    path = file_path or settings.data_raw_path
    return pd.read_csv(path, sep=";")


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a variável alvo 'purchased' baseada no comportamento de sessão.

    Uma sessão é marcada como 'purchased=1' se em algum momento o
    usuário visitou a categoria 'sale' (código 4). Os clicks que já
    estão na página sale são removidos para evitar data leakage.

    Args:
        df: DataFrame bruto com todas as colunas do clickstream.

    Returns:
        DataFrame filtrado com coluna 'purchased' adicionada.
    """
    category_column = "page 1 (main category)"
    sale_category = 4

    sessions_with_purchase = df[df[category_column] == sale_category]["session ID"].unique()
    df = df.copy()
    df["purchased"] = df["session ID"].isin(sessions_with_purchase).astype(int)

    # Remover clicks na própria página sale (evitar leakage)
    df_filtered = df[df[category_column] != sale_category].copy()
    return df_filtered


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e transforma features para o modelo.

    Remove colunas identificadoras e renomeia para nomes descritivos.

    Args:
        df: DataFrame com target criado.

    Returns:
        DataFrame com features selecionadas e target.
    """
    columns_to_drop = ["year", "session ID", "page 2 (clothing model)"]
    df_features = df.drop(columns=columns_to_drop)

    column_mapping = {
        "month": "month",
        "day": "day",
        "order": "click_order",
        "country": "country",
        "page 1 (main category)": "main_category",
        "colour": "colour",
        "location": "photo_location",
        "model photography": "model_photography",
        "price": "price",
        "price 2": "price_above_avg",
        "page": "page_number",
        "purchased": "purchased",
    }
    return df_features.rename(columns=column_mapping)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features da variável alvo (purchased).

    Args:
        df: DataFrame com features e target.

    Returns:
        Tupla contendo (features, target).
    """
    target_column = "purchased"
    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Aplica StandardScaler nas features numéricas.

    Args:
        X_train: Features de treino.
        X_test: Features de teste.

    Returns:
        Tupla com (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing(file_path: Path | None = None) -> dict:
    """Executa o pipeline completo de pré-processamento.

    Pipeline: load → create target → engineer features → split → scale → save.

    Args:
        file_path: Caminho opcional para o CSV de entrada.

    Returns:
        Dicionário com X_train, X_test, y_train, y_test e scaler.
    """
    df = load_raw_data(file_path)
    df_with_target = create_target_variable(df)
    df_features = engineer_features(df_with_target)
    features, target = split_features_and_target(df_features)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=target,
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Salvar dados processados
    output_path = settings.data_processed_path
    output_path.mkdir(parents=True, exist_ok=True)

    X_train_scaled.to_csv(output_path / "X_train.csv", index=False)
    X_test_scaled.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)

    print(f"Dados processados salvos em: {output_path}")
    print(f"  Treino: {len(X_train_scaled)} amostras")
    print(f"  Teste:  {len(X_test_scaled)} amostras")
    print(f"  Proporção positiva: {target.mean():.2%}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


if __name__ == "__main__":
    run_preprocessing()
