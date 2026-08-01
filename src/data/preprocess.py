"""Módulo de pré-processamento dos dados de clickstream do e-commerce."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import settings

_COLUMN_RENAMES = {
    "order": "click_order",
    "page 1 (main category)": "main_category",
    "location": "photo_location",
    "model photography": "model_photography",
    "price 2": "price_above_avg",
    "page": "page_number",
}


def load_raw_data(file_path: Path | None = None) -> pd.DataFrame:
    """Carrega o dataset bruto (CSV separado por ';')."""
    path = file_path or settings.data_raw_path
    return pd.read_csv(path, sep=";")


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Cria 'purchased' e remove clicks da página sale (leakage)."""
    cat_col = "page 1 (main category)"
    sale_sessions = df[df[cat_col] == 4]["session ID"].unique()
    df = df.copy()
    df["purchased"] = df["session ID"].isin(sale_sessions).astype(int)
    return df[df[cat_col] != 4].copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas identificadoras e renomeia para nomes descritivos."""
    df_clean = df.drop(columns=["year", "session ID", "page 2 (clothing model)"])
    return df_clean.rename(columns=_COLUMN_RENAMES)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features da variável alvo (purchased)."""
    return df.drop(columns=["purchased"]), df["purchased"]


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Aplica StandardScaler nas features."""
    scaler = StandardScaler()
    cols, idx_tr, idx_te = X_train.columns, X_train.index, X_test.index
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=idx_tr)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=cols, index=idx_te)
    return X_tr, X_te, scaler


def _save_splits(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series) -> None:
    """Persiste dados processados em CSV."""
    out = settings.data_processed_path
    out.mkdir(parents=True, exist_ok=True)
    X_tr.to_csv(out / "X_train.csv", index=False)
    X_te.to_csv(out / "X_test.csv", index=False)
    y_tr.to_csv(out / "y_train.csv", index=False)
    y_te.to_csv(out / "y_test.csv", index=False)


def _split_data(features: pd.DataFrame, target: pd.Series) -> tuple:
    """Divide em treino/teste com estratificação."""
    return train_test_split(
        features,
        target,
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=target,
    )


def run_preprocessing(file_path: Path | None = None) -> dict:
    """Executa pipeline: load → target → features → split → scale → save."""
    df = load_raw_data(file_path)
    df = create_target_variable(df)
    df = engineer_features(df)
    features, target = split_features_and_target(df)

    X_train, X_test, y_train, y_test = _split_data(features, target)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    _save_splits(X_train, X_test, y_train, y_test)
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


if __name__ == "__main__":
    run_preprocessing()
