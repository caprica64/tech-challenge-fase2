"""Testes unitários para o módulo de pré-processamento."""

import pandas as pd
import pytest

from src.data.preprocess import (
    create_target_variable,
    engineer_features,
    split_features_and_target,
)


@pytest.fixture
def sample_clickstream() -> pd.DataFrame:
    """Cria um DataFrame de clickstream simulado para testes."""
    return pd.DataFrame(
        {
            "year": [2008, 2008, 2008, 2008, 2008],
            "month": [4, 4, 4, 4, 4],
            "day": [1, 1, 1, 1, 2],
            "order": [1, 2, 3, 1, 1],
            "country": [29, 29, 29, 15, 15],
            "session ID": [1, 1, 1, 2, 2],
            "page 1 (main category)": [1, 2, 4, 1, 3],
            "page 2 (clothing model)": ["A1", "B2", "C3", "A1", "B2"],
            "colour": [1, 2, 3, 1, 5],
            "location": [1, 2, 3, 4, 5],
            "model photography": [1, 2, 1, 1, 2],
            "price": [28, 35, 50, 42, 30],
            "price 2": [1, 2, 1, 2, 1],
            "page": [1, 1, 2, 1, 1],
        }
    )


def test_create_target_variable(sample_clickstream: pd.DataFrame) -> None:
    """Verifica criação do target e remoção de leakage."""
    result = create_target_variable(sample_clickstream)

    assert "purchased" in result.columns
    assert 4 not in result["page 1 (main category)"].values

    session_1 = result[result["session ID"] == 1]
    session_2 = result[result["session ID"] == 2]
    assert (session_1["purchased"] == 1).all()
    assert (session_2["purchased"] == 0).all()


def test_engineer_features(sample_clickstream: pd.DataFrame) -> None:
    """Verifica renomeação e remoção de colunas desnecessárias."""
    df_with_target = create_target_variable(sample_clickstream)
    result = engineer_features(df_with_target)

    assert "year" not in result.columns
    assert "session ID" not in result.columns
    assert "click_order" in result.columns
    assert "main_category" in result.columns
    assert "purchased" in result.columns


def test_split_features_and_target(sample_clickstream: pd.DataFrame) -> None:
    """Verifica separação correta de features e target."""
    df_with_target = create_target_variable(sample_clickstream)
    df_features = engineer_features(df_with_target)
    features, target = split_features_and_target(df_features)

    assert "purchased" not in features.columns
    assert target.name == "purchased"
    assert len(features) == len(target)
