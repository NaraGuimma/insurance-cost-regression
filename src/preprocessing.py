"""Etapas 2, 4, 5 e 6 — Limpeza, engenharia de atributos, split e pré-processamento."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_raw(path=config.RAW_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode: python -m src.download_data")
    return pd.read_csv(path)


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Padroniza nomes/strings, valida esquema, remove duplicatas e valida valores."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    missing_cols = set(config.RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colunas ausentes no dataset: {sorted(missing_cols)}")
    df = df[config.RAW_COLUMNS]

    for col in ("sex", "smoker", "region"):
        df[col] = df[col].astype(str).str.strip().str.lower()

    n_before = len(df)
    n_nulls = int(df.isna().sum().sum())
    n_dups = int(df.dropna().duplicated().sum())
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Regras de sanidade (valores impossíveis indicariam erro de coleta)
    assert (df["age"] > 0).all() and (df["bmi"] > 0).all(), "age/bmi com valores inválidos"
    assert (df["children"] >= 0).all(), "children negativo"
    assert (df["charges"] > 0).all(), "charges <= 0 (quebraria log e MAPE)"

    if verbose:
        print(f"[clean] linhas: {n_before} -> {len(df)} | células nulas: {n_nulls} | duplicatas removidas: {n_dups}")
    df.index.name = "row_id"
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering determinística (sem estatísticas aprendidas => sem vazamento de dados)."""
    df = df.copy()
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, np.inf], labels=["underweight", "normal", "overweight", "obese"], right=False
    ).astype(str)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 40, 50, np.inf], labels=["18-29", "30-39", "40-49", "50+"], right=False
    ).astype(str)
    # Interação conhecida: fumante + obeso é o grupo de maior custo
    df["smoker_obese"] = ((df["smoker"] == "yes") & (df["bmi"] >= 30)).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Padroniza numéricas e aplica one-hot nas categóricas. Ajustado SÓ no treino (dentro do Pipeline)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def split_data(df: pd.DataFrame):
    """Split treino/teste estratificado por `smoker` (principal driver do custo)."""
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df["smoker"]
    )
    return train_df, test_df
