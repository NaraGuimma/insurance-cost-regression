"""Configurações centrais do projeto (caminhos, constantes, colunas, conexão com o banco)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # .env é opcional
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------- Caminhos
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
SQL_DIR = ROOT / "sql"

RAW_FILE = DATA_RAW / "insurance.csv"
PROCESSED_FILE = DATA_PROCESSED / "insurance_features.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"

# ---------------------------------------------------------------- Dataset
KAGGLE_DATASET = "mirichoi0218/insurance"  # Medical Cost Personal Datasets
TARGET = "charges"

RAW_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
NUMERIC_FEATURES = ["age", "bmi", "children", "smoker_obese"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region", "bmi_category", "age_group"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------- Modelagem
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_MODELS_TO_TUNE = 3      # quantos dos melhores modelos (com grade definida) serão ajustados
N_ITER_SEARCH = 25        # iterações do RandomizedSearchCV

# ---------------------------------------------------------------- Banco de dados
_user = os.getenv("POSTGRES_USER", "analytics")
_pwd = os.getenv("POSTGRES_PASSWORD", "analytics")
_db = os.getenv("POSTGRES_DB", "analytics")
_port = os.getenv("POSTGRES_PORT", "5433")
DATABASE_URL = os.getenv(
    "DATABASE_URL", f"postgresql+psycopg2://{_user}:{_pwd}@localhost:{_port}/{_db}"
)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
