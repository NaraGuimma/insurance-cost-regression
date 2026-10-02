"""Etapa 1 — Aquisição dos dados.

Baixa o dataset "Medical Cost Personal Datasets" do Kaggle (mirichoi0218/insurance) usando
`kagglehub` e copia `insurance.csv` para data/raw/. Se o Kaggle não estiver acessível, use
`--synthetic` para gerar um dataset sintético com o MESMO esquema (útil para testar o pipeline).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def generate_synthetic(n: int = 1338, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Gera dados sintéticos com a mesma estrutura e relações aproximadas do dataset real."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, n)
    sex = rng.choice(["female", "male"], n)
    bmi = np.clip(rng.normal(30.6, 6.1, n), 15.9, 53.1).round(2)
    children = np.clip(rng.poisson(1.0, n), 0, 5)
    smoker = rng.choice(["yes", "no"], n, p=[0.2, 0.8])
    region = rng.choice(["northeast", "northwest", "southeast", "southwest"], n)
    is_smoker = (smoker == "yes").astype(int)
    charges = (
        -2500
        + 260 * age
        + 330 * bmi
        + 480 * children
        + 12000 * is_smoker
        + 14000 * is_smoker * (bmi >= 30)
        + 150 * (region == "southeast")
    )
    charges = charges * rng.lognormal(0, 0.16, n)
    charges = np.clip(charges, 1100, None).round(2)
    return pd.DataFrame(
        dict(age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region, charges=charges)
    )


def download_from_kaggle() -> Path:
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Instale o kagglehub: pip install kagglehub") from exc

    print(f"Baixando '{config.KAGGLE_DATASET}' do Kaggle...")
    try:
        path = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    except Exception as exc:
        raise SystemExit(
            f"Falha ao baixar do Kaggle: {exc}\n"
            "Opções: (1) configure suas credenciais (KAGGLE_USERNAME/KAGGLE_KEY ou ~/.kaggle/kaggle.json);\n"
            "(2) baixe manualmente em https://www.kaggle.com/datasets/mirichoi0218/insurance e salve como "
            f"{config.RAW_FILE};\n(3) rode com --synthetic para testar o pipeline."
        ) from exc
    csv = next(path.rglob("insurance.csv"), None)
    if csv is None:
        raise SystemExit(f"insurance.csv não encontrado em {path}")
    return csv


def main(synthetic: bool = False, force: bool = False) -> Path:
    config.ensure_dirs()
    if config.RAW_FILE.exists() and not force:
        print(f"[ok] {config.RAW_FILE} já existe (use --force para baixar novamente).")
        return config.RAW_FILE

    if synthetic:
        df = generate_synthetic()
        df.to_csv(config.RAW_FILE, index=False)
        print(f"[ok] Dataset SINTÉTICO salvo em {config.RAW_FILE} ({len(df)} linhas).")
    else:
        shutil.copy(download_from_kaggle(), config.RAW_FILE)
        print(f"[ok] Dataset salvo em {config.RAW_FILE}")
    return config.RAW_FILE


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true", help="gera dados sintéticos em vez de baixar do Kaggle")
    ap.add_argument("--force", action="store_true", help="sobrescreve o arquivo existente")
    a = ap.parse_args()
    main(synthetic=a.synthetic, force=a.force)
