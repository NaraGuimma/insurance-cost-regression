"""Orquestra o pipeline ponta a ponta: dados -> EDA -> treino -> avaliação -> banco."""
from __future__ import annotations

import argparse

from . import download_data, eda, evaluate, load_to_db, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true", help="usa dados sintéticos (sem Kaggle)")
    ap.add_argument("--fast", action="store_true", help="treino rápido (para teste)")
    ap.add_argument("--skip-db", action="store_true", help="não carrega no banco")
    ap.add_argument("--database-url", help="sobrescreve DATABASE_URL")
    a = ap.parse_args()

    steps = [
        ("1/5 Dados", lambda: download_data.main(synthetic=a.synthetic)),
        ("2/5 EDA", eda.main),
        ("3/5 Treino", lambda: train.main(fast=a.fast)),
        ("4/5 Avaliação", evaluate.main),
    ]
    if not a.skip_db:
        steps.append(("5/5 Banco", lambda: load_to_db.main(a.database_url)))
    for title, fn in steps:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        fn()
    print("\nPipeline concluído.")


if __name__ == "__main__":
    main()
