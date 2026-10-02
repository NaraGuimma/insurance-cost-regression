"""Etapa 13 — Carga dos resultados em um banco SQL (Postgres) e criação de views para o BI."""
from __future__ import annotations

import argparse
import re

import pandas as pd
from sqlalchemy import create_engine, text

from . import config

TABLES = {
    "insurance_clean": config.PROCESSED_FILE,
    "model_comparison": config.METRICS_DIR / "model_comparison.csv",
    "predictions_test": config.METRICS_DIR / "predictions_test.csv",
    "feature_importance": config.METRICS_DIR / "feature_importance.csv",
    "segment_errors": config.METRICS_DIR / "segment_errors.csv",
}


def _split_statements(sql: str) -> list[str]:
    no_comments = re.sub(r"--.*", "", sql)
    return [s.strip() for s in no_comments.split(";") if s.strip()]


def main(database_url: str | None = None) -> None:
    url = database_url or config.DATABASE_URL
    engine = create_engine(url)
    views_sql = (config.SQL_DIR / "views.sql").read_text()
    statements = _split_statements(views_sql)
    view_names = [m.group(1) for s in statements if (m := re.match(r"CREATE VIEW (\w+)", s, re.I))]

    with engine.begin() as conn:
        # views dependem das tabelas: derruba antes de recriar as tabelas
        for v in reversed(view_names):
            conn.execute(text(f"DROP VIEW IF EXISTS {v}"))

    for table, path in TABLES.items():
        df = pd.read_csv(path)
        df.to_sql(table, engine, if_exists="replace", index=False, method="multi", chunksize=500)
        print(f"[ok] tabela {table:<20} {len(df):>6} linhas")

    with engine.begin() as conn:
        for s in statements:
            conn.execute(text(s))
    print(f"[ok] {len(view_names)} views criadas: {', '.join(view_names)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--database-url", help="sobrescreve DATABASE_URL (ex.: sqlite:///teste.db)")
    main(ap.parse_args().database_url)
