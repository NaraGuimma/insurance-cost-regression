"""Cria automaticamente no Metabase: usuário admin (1º acesso), conexão com o Postgres, os cards e o dashboard.

Pré-requisitos:  docker compose up -d   e   python -m src.run_pipeline   (tabelas/views já carregadas)
Uso:             python dashboards/create_metabase_dashboard.py

Observação: a API do Metabase evolui entre versões. Este script segue a API v0.50+ (dashcards via PUT /api/dashboard/:id).
Se algum passo falhar, use o passo a passo manual do README (as queries estão em sql/dashboard_queries.sql).
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MB_URL = os.getenv("METABASE_URL", "http://localhost:3000").rstrip("/")
EMAIL = os.getenv("METABASE_ADMIN_EMAIL", "admin@example.com")
PASSWORD = os.getenv("METABASE_ADMIN_PASSWORD", "Ch4nge-me-please-2026!")
PG_USER = os.getenv("POSTGRES_USER", "analytics")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "analytics")
PG_DB = os.getenv("POSTGRES_DB", "analytics")
DB_NAME = "Insurance Analytics"
DASH_NAME = "Custos de Seguro Saúde — Visão Executiva"

# chave (de sql/dashboard_queries.sql) -> (título, tipo de gráfico, configurações, linha, coluna, largura, altura)
# Grade do Metabase: 24 colunas.
CARDS = [
    ("kpi_customers",   "Clientes na base",             "scalar", {}, 0, 0, 4, 3),
    ("kpi_avg_charge",  "Custo médio (US$)",            "scalar", {}, 0, 4, 4, 3),
    ("kpi_pct_smokers", "% fumantes",                   "scalar", {}, 0, 8, 4, 3),
    ("kpi_r2",          "R² do modelo (teste)",         "scalar", {}, 0, 12, 4, 3),
    ("kpi_mae",         "Erro médio — MAE (US$)",       "scalar", {}, 0, 16, 4, 3),
    ("kpi_within_20",   "% previsões com erro ≤ 20%",   "scalar", {}, 0, 20, 4, 3),
    ("charges_by_smoker",     "Custo médio: fumante x não fumante", "bar",
        {"graph.dimensions": ["fumante"], "graph.metrics": ["custo_medio"]}, 3, 0, 8, 5),
    ("charges_by_age_smoker", "Custo médio por faixa etária e fumante", "bar",
        {"graph.dimensions": ["faixa_etaria", "fumante"], "graph.metrics": ["custo_medio"]}, 3, 8, 8, 5),
    ("charges_by_bmi",        "Custo médio por categoria de IMC", "bar",
        {"graph.dimensions": ["categoria_imc"], "graph.metrics": ["custo_medio"]}, 3, 16, 8, 5),
    ("feature_importance", "O que mais explica o custo? (importância %)", "row",
        {"graph.dimensions": ["variavel"], "graph.metrics": ["importancia_pct"]}, 8, 0, 8, 6),
    ("model_ranking",      "Comparação de modelos — RMSE no teste (menor é melhor)", "row",
        {"graph.dimensions": ["modelo"], "graph.metrics": ["rmse_teste"]}, 8, 8, 8, 6),
    ("charges_by_region",  "Custo médio por região", "bar",
        {"graph.dimensions": ["regiao"], "graph.metrics": ["custo_medio"]}, 8, 16, 8, 6),
    ("actual_vs_pred", "Real x Previsto (melhor modelo, conjunto de teste)", "scatter",
        {"graph.dimensions": ["real"], "graph.metrics": ["previsto"]}, 14, 0, 12, 6),
    ("residual_hist",  "Distribuição do erro (real − previsto)", "bar",
        {"graph.dimensions": ["faixa_residuo"], "graph.metrics": ["casos"]}, 14, 12, 12, 6),
    ("segment_errors", "Onde o modelo erra mais? (por segmento)", "table", {}, 20, 0, 12, 7),
    ("model_table",    "Tabela de modelos (CV e teste)", "table", {}, 20, 12, 12, 7),
]


def load_queries() -> dict[str, str]:
    text = (ROOT / "sql" / "dashboard_queries.sql").read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    return {k: re.sub(r"--.*", "", q).strip().rstrip(";") for k, q in zip(parts[1::2], parts[2::2])}


def wait_for_metabase(timeout: int = 240) -> None:
    print(f"Aguardando Metabase em {MB_URL} ...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"{MB_URL}/api/health", timeout=3).json().get("status") == "ok":
                print(" ok")
                return
        except (requests.RequestException, ValueError):
            pass
        print(".", end="", flush=True)
        time.sleep(4)
    sys.exit("\nMetabase não respondeu a tempo. Verifique: docker compose logs metabase")


def get_session() -> str:
    props = requests.get(f"{MB_URL}/api/session/properties", timeout=10).json()
    token = props.get("setup-token")
    if token:  # primeiro acesso: cria o administrador
        print("Primeiro acesso: criando usuário administrador...")
        r = requests.post(f"{MB_URL}/api/setup", timeout=60, json={
            "token": token,
            "user": {"first_name": "Admin", "last_name": "Analytics", "email": EMAIL, "password": PASSWORD},
            "prefs": {"site_name": "Insurance Analytics", "allow_tracking": False},
        })
    else:
        r = requests.post(f"{MB_URL}/api/session", json={"username": EMAIL, "password": PASSWORD}, timeout=15)
    if not r.ok:
        sys.exit(f"Falha na autenticação/setup ({r.status_code}): {r.text}\n"
                 "Se o Metabase já foi configurado com outro usuário, ajuste METABASE_ADMIN_EMAIL/PASSWORD no .env.")
    return r.json()["id"]


def ensure_database(h: dict) -> int:
    r = requests.get(f"{MB_URL}/api/database", headers=h, timeout=15)
    r.raise_for_status()
    data = r.json()
    for db in (data["data"] if isinstance(data, dict) else data):
        if db["name"] == DB_NAME:
            return db["id"]
    print("Registrando o banco Postgres no Metabase...")
    r = requests.post(f"{MB_URL}/api/database", headers=h, timeout=60, json={
        "engine": "postgres", "name": DB_NAME,
        "details": {"host": "postgres", "port": 5432, "dbname": PG_DB, "user": PG_USER, "password": PG_PASS, "ssl": False},
        "is_full_sync": True,
    })
    if not r.ok:
        sys.exit(f"Falha ao registrar o banco ({r.status_code}): {r.text}")
    return r.json()["id"]


def main() -> None:
    wait_for_metabase()
    h = {"X-Metabase-Session": get_session()}

    existing = requests.get(f"{MB_URL}/api/dashboard", headers=h, timeout=15).json()
    if any(d["name"] == DASH_NAME for d in existing):
        sys.exit(f"O dashboard '{DASH_NAME}' já existe. Apague-o no Metabase para recriar.")

    db_id = ensure_database(h)
    queries = load_queries()

    dash = requests.post(f"{MB_URL}/api/dashboard", headers=h, timeout=15, json={
        "name": DASH_NAME,
        "description": "Quanto custa um cliente, o que explica o custo e quão confiável é o nosso modelo de previsão.",
    })
    dash.raise_for_status()
    dash_id = dash.json()["id"]

    dashcards = []
    for i, (key, title, display, viz, row, col, w, hgt) in enumerate(CARDS):
        r = requests.post(f"{MB_URL}/api/card", headers=h, timeout=60, json={
            "name": title, "display": display, "visualization_settings": viz,
            "dataset_query": {"type": "native", "database": db_id, "native": {"query": queries[key]}},
        })
        if not r.ok:
            sys.exit(f"Falha ao criar o card '{title}' ({r.status_code}): {r.text}")
        dashcards.append({"id": -(i + 1), "card_id": r.json()["id"], "row": row, "col": col,
                          "size_x": w, "size_y": hgt, "series": [], "visualization_settings": {}, "parameter_mappings": []})
        print(f"  card criado: {title}")

    r = requests.put(f"{MB_URL}/api/dashboard/{dash_id}", headers=h, timeout=60, json={"dashcards": dashcards})
    if not r.ok:
        sys.exit(f"Falha ao montar o dashboard ({r.status_code}): {r.text}")
    print(f"\nDashboard pronto: {MB_URL}/dashboard/{dash_id}\nLogin: {EMAIL}")


if __name__ == "__main__":
    main()
