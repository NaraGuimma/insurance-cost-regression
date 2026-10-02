.PHONY: install data data-synthetic eda train evaluate load all up down dashboard clean

install:            ## Instala dependências Python
	pip install -r requirements.txt

data:               ## Baixa o dataset real do Kaggle
	python -m src.download_data

data-synthetic:     ## Gera dataset sintético (sem Kaggle)
	python -m src.download_data --synthetic

eda:                ## Análise exploratória + gráficos
	python -m src.eda

train:              ## Treina, valida (CV), ajusta hiperparâmetros e avalia no teste
	python -m src.train

evaluate:           ## Diagnóstico do melhor modelo, importâncias e erros por segmento
	python -m src.evaluate

load:               ## Carrega resultados no Postgres e cria views
	python -m src.load_to_db

all:                ## Pipeline completo (dados reais)
	python -m src.run_pipeline

up:                 ## Sobe Postgres + Metabase
	docker compose up -d

down:               ## Derruba containers (mantém volumes)
	docker compose down

dashboard:          ## Cria o dashboard no Metabase via API
	python dashboards/create_metabase_dashboard.py

clean:              ## Remove artefatos gerados
	rm -f reports/figures/*.png reports/metrics/* models/*.joblib data/processed/*.csv
