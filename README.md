# 🏥 Previsão de Custos de Seguro Saúde — Projeto de Regressão ponta a ponta

Projeto completo de análise de dados: **aquisição → limpeza → EDA → engenharia de atributos → comparação de modelos de regressão → métricas → diagnóstico → banco de dados → dashboard executivo (Metabase)**.

> **Problema de negócio:** uma operadora de saúde quer (1) entender **o que faz o custo de um beneficiário subir**, (2) **prever** esse custo para precificar planos e provisionar reservas, e (3) saber **o quanto confiar** nessas previsões.

---

## Sumário
1. [Visão geral e arquitetura](#1-visão-geral-e-arquitetura)
2. [Dataset](#2-dataset)
3. [Estrutura do repositório](#3-estrutura-do-repositório)
4. [Como executar](#4-como-executar)
5. [Etapas do pipeline, passo a passo](#5-etapas-do-pipeline-passo-a-passo)
6. [Modelos testados](#6-modelos-testados)
7. [Métricas de avaliação (e como explicá-las)](#7-métricas-de-avaliação)
8. [Artefatos gerados](#8-artefatos-gerados)
9. [Dashboard para a liderança (Metabase)](#9-dashboard-para-a-liderança-metabase)
10. [Alternativa: Apache Superset](#10-alternativa-apache-superset)
11. [Roteiro de apresentação para a liderança](#11-roteiro-de-apresentação-para-a-liderança)
12. [Decisões de projeto e boas práticas](#12-decisões-de-projeto-e-boas-práticas)
13. [Solução de problemas](#13-solução-de-problemas)
14. [Próximos passos](#14-próximos-passos)

---

## 1. Visão geral e arquitetura

```
 Kaggle ──► data/raw ──► limpeza + features ──► split treino/teste
                                                     │
                      ┌──────────────────────────────┤
                      ▼                              ▼
                    EDA (gráficos)        Validação cruzada (12 modelos)
                                                     │
                                          Tuning dos melhores (RandomizedSearchCV)
                                                     │
                                          Teste final (1x) + escolha do melhor
                                                     │
                       diagnóstico · importâncias · erro por segmento
                                                     │
                      CSVs em reports/metrics ──► Postgres (tabelas + views)
                                                     │
                                                 Metabase ──► Dashboard executivo
```

**Stack:** Python 3.10+ · pandas · scikit-learn · seaborn/matplotlib · SQLAlchemy · PostgreSQL · Metabase (Docker).

---

## 2. Dataset

**[Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)** (`mirichoi0218/insurance`) — 1.338 beneficiários de seguro saúde nos EUA.

| Coluna | Tipo | Descrição |
|---|---|---|
| `age` | int | Idade do beneficiário principal |
| `sex` | cat | `male` / `female` |
| `bmi` | float | Índice de massa corporal (kg/m²) |
| `children` | int | Nº de dependentes cobertos |
| `smoker` | cat | `yes` / `no` |
| `region` | cat | `northeast`, `northwest`, `southeast`, `southwest` |
| **`charges`** | float | **Alvo:** custo médico anual (US$) |

Por que este dataset? É pequeno (roda em segundos), limpo, tem variáveis numéricas e categóricas, uma **interação forte e explicável** (fumante × obesidade) e um alvo assimétrico — ótimo para mostrar *por que* escolher e avaliar modelos com critério. Os conceitos se transferem para bases maiores (inclusive internas).

> Sem acesso ao Kaggle? `python -m src.download_data --synthetic` gera dados **sintéticos** com o mesmo esquema para você testar o pipeline e o dashboard. **Não use esses dados para tirar conclusões de negócio.**

---

## 3. Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── Makefile                    # atalhos: make all, make up, make dashboard...
├── docker-compose.yml          # Postgres (dados analíticos) + Metabase
├── .env.example                # credenciais locais (copie para .env)
├── data/
│   ├── raw/                    # insurance.csv baixado (ignorado no git)
│   └── processed/              # dataset com features + coluna split (train/test)
├── src/
│   ├── config.py               # caminhos, constantes, colunas, conexão com o banco
│   ├── download_data.py        # Etapa 1: Kaggle (ou dados sintéticos)
│   ├── preprocessing.py        # Etapas 2, 4, 5, 6: limpeza, features, split, ColumnTransformer
│   ├── eda.py                  # Etapa 3: análise exploratória
│   ├── metrics.py              # Etapa 10: métricas de regressão
│   ├── train.py                # Etapas 7–10: CV, tuning, teste, seleção do melhor
│   ├── evaluate.py             # Etapas 11–12: resíduos, importância, erros por segmento
│   ├── load_to_db.py           # Etapa 13: carga no Postgres + views
│   └── run_pipeline.py         # Orquestra tudo
├── sql/
│   ├── views.sql               # views usadas pelo dashboard
│   └── dashboard_queries.sql   # queries de cada card (Metabase/Superset)
├── dashboards/
│   └── create_metabase_dashboard.py   # cria o dashboard automaticamente via API
├── models/                     # best_model.joblib (pipeline completo serializado)
└── reports/
    ├── figures/                # PNGs da EDA e da avaliação
    └── metrics/                # CSV/JSON com todos os resultados
```

---

## 4. Como executar

### Pré-requisitos
- Python 3.10+
- Docker + Docker Compose (para Postgres/Metabase)
- (Opcional) conta no Kaggle — datasets públicos normalmente baixam com `kagglehub`; se pedir autenticação, defina `KAGGLE_USERNAME` e `KAGGLE_KEY` ou use `~/.kaggle/kaggle.json`.

### Passo a passo (dados reais)

```bash
# 1) Ambiente
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 2) Infra do dashboard (Postgres + Metabase)
docker compose up -d

# 3) Pipeline completo: download → EDA → treino → avaliação → carga no banco
python -m src.run_pipeline

# 4) Dashboard no Metabase (cria banco, cards e dashboard via API)
python dashboards/create_metabase_dashboard.py
# abra http://localhost:3000
```

Ou com `make`: `make install && make up && make all && make dashboard`.

### Variações úteis

| Comando | O que faz |
|---|---|
| `python -m src.run_pipeline --synthetic` | Usa dados sintéticos (sem Kaggle) |
| `python -m src.run_pipeline --fast` | Treino rápido (menos árvores/iterações) para testes |
| `python -m src.run_pipeline --skip-db` | Roda só a parte de ciência de dados |
| `python -m src.run_pipeline --database-url sqlite:///teste.db` | Carrega em SQLite (útil para testar sem Docker) |
| `python -m src.train` / `src.evaluate` / `src.eda` | Roda etapas isoladas |

---

## 5. Etapas do pipeline, passo a passo

### Etapa 1 — Aquisição (`src/download_data.py`)
Usa `kagglehub` para baixar `insurance.csv` e copiá-lo para `data/raw/`. O script é idempotente (não baixa de novo se o arquivo existir; `--force` sobrescreve) e orienta o usuário se o download falhar.

### Etapa 2 — Limpeza e validação (`preprocessing.clean`)
- Padroniza nomes de colunas e textos (minúsculas, sem espaços).
- **Valida o esquema** (falha cedo se faltar coluna).
- Remove nulos e **duplicatas** (o dataset original tem 1 linha duplicada).
- **Asserts de sanidade:** idade/IMC > 0, filhos ≥ 0, custo > 0 (necessário para `log` e MAPE).
- Define `row_id` como índice, que acompanha cada linha até o banco/dashboard.

### Etapa 3 — Análise exploratória (`src/eda.py`)
Gera 6 figuras em `reports/figures/` e `eda_insights.json`:

| Figura | Pergunta que responde |
|---|---|
| `01_target_distribution` | O alvo é simétrico? (não: cauda longa à direita → `log` ajuda) |
| `02_charges_by_smoker` | Fumar muda o custo? (muito) |
| `03_age_vs_charges`, `04_bmi_vs_charges` | Como idade/IMC se relacionam com custo, separando fumantes? (faixas distintas; efeito do IMC aparece em IMC ≥ 30 entre fumantes) |
| `05_correlation_heatmap` | Quais variáveis se correlacionam linearmente com o alvo? |
| `06_categorical_means` | Região, sexo, faixa de IMC e idade fazem diferença? |

Também executa um **teste t de Welch** (fumantes × não fumantes) e conta outliers por IQR. Os outliers são **mantidos**: são casos reais de alto custo, exatamente o que a operadora precisa prever.

### Etapa 4 — Engenharia de atributos (`preprocessing.add_features`)
| Feature | Lógica | Motivo |
|---|---|---|
| `bmi_category` | OMS: <18,5 / 18,5–25 / 25–30 / ≥30 | Efeito não linear do IMC |
| `age_group` | 18–29 / 30–39 / 40–49 / 50+ | Captura degraus de risco por idade |
| `smoker_obese` | fumante **e** IMC ≥ 30 | Interação identificada na EDA; ajuda modelos lineares |

Transformações **determinísticas** (sem estatísticas aprendidas dos dados) → seguras antes do split, sem vazamento.

### Etapa 5 — Split treino/teste (`preprocessing.split_data`)
80/20 com `random_state=42`, **estratificado por `smoker`** para manter a mesma proporção de fumantes nos dois conjuntos. O **teste fica "trancado"**: só é usado para o relatório final. O arquivo `data/processed/insurance_features.csv` marca cada linha com `split`.

### Etapa 6 — Pré-processamento (`preprocessing.build_preprocessor`)
`ColumnTransformer` com `StandardScaler` (numéricas) e `OneHotEncoder(handle_unknown="ignore")` (categóricas), **dentro de um `Pipeline`** junto com o modelo. Assim, em cada fold da validação cruzada o pré-processador é ajustado **apenas nos dados de treino do fold** → **sem data leakage**.

### Etapa 7 — Baseline e modelos (`train.get_models`)
Sempre compare com um **baseline** (`DummyRegressor` que prevê a média). Se um modelo não bate o baseline com folga, ele não agrega valor. Veja a lista na [seção 6](#6-modelos-testados).

### Etapa 8 — Validação cruzada (`train.cv_evaluate`)
`KFold(5, shuffle=True)` **somente no treino**. Para cada modelo registramos RMSE, MAE e R² (média ± desvio) e o RMSE **de treino** — a diferença `cv_rmse − train_rmse` (`overfit_gap`) indica **overfitting**.

### Etapa 9 — Ajuste de hiperparâmetros (`RandomizedSearchCV`)
Os **3 melhores** modelos que possuem espaço de busca (Ridge, KNN, Random Forest, Gradient Boosting, HistGradientBoosting) são ajustados com 25 amostragens aleatórias, otimizando RMSE com os mesmos folds. O resultado entra na comparação como `<modelo> (tuned)`, ao lado da versão padrão.

### Etapa 10 — Avaliação final e seleção (`train.main`)
Todos os candidatos são re-treinados no treino completo e **avaliados uma única vez no teste**. O **melhor modelo é escolhido pelo RMSE de validação cruzada, não pelo teste** — escolher pelo teste o "contaminaria" (você otimizaria para ele). O teste serve para estimar o desempenho em produção. Saídas:
`model_comparison.csv`, `predictions_test.csv`, `best_model.json` e `models/best_model.joblib` (pipeline completo, pronto para `.predict()` em dados brutos com as features).

### Etapa 11 — Diagnóstico (`src/evaluate.py`)
| Figura | O que observar |
|---|---|
| `10_model_comparison` | RMSE CV vs. teste de todos os modelos (barras de erro = ±1 desvio). CV e teste próximos = estimativa confiável |
| `11_actual_vs_predicted` | Pontos perto da diagonal = boas previsões; leques indicam heterocedasticidade |
| `12_residual_diagnostics` | Resíduos × previsto (sem padrão/funil), histograma (centrado em 0), QQ-plot (caudas pesadas = erros extremos) |
| `13_learning_curve` | Curvas convergindo = mais dados ajudam pouco; grande distância = alta variância |

### Etapa 12 — Interpretabilidade e erros por segmento
- **Importância por permutação** (`14_permutation_importance`): quanto o RMSE **piora** ao embaralhar cada variável no teste. É agnóstica ao modelo e fácil de explicar.
- **Erro por segmento** (`segment_errors.csv`, `15_error_by_segment`): MAE, RMSE, MAPE e viés por fumante, faixa etária, IMC, região e sexo. Responde: *"em quais clientes o modelo é menos confiável?"* — informação essencial para decisões de precificação.

### Etapa 13 — Carga no banco (`src/load_to_db.py`)
Carrega 5 tabelas no Postgres e cria 6 views (`sql/views.sql`):

| Tabela | Conteúdo |
|---|---|
| `insurance_clean` | Dados limpos + features + `split` |
| `model_comparison` | Métricas de CV e teste de cada modelo |
| `predictions_test` | Real × previsto de cada modelo (formato longo) + atributos do cliente |
| `feature_importance` | Importância por permutação |
| `segment_errors` | Erros por segmento |

As views (`vw_best_model`, `vw_predictions_best`, `vw_charges_by_*`) isolam a lógica de negócio do BI: se o modelo campeão mudar, **o dashboard se atualiza sozinho** após rodar o pipeline.

### Etapa 14 — Dashboard
Veja a [seção 9](#9-dashboard-para-a-liderança-metabase).

---

## 6. Modelos testados

| Modelo | Por que está aqui |
|---|---|
| Baseline (média) | Piso de comparação |
| Regressão Linear | Referência simples e interpretável |
| Regressão Linear / Ridge / Lasso / ElasticNet **(log)** | Treinam em `log(1+charges)` (via `TransformedTargetRegressor`) para lidar com a assimetria do alvo; previsões voltam à escala original em US$. Ridge/Lasso/ElasticNet adicionam regularização |
| KNN | Modelo baseado em vizinhança (exige features escaladas) |
| Árvore de Decisão | Captura não linearidades/interações; regras legíveis |
| Random Forest | Ensemble de árvores (bagging); robusto |
| Gradient Boosting / HistGradientBoosting | Ensembles de boosting; geralmente os mais precisos em dados tabulares |

---

## 7. Métricas de avaliação

| Métrica | Fórmula/ideia | Como explicar para o negócio | Observações |
|---|---|---|---|
| **MAE** | média de \|real − previsto\| | "Em média, erramos US$ X por cliente" | Robusta a outliers; fácil de comunicar |
| **RMSE** | √média(erro²) | "Erro típico, penalizando mais os erros grandes" | Mesma unidade do alvo; **métrica usada para escolher o modelo** |
| **MSE** | média(erro²) | — | Base do RMSE; unidade ao quadrado |
| **R²** | 1 − SQres/SQtot | "O modelo explica X% da variação dos custos" | 0 = igual à média; pode ser negativo |
| **R² ajustado** | R² penalizado pelo nº de variáveis | — | Útil para comparar modelos com nº de features distinto |
| **MAPE** | média(\|erro\|/real) | "Erramos X% em média" | Distorce com valores reais muito pequenos |
| **MedAE** | mediana de \|erro\| | "Caso típico, ignorando extremos" | Complementa o MAE |
| **Viés (bias)** | média(previsto − real) | "Superestimamos/subestimamos em média?" | Positivo = superestima (risco de preço alto); negativo = sub-provisiona |
| **% dentro de ±10% / ±20%** | proporção de previsões com erro relativo ≤ limite | "Em X% dos clientes acertamos em até 20%" | **Métrica de negócio** — a mais intuitiva para a liderança |

**Como ler os resultados:**
- `test_r2` ≈ `cv_r2_mean` → o desempenho deve generalizar.
- `overfit_gap` alto (ex.: Random Forest/KNN) → modelo decorou o treino.
- Compare **sempre** com o baseline.
- Olhe os **erros por segmento**: um bom R² global pode esconder um MAE alto em fumantes.

> Referência (não é garantia): com o dataset real, modelos de boosting/ensemble costumam alcançar R² entre ~0,85 e 0,90 no teste, e a regressão linear simples fica um pouco abaixo. Os números exatos dependem do split e da versão das bibliotecas — **confie nos resultados gerados na sua execução** (`reports/metrics/model_comparison.csv`).

---

## 8. Artefatos gerados

| Arquivo | Descrição |
|---|---|
| `reports/metrics/model_comparison.csv` | Ranking de todos os modelos (CV + teste + melhores hiperparâmetros) |
| `reports/metrics/best_model.json` | Modelo vencedor, parâmetros, métricas, versão do sklearn, data de treino |
| `reports/metrics/predictions_test.csv` | Previsões de todos os modelos no teste |
| `reports/metrics/feature_importance.csv` | Importância por permutação |
| `reports/metrics/segment_errors.csv` | Erros por segmento |
| `reports/metrics/eda_summary.csv` / `eda_insights.json` | Estatísticas e achados da EDA |
| `reports/figures/*.png` | 12 gráficos (EDA + avaliação) |
| `models/best_model.joblib` | Pipeline treinado (pré-processamento + modelo) |

Uso do modelo salvo:

```python
import joblib, pandas as pd
from src import config
from src.preprocessing import add_features

model = joblib.load("models/best_model.joblib")
novo = pd.DataFrame([{"age": 45, "sex": "male", "bmi": 32.5, "children": 2, "smoker": "yes", "region": "southeast"}])
novo = add_features(novo)                       # cria bmi_category, age_group e smoker_obese
print(model.predict(novo[config.FEATURES]))     # custo anual previsto (US$)
```
> Use a mesma versão do `scikit-learn` registrada em `reports/metrics/best_model.json`.

---

## 9. Dashboard para a liderança (Metabase)

### 9.1 Automático (recomendado)
```bash
docker compose up -d                       # Postgres + Metabase
python -m src.run_pipeline                 # popula o banco
python dashboards/create_metabase_dashboard.py
```
O script: espera o Metabase subir → cria o administrador no primeiro acesso (credenciais do `.env`) → registra o Postgres → cria **16 cards** a partir de `sql/dashboard_queries.sql` → monta o dashboard **"Custos de Seguro Saúde — Visão Executiva"**. Ao final imprime a URL.

> ⚠️ A API do Metabase muda entre versões e este script foi escrito para a API recente (v0.50+) — ele não foi executado contra uma instância real durante a criação deste projeto. Se algum passo falhar, a mensagem de erro indica qual; use o modo manual abaixo.

### 9.2 Manual (passo a passo)
1. Abra http://localhost:3000 e conclua o assistente inicial (crie o usuário admin).
2. **Add your data** → PostgreSQL → `Host: postgres` · `Port: 5432` · `Database/User/Password`: os do `.env` (padrão `analytics`). *O host é `postgres` (nome do serviço Docker), não `localhost`.*
3. **+ New → SQL query**, escolha o banco, cole uma query de `sql/dashboard_queries.sql`, execute, **Visualization** → escolha o gráfico e **Save**.
4. **+ New → Dashboard** e adicione os cards.

### 9.3 Layout do dashboard

| Faixa | Cards | Mensagem |
|---|---|---|
| **KPIs** | Clientes · Custo médio · % fumantes · R² · MAE · % previsões com erro ≤ 20% | Retrato da carteira + confiabilidade do modelo |
| **Drivers de custo** | Fumante × não fumante · Faixa etária × fumante · Categoria de IMC | O que encarece o cliente |
| **Modelo** | Importância das variáveis · Ranking de modelos (RMSE) · Custo por região | Por que o modelo prevê o que prevê; qual foi escolhido |
| **Confiança** | Real × Previsto · Distribuição do erro | O modelo acerta? Há viés? |
| **Risco** | Erro por segmento · Tabela de modelos | Onde ter cautela; transparência na comparação |

Dicas: adicione filtros (região, faixa etária) mapeados às colunas das views; ative **auto-refresh** em modo apresentação; use *Share → Public link* ou assine por e-mail (*Subscriptions*) para a liderança.

---

## 10. Alternativa: Apache Superset

As tabelas/views no Postgres e as queries de `sql/dashboard_queries.sql` funcionam igualmente no Superset:

1. Suba o Superset seguindo a documentação oficial (por exemplo, `git clone https://github.com/apache/superset` e `docker compose -f docker-compose-non-dev.yml up`); acesse em http://localhost:8088.
2. **Settings → Database Connections → + Database → PostgreSQL**. URI: `postgresql://analytics:analytics@host.docker.internal:5433/analytics` (do container para o Postgres do host; em Linux, pode ser necessário usar o IP do host ou colocar ambos na mesma rede Docker).
3. **SQL Lab**: execute cada query de `dashboard_queries.sql` → **Create chart** (Big Number, Bar Chart, Scatter, Table) → salve.
4. **Dashboards → + Dashboard**: arraste os charts seguindo o layout da seção 9.3.

---

## 11. Roteiro de apresentação para a liderança

1. **Contexto (30s):** "Queremos prever o custo anual por beneficiário para precificar e provisionar."
2. **KPIs:** tamanho da carteira, custo médio, % de fumantes.
3. **O que encarece:** fumantes custam várias vezes mais; fumante + obeso é o grupo mais caro; custo cresce com a idade.
4. **Quão bom é o modelo:** R², MAE em US$ e "% das previsões com erro ≤ 20%" (traduz estatística em decisão). Compare com o baseline.
5. **Onde cautela:** erro por segmento (tipicamente maior nos segmentos de custo mais alto/variável).
6. **Recomendações / decisões:** ex.: usar o modelo para triagem/precificação base com margem de segurança nos segmentos de maior erro; investir em dados adicionais (histórico de sinistros, doenças crônicas) para melhorar os segmentos críticos.

---

## 12. Decisões de projeto e boas práticas

- **Sem vazamento de dados:** todo pré-processamento aprendido está dentro do `Pipeline` e é ajustado por fold.
- **Teste "trancado":** só avalia; não participa de escolha de modelo nem de tuning.
- **Seleção por CV**, não pelo teste.
- **Reprodutibilidade:** `random_state=42`, dependências em `requirements.txt`, versão do sklearn registrada no artefato.
- **Alvo assimétrico:** testamos o alvo em escala original **e** em `log`.
- **Métricas técnicas + de negócio** (±10%/±20%) e **análise por segmento**.
- **BI desacoplado:** o dashboard lê views no banco; reexecutar o pipeline atualiza tudo.
- **Limitações:** dataset pequeno (1.338 linhas), sem histórico temporal nem diagnósticos clínicos; relação estatística ≠ causalidade; dados de demonstração — não usar diretamente para precificar.

---

## 13. Solução de problemas

| Sintoma | Causa provável / solução |
|---|---|
| `Falha ao baixar do Kaggle` | Configure credenciais, baixe manualmente para `data/raw/insurance.csv` ou use `--synthetic` |
| `connection refused` ao carregar no banco | `docker compose up -d` e aguarde o healthcheck; confira `POSTGRES_PORT` (padrão **5433** no host) |
| Porta 5433 ou 3000 ocupada | Altere a porta no `.env` (Postgres) ou em `docker-compose.yml` (Metabase) |
| Metabase não conecta ao banco | Use `Host: postgres` / `Port: 5432` (rede interna do Docker) |
| `cannot drop table ... other objects depend on it` | O loader já derruba as views antes; se criou views manualmente, derrube-as |
| Erro de senha ao criar admin no Metabase | A senha precisa ser "forte" — ajuste `METABASE_ADMIN_PASSWORD` |
| Script do Metabase falha em `/api/...` | Versão diferente da API → use o modo manual (9.2) |
| Aviso de versão ao carregar `best_model.joblib` | Use a mesma versão do scikit-learn do treino |

---

## 14. Próximos passos

- Substituir/adicionar **XGBoost/LightGBM/CatBoost** e comparar.
- **SHAP** para explicações por cliente.
- **Intervalos de previsão** (quantile regression / conformal prediction) para precificar com margem de risco.
- Tracking de experimentos com **MLflow**; agendamento do pipeline com **Airflow/cron**.
- Testes automatizados (pytest) e CI; validação de dados com **Great Expectations**.
- Monitoramento de **drift** de dados e de desempenho após o deploy.

---

**Créditos:** dataset *Medical Cost Personal Datasets* (Miri Choi, Kaggle). Verifique a licença no próprio dataset antes de redistribuir os dados.
