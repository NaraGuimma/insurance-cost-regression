"""Etapas 7–10 — Baseline, comparação de modelos (CV), ajuste de hiperparâmetros e avaliação no teste.

Fluxo:
  1. Carrega/limpa dados, cria features, separa treino/teste.
  2. Valida (K-Fold CV, SÓ no treino) um conjunto de modelos de regressão + baseline.
  3. Ajusta hiperparâmetros (RandomizedSearchCV) dos N melhores modelos que possuem grade.
  4. Treina todos no treino completo e mede métricas UMA vez no teste.
  5. Seleciona o melhor pelo RMSE de validação cruzada (NÃO pelo teste) e salva artefatos.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from . import config
from .metrics import regression_metrics
from .preprocessing import add_features, build_preprocessor, clean, load_raw, split_data

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RS = config.RANDOM_STATE


def _log_target(est):
    """Treina em log(1+y) e devolve previsões na escala original (reduz assimetria do alvo)."""
    return TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)


def get_models(fast: bool = False) -> dict:
    n_trees = 100 if fast else 300
    return {
        "Baseline (media)": DummyRegressor(strategy="mean"),
        "Regressao Linear": LinearRegression(),
        "Regressao Linear (log)": _log_target(LinearRegression()),
        "Ridge (log)": _log_target(Ridge(alpha=1.0)),
        "Lasso (log)": _log_target(Lasso(alpha=0.001, max_iter=20000)),
        "ElasticNet (log)": _log_target(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000)),
        "KNN": KNeighborsRegressor(n_neighbors=7),
        "Arvore de Decisao": DecisionTreeRegressor(max_depth=5, random_state=RS),
        "Random Forest": RandomForestRegressor(n_estimators=n_trees, random_state=RS),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RS),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RS),
    }


# Espaços de busca (prefixo "model__" = etapa do Pipeline; "regressor__" = estimador dentro do TransformedTargetRegressor)
PARAM_GRIDS = {
    "Ridge (log)": {"model__regressor__alpha": loguniform(1e-3, 1e2)},
    "KNN": {"model__n_neighbors": randint(3, 25), "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
    "Random Forest": {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [3, 4, 5, 6, 8, None],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": [0.5, 0.8, 1.0],
    },
    "Gradient Boosting": {
        "model__n_estimators": randint(50, 400),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__max_depth": [2, 3, 4],
        "model__subsample": uniform(0.6, 0.4),
        "model__min_samples_leaf": randint(1, 20),
    },
    "HistGradientBoosting": {
        "model__learning_rate": uniform(0.01, 0.2),
        "model__max_iter": randint(100, 500),
        "model__max_leaf_nodes": randint(8, 40),
        "model__min_samples_leaf": randint(10, 40),
        "model__l2_regularization": uniform(0, 1),
    },
}


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def cv_evaluate(pipe, X, y, cv) -> dict:
    s = cross_validate(
        pipe, X, y, cv=cv, n_jobs=-1, return_train_score=True,
        scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
    )
    return {
        "cv_rmse_mean": -s["test_rmse"].mean(),
        "cv_rmse_std": s["test_rmse"].std(),
        "cv_mae_mean": -s["test_mae"].mean(),
        "cv_r2_mean": s["test_r2"].mean(),
        "train_rmse_mean": -s["train_rmse"].mean(),   # comparar com cv_rmse => sinal de overfitting
        "fit_time_mean": s["fit_time"].mean(),
    }


def _clean_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        v = v.item() if hasattr(v, "item") else v
        out[k.replace("model__", "").replace("regressor__", "")] = v
    return out


def main(fast: bool = False) -> None:
    t0 = time.time()
    config.ensure_dirs()

    # -------------------------------------------------- dados
    df = add_features(clean(load_raw()))
    train_df, test_df = split_data(df)
    X_train, y_train = train_df[config.FEATURES], train_df[config.TARGET]
    X_test, y_test = test_df[config.FEATURES], test_df[config.TARGET]
    print(f"[split] treino={len(train_df)} | teste={len(test_df)}")

    out = df.copy()
    out["split"] = np.where(out.index.isin(test_df.index), "test", "train")
    out.to_csv(config.PROCESSED_FILE)

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)

    # -------------------------------------------------- validação cruzada (modelos padrão)
    candidates: dict[str, dict] = {}
    print(f"\n== Validação cruzada ({config.CV_FOLDS}-fold) com hiperparâmetros padrão ==")
    for name, est in get_models(fast).items():
        pipe = make_pipeline(est)
        cvm = cv_evaluate(pipe, X_train, y_train, cv)
        candidates[name] = {"pipe": pipe, "is_tuned": False, "best_params": {}, "base": name, **cvm}
        print(f"  {name:<26} RMSE={cvm['cv_rmse_mean']:>9.1f} (±{cvm['cv_rmse_std']:.0f})  R²={cvm['cv_r2_mean']:.3f}")

    # -------------------------------------------------- tuning dos melhores
    n_tune = 1 if fast else config.N_MODELS_TO_TUNE
    n_iter = 5 if fast else config.N_ITER_SEARCH
    tunable = sorted((n for n in candidates if n in PARAM_GRIDS), key=lambda n: candidates[n]["cv_rmse_mean"])[:n_tune]
    print(f"\n== RandomizedSearchCV (n_iter={n_iter}) em: {tunable} ==")
    for name in tunable:
        search = RandomizedSearchCV(
            clone(candidates[name]["pipe"]), PARAM_GRIDS[name], n_iter=n_iter, cv=cv,
            scoring="neg_root_mean_squared_error", random_state=RS, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned_pipe = clone(search.best_estimator_)
        cvm = cv_evaluate(tuned_pipe, X_train, y_train, cv)
        tuned_name = f"{name} (tuned)"
        candidates[tuned_name] = {
            "pipe": tuned_pipe, "is_tuned": True, "best_params": _clean_params(search.best_params_), "base": name, **cvm
        }
        print(f"  {tuned_name:<26} RMSE={cvm['cv_rmse_mean']:>9.1f} (±{cvm['cv_rmse_std']:.0f})  "
              f"R²={cvm['cv_r2_mean']:.3f}  params={candidates[tuned_name]['best_params']}")

    # -------------------------------------------------- treino final + teste (uma única vez)
    print("\n== Treino final e avaliação no conjunto de teste ==")
    rows, preds = [], []
    for name, c in candidates.items():
        pipe = c["pipe"].fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        n_feat = len(pipe.named_steps["prep"].get_feature_names_out())
        tm = regression_metrics(y_test, y_pred, n_features=n_feat)
        c["fitted"] = pipe
        rows.append({
            "model": name, "base_model": c["base"], "is_tuned": c["is_tuned"],
            "best_params": json.dumps(c["best_params"], default=str),
            **{k: c[k] for k in ("cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_r2_mean", "train_rmse_mean", "fit_time_mean")},
            **{f"test_{k}": v for k, v in tm.items()},
        })
        preds.append(pd.DataFrame({
            "row_id": X_test.index, "model": name, "y_true": y_test.values, "y_pred": y_pred,
        }))

    results = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    results["overfit_gap"] = results["cv_rmse_mean"] - results["train_rmse_mean"]
    results["rank_cv"] = np.arange(1, len(results) + 1)
    best_name = results.loc[0, "model"]
    results["is_best"] = results["model"] == best_name
    results.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

    # -------------------------------------------------- previsões (formato longo) p/ dashboard
    pred_df = pd.concat(preds, ignore_index=True)
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
    pred_df["abs_error"] = pred_df["residual"].abs()
    pred_df["pct_error"] = pred_df["abs_error"] / pred_df["y_true"] * 100
    pred_df["is_best_model"] = pred_df["model"] == best_name
    feats = test_df[config.FEATURES].reset_index()
    pred_df = pred_df.merge(feats, on="row_id", how="left")
    pred_df.to_csv(config.METRICS_DIR / "predictions_test.csv", index=False)

    # -------------------------------------------------- melhor modelo
    best = candidates[best_name]["fitted"]
    joblib.dump(best, config.BEST_MODEL_FILE)
    best_row = results.loc[0]
    info = {
        "best_model": best_name,
        "selection_criterion": "menor RMSE médio na validação cruzada (treino)",
        "best_params": candidates[best_name]["best_params"],
        "cv_rmse": best_row["cv_rmse_mean"],
        "test_metrics": {k.replace("test_", ""): float(v) for k, v in best_row.items() if k.startswith("test_")},
        "n_train": len(train_df), "n_test": len(test_df),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (config.METRICS_DIR / "best_model.json").write_text(json.dumps(info, indent=2, default=float))

    show = results[["rank_cv", "model", "cv_rmse_mean", "test_rmse", "test_mae", "test_r2", "test_mape", "test_within_20pct"]]
    print(show.round(3).to_string(index=False))
    print(f"\n[OK] Melhor modelo (por CV): {best_name} | RMSE teste = {best_row['test_rmse']:.1f} | "
          f"R² teste = {best_row['test_r2']:.3f} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="modo rápido (menos árvores/iterações) para testes")
    main(fast=ap.parse_args().fast)
