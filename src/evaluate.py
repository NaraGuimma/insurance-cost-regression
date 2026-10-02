"""Etapas 11–12 — Diagnóstico do melhor modelo, interpretabilidade e análise de erros por segmento."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import KFold, learning_curve  # noqa: E402

from . import config  # noqa: E402
from .metrics import regression_metrics  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame) -> None:
    d = results.sort_values("test_rmse", ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].barh(d["model"], d["cv_rmse_mean"], xerr=d["cv_rmse_std"], color="#8da0cb", label="CV (treino)")
    ax[0].scatter(d["test_rmse"], d["model"], color="#d62728", zorder=3, label="Teste")
    ax[0].set_title("RMSE — validação cruzada vs. teste (menor = melhor)")
    ax[0].legend()
    ax[1].barh(d["model"], d["test_r2"], color="#66c2a5")
    ax[1].set_xlim(min(0, d["test_r2"].min() - 0.05), 1)
    ax[1].set_title("R² no teste (maior = melhor)")
    ax[1].set_yticklabels([])
    _save(fig, "10_model_comparison.png")


def plot_pred_vs_actual(pred: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=pred, x="y_true", y="y_pred", hue="smoker", alpha=0.7,
                    palette={"yes": "#d62728", "no": "#1f77b4"}, ax=ax)
    lim = [0, max(pred["y_true"].max(), pred["y_pred"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="previsão perfeita")
    ax.set(xlim=lim, ylim=lim, xlabel="Real", ylabel="Previsto", title=f"Real x Previsto — {name}")
    ax.legend()
    _save(fig, "11_actual_vs_predicted.png")


def plot_residuals(pred: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    sns.scatterplot(data=pred, x="y_pred", y="residual", hue="smoker", alpha=0.7,
                    palette={"yes": "#d62728", "no": "#1f77b4"}, ax=ax[0])
    ax[0].axhline(0, color="k", ls="--", lw=1)
    ax[0].set_title("Resíduos x Previsto (procure padrões/funil)")
    sns.histplot(pred["residual"], kde=True, bins=30, ax=ax[1], color="#4c72b0")
    ax[1].set_title("Distribuição dos resíduos")
    stats.probplot(pred["residual"], dist="norm", plot=ax[2])
    ax[2].set_title("QQ-plot dos resíduos")
    fig.suptitle(f"Diagnóstico de resíduos — {name}", y=1.02)
    _save(fig, "12_residual_diagnostics.png")


def plot_learning_curve(best, X, y) -> None:
    cv = KFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    sizes, tr, va = learning_curve(
        best, X, y, cv=cv, train_sizes=np.linspace(0.15, 1.0, 6),
        scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=config.RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, -tr.mean(1), "o-", label="Treino")
    ax.plot(sizes, -va.mean(1), "o-", label="Validação (CV)")
    ax.fill_between(sizes, -va.mean(1) - va.std(1), -va.mean(1) + va.std(1), alpha=0.15)
    ax.set(xlabel="Tamanho do treino", ylabel="RMSE", title="Curva de aprendizado (melhor modelo)")
    ax.legend()
    _save(fig, "13_learning_curve.png")


def feature_importance(best, X_test, y_test) -> pd.DataFrame:
    r = permutation_importance(
        best, X_test, y_test, n_repeats=30, random_state=config.RANDOM_STATE,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    fi = (
        pd.DataFrame({"feature": X_test.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    pos = fi["importance_mean"].clip(lower=0)
    fi["importance_pct"] = pos / pos.sum() * 100
    fi.to_csv(config.METRICS_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    d = fi.iloc[::-1]
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#4c72b0")
    ax.set(xlabel="Aumento do RMSE ao embaralhar a variável (US$)", title="Importância por permutação (conjunto de teste)")
    _save(fig, "14_permutation_importance.png")
    return fi


def segment_errors(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seg in ["smoker", "age_group", "bmi_category", "region", "sex"]:
        for val, g in pred.groupby(seg):
            m = regression_metrics(g["y_true"], g["y_pred"])
            rows.append({
                "segment_type": seg, "segment": str(val), "n": len(g), "avg_charge": g["y_true"].mean(),
                "mae": m["mae"], "rmse": m["rmse"], "mape": m["mape"], "bias": m["bias"],
            })
    se = pd.DataFrame(rows)
    se.to_csv(config.METRICS_DIR / "segment_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    d = se.assign(label=se["segment_type"] + " = " + se["segment"]).sort_values("mae")
    ax.barh(d["label"], d["mae"], color="#dd8452")
    ax.set(xlabel="MAE (US$)", title="Erro absoluto médio por segmento — onde o modelo erra mais?")
    _save(fig, "15_error_by_segment.png")
    return se


def main() -> None:
    config.ensure_dirs()
    results = pd.read_csv(config.METRICS_DIR / "model_comparison.csv")
    info = json.loads((config.METRICS_DIR / "best_model.json").read_text())
    best_name = info["best_model"]
    best = joblib.load(config.BEST_MODEL_FILE)

    data = pd.read_csv(config.PROCESSED_FILE, index_col="row_id")
    train, test = data[data["split"] == "train"], data[data["split"] == "test"]
    X_train, y_train = train[config.FEATURES], train[config.TARGET]
    X_test, y_test = test[config.FEATURES], test[config.TARGET]

    pred = pd.read_csv(config.METRICS_DIR / "predictions_test.csv")
    pred = pred[pred["model"] == best_name].copy()

    print(f"== Avaliando o melhor modelo: {best_name} ==")
    plot_model_comparison(results)
    plot_pred_vs_actual(pred, best_name)
    plot_residuals(pred, best_name)
    plot_learning_curve(best, X_train, y_train)
    fi = feature_importance(best, X_test, y_test)
    se = segment_errors(pred)

    m = info["test_metrics"]
    print(f"Teste: RMSE={m['rmse']:.0f} | MAE={m['mae']:.0f} | R²={m['r2']:.3f} | MAPE={m['mape']:.1f}% | "
          f"±20%: {m['within_20pct']:.0f}% dos casos")
    print("\nTop variáveis (permutação):")
    print(fi.head(5)[["feature", "importance_mean", "importance_pct"]].round(1).to_string(index=False))
    worst = se.sort_values("mae", ascending=False).iloc[0]
    print(f"\nSegmento com maior MAE: {worst['segment_type']}={worst['segment']} (MAE={worst['mae']:.0f}, n={worst['n']})")
    print(f"[ok] Figuras em {config.FIGURES_DIR} | tabelas em {config.METRICS_DIR}")


if __name__ == "__main__":
    main()
