"""Etapa 3 — Análise exploratória de dados (EDA).

Gera gráficos em reports/figures/ e um resumo em reports/metrics/ (eda_summary.csv, eda_insights.json).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from . import config  # noqa: E402
from .preprocessing import add_features, clean, load_raw  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"yes": "#d62728", "no": "#1f77b4"}


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    raw = load_raw()
    print("== Qualidade dos dados (bruto) ==")
    print(f"formato: {raw.shape} | nulos: {int(raw.isna().sum().sum())} | duplicatas: {int(raw.duplicated().sum())}")
    df = add_features(clean(raw))

    # ---------------------------------------------------------- resumo estatístico
    summary = df.describe(include="all").T
    summary.to_csv(config.METRICS_DIR / "eda_summary.csv")

    # ---------------------------------------------------------- 1. distribuição do alvo
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["charges"], bins=40, kde=True, ax=ax[0], color="#4c72b0")
    ax[0].set_title(f"Distribuição de charges (assimetria = {df['charges'].skew():.2f})")
    sns.histplot(np.log1p(df["charges"]), bins=40, kde=True, ax=ax[1], color="#55a868")
    ax[1].set_title(f"log(1+charges) (assimetria = {np.log1p(df['charges']).skew():.2f})")
    _save(fig, "01_target_distribution.png")

    # ---------------------------------------------------------- 2. fumante x custo
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df, x="smoker", y="charges", hue="smoker", palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Custo por condição de fumante")
    _save(fig, "02_charges_by_smoker.png")

    # ---------------------------------------------------------- 3 e 4. idade/IMC x custo
    for fname, col, title in (("03_age_vs_charges.png", "age", "Idade"), ("04_bmi_vs_charges.png", "bmi", "IMC")):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=col, y="charges", hue="smoker", palette=PALETTE, alpha=0.7, ax=ax)
        ax.set_title(f"{title} x custo (por fumante)")
        _save(fig, fname)

    # ---------------------------------------------------------- 5. correlação
    corr_df = df[["age", "bmi", "children", "smoker_obese", "charges"]].copy()
    corr_df["smoker_bin"] = (df["smoker"] == "yes").astype(int)
    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlação (Pearson)")
    _save(fig, "05_correlation_heatmap.png")

    # ---------------------------------------------------------- 6. médias por categoria
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), ["region", "sex", "bmi_category", "age_group"]):
        order = sorted(df[col].unique())
        sns.barplot(data=df, x=col, y="charges", hue="smoker", palette=PALETTE, order=order, errorbar=None, ax=ax)
        ax.set_title(f"Custo médio por {col}")
    _save(fig, "06_categorical_means.png")

    # ---------------------------------------------------------- insights quantitativos
    smoke = df.loc[df["smoker"] == "yes", "charges"]
    nosmoke = df.loc[df["smoker"] == "no", "charges"]
    t_stat, p_val = stats.ttest_ind(smoke, nosmoke, equal_var=False)
    q1, q3 = df["charges"].quantile([0.25, 0.75])
    iqr = q3 - q1
    insights = {
        "n_rows": int(len(df)),
        "mean_charge": float(df["charges"].mean()),
        "median_charge": float(df["charges"].median()),
        "pct_smokers": float((df["smoker"] == "yes").mean() * 100),
        "mean_charge_smoker": float(smoke.mean()),
        "mean_charge_non_smoker": float(nosmoke.mean()),
        "smoker_cost_ratio": float(smoke.mean() / nosmoke.mean()),
        "welch_ttest_p_value_smoker": float(p_val),
        "skew_charges": float(df["charges"].skew()),
        "skew_log_charges": float(np.log1p(df["charges"]).skew()),
        "n_outliers_iqr_charges": int(((df["charges"] < q1 - 1.5 * iqr) | (df["charges"] > q3 + 1.5 * iqr)).sum()),
        "corr_with_charges": corr["charges"].drop("charges").round(3).to_dict(),
    }
    (config.METRICS_DIR / "eda_insights.json").write_text(json.dumps(insights, indent=2, ensure_ascii=False))

    print("\n== Principais achados ==")
    print(f"- Custo médio: {insights['mean_charge']:.0f} | mediana: {insights['median_charge']:.0f}")
    print(f"- Fumantes custam {insights['smoker_cost_ratio']:.1f}x mais (Welch p={p_val:.1e})")
    print(f"- Assimetria de charges: {insights['skew_charges']:.2f} -> com log: {insights['skew_log_charges']:.2f}")
    print(f"- Outliers (IQR) em charges: {insights['n_outliers_iqr_charges']} (mantidos: são casos reais de alto custo)")
    print(f"[ok] Figuras em {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
