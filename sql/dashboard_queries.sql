-- Queries do dashboard executivo. Fonte única: usadas por dashboards/create_metabase_dashboard.py
-- e também podem ser coladas como "Datasets"/"Charts" no Superset ou como "Native query" no Metabase.
-- Cada bloco começa com "-- name: <chave>".

-- name: kpi_customers
SELECT COUNT(*) AS clientes FROM insurance_clean;

-- name: kpi_avg_charge
SELECT ROUND(CAST(AVG(charges) AS NUMERIC), 0) AS custo_medio FROM insurance_clean;

-- name: kpi_pct_smokers
SELECT ROUND(CAST(100.0 * AVG(CASE WHEN smoker = 'yes' THEN 1.0 ELSE 0.0 END) AS NUMERIC), 1) AS pct_fumantes
FROM insurance_clean;

-- name: kpi_r2
SELECT r2 FROM vw_best_model;

-- name: kpi_rmse
SELECT rmse FROM vw_best_model;

-- name: kpi_mae
SELECT mae FROM vw_best_model;

-- name: kpi_within_20
SELECT within_20pct AS pct_previsoes_ate_20 FROM vw_best_model;

-- name: charges_by_smoker
SELECT smoker AS fumante, avg_charge AS custo_medio FROM vw_charges_by_smoker ORDER BY smoker;

-- name: charges_by_age_smoker
SELECT age_group AS faixa_etaria, smoker AS fumante, avg_charge AS custo_medio
FROM vw_charges_by_age_smoker ORDER BY age_group, smoker;

-- name: charges_by_bmi
SELECT bmi_category AS categoria_imc, avg_charge AS custo_medio FROM vw_charges_by_bmi ORDER BY bmi_category;

-- name: charges_by_region
SELECT region AS regiao, avg_charge AS custo_medio FROM vw_charges_by_region ORDER BY avg_charge DESC;

-- name: model_ranking
SELECT model AS modelo, ROUND(CAST(test_rmse AS NUMERIC), 0) AS rmse_teste
FROM model_comparison ORDER BY test_rmse ASC;

-- name: feature_importance
SELECT feature AS variavel, ROUND(CAST(importance_pct AS NUMERIC), 1) AS importancia_pct
FROM feature_importance ORDER BY importance_pct DESC;

-- name: actual_vs_pred
SELECT ROUND(CAST(y_true AS NUMERIC), 0) AS real, ROUND(CAST(y_pred AS NUMERIC), 0) AS previsto
FROM vw_predictions_best;

-- name: residual_hist
SELECT FLOOR(residual / 2000.0) * 2000 AS faixa_residuo, COUNT(*) AS casos
FROM vw_predictions_best
GROUP BY FLOOR(residual / 2000.0) * 2000
ORDER BY 1;

-- name: segment_errors
SELECT segment_type AS dimensao, segment AS segmento, n AS casos,
       ROUND(CAST(mae AS NUMERIC), 0) AS mae, ROUND(CAST(mape AS NUMERIC), 1) AS mape_pct,
       ROUND(CAST(bias AS NUMERIC), 0) AS vies
FROM segment_errors ORDER BY mae DESC;

-- name: model_table
SELECT model AS modelo,
       ROUND(CAST(cv_rmse_mean AS NUMERIC), 0) AS rmse_cv,
       ROUND(CAST(test_rmse AS NUMERIC), 0) AS rmse_teste,
       ROUND(CAST(test_mae AS NUMERIC), 0) AS mae_teste,
       ROUND(CAST(test_r2 AS NUMERIC), 3) AS r2_teste,
       ROUND(CAST(test_mape AS NUMERIC), 1) AS mape_pct
FROM model_comparison ORDER BY cv_rmse_mean ASC;
