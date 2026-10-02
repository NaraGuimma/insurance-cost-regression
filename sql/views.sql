-- Views consumidas pelo dashboard. Sintaxe compatível com Postgres e SQLite.
-- (ROUND(CAST(x AS NUMERIC), n) é necessário no Postgres: ROUND(double precision, int) não existe.)

CREATE VIEW vw_best_model AS
SELECT model,
       ROUND(CAST(test_rmse AS NUMERIC), 0)  AS rmse,
       ROUND(CAST(test_mae AS NUMERIC), 0)   AS mae,
       ROUND(CAST(test_r2 AS NUMERIC), 3)    AS r2,
       ROUND(CAST(test_mape AS NUMERIC), 1)  AS mape_pct,
       ROUND(CAST(test_within_20pct AS NUMERIC), 1) AS within_20pct,
       ROUND(CAST(cv_rmse_mean AS NUMERIC), 0) AS cv_rmse
FROM model_comparison
WHERE is_best;

CREATE VIEW vw_predictions_best AS
SELECT * FROM predictions_test WHERE is_best_model;

CREATE VIEW vw_charges_by_smoker AS
SELECT smoker,
       COUNT(*) AS customers,
       ROUND(CAST(AVG(charges) AS NUMERIC), 0) AS avg_charge
FROM insurance_clean
GROUP BY smoker;

CREATE VIEW vw_charges_by_age_smoker AS
SELECT age_group, smoker,
       COUNT(*) AS customers,
       ROUND(CAST(AVG(charges) AS NUMERIC), 0) AS avg_charge
FROM insurance_clean
GROUP BY age_group, smoker;

CREATE VIEW vw_charges_by_bmi AS
SELECT CASE bmi_category WHEN 'underweight' THEN '1-underweight' WHEN 'normal' THEN '2-normal'
                         WHEN 'overweight'  THEN '3-overweight'  ELSE '4-obese' END AS bmi_category,
       COUNT(*) AS customers,
       ROUND(CAST(AVG(charges) AS NUMERIC), 0) AS avg_charge
FROM insurance_clean
GROUP BY bmi_category;

CREATE VIEW vw_charges_by_region AS
SELECT region,
       COUNT(*) AS customers,
       ROUND(CAST(AVG(charges) AS NUMERIC), 0) AS avg_charge
FROM insurance_clean
GROUP BY region;
