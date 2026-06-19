-- =============================================
-- EDA Queries — Advanced SQL Practice
-- Covers: Window functions, CTEs, aggregation
-- =============================================

USE PakistanHealthDB;
GO

-- 1. Year-over-year OOP change using LAG window function
SELECT
    year,
    oop_pct_health_exp,
    LAG(oop_pct_health_exp, 1) OVER (ORDER BY year) AS prev_year_oop,
    ROUND(
        oop_pct_health_exp - LAG(oop_pct_health_exp, 1) OVER (ORDER BY year),
        3
    ) AS yoy_change,
    ROUND(
        (oop_pct_health_exp - LAG(oop_pct_health_exp, 1) OVER (ORDER BY year))
        / NULLIF(LAG(oop_pct_health_exp, 1) OVER (ORDER BY year), 0) * 100,
        2
    ) AS yoy_pct_change
FROM dbo.pakistan_health_clean
ORDER BY year;
GO

-- 2. 5-year rolling average of OOP spending
SELECT
    year,
    oop_pct_health_exp,
    ROUND(AVG(oop_pct_health_exp) OVER (
        ORDER BY year ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ), 3) AS rolling_5yr_avg
FROM dbo.pakistan_health_clean
ORDER BY year;
GO

-- 3. CTE: Identify years where OOP exceeded WHO 40% threshold
WITH who_threshold AS (
    SELECT
        year,
        oop_pct_health_exp,
        CASE WHEN oop_pct_health_exp > 40 THEN 'ABOVE' ELSE 'BELOW' END AS threshold_status
    FROM dbo.pakistan_health_clean
),
summary AS (
    SELECT
        threshold_status,
        COUNT(*) AS num_years,
        ROUND(AVG(oop_pct_health_exp), 2) AS avg_oop,
        MIN(year) AS first_year,
        MAX(year) AS last_year
    FROM who_threshold
    GROUP BY threshold_status
)
SELECT * FROM summary;
GO

-- 4. Pearson-style correlation proxy: OOP vs Infant Mortality
-- (rank-based approach in SQL)
WITH ranked AS (
    SELECT
        year,
        oop_pct_health_exp,
        infant_mortality,
        RANK() OVER (ORDER BY oop_pct_health_exp) AS rank_oop,
        RANK() OVER (ORDER BY infant_mortality)    AS rank_mort
    FROM dbo.pakistan_health_clean
    WHERE oop_pct_health_exp IS NOT NULL
      AND infant_mortality IS NOT NULL
)
SELECT
    COUNT(*)                              AS n,
    ROUND(1 - (6.0 * SUM(POWER(rank_oop - rank_mort, 2)))
              / (COUNT(*) * (POWER(COUNT(*), 2) - 1)), 4) AS spearman_corr_oop_mortality
FROM ranked;
GO

-- 5. Percentile ranking of each year's health spending
SELECT
    year,
    health_exp_per_capita,
    ROUND(PERCENT_RANK() OVER (ORDER BY health_exp_per_capita) * 100, 1) AS percentile_rank,
    NTILE(4) OVER (ORDER BY health_exp_per_capita) AS quartile
FROM dbo.pakistan_health_clean
ORDER BY year;
GO

-- 6. DiD setup: Pre/Post treatment periods
SELECT
    year,
    oop_pct_health_exp,
    under5_mortality,
    CASE WHEN year >= 2018 THEN 1 ELSE 0 END AS post_reform,
    CASE WHEN year >= 2018 THEN 'Post-Reform (2018+)' ELSE 'Pre-Reform (2000-2017)' END AS period
FROM dbo.pakistan_health_clean
ORDER BY year;
GO
