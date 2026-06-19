-- =============================================
-- Pakistan OOP & Disease Burden Project
-- Step 1: Create database schema
-- =============================================

USE PakistanHealthDB;
GO

-- Raw World Bank data
IF OBJECT_ID('dbo.raw_world_bank', 'U') IS NOT NULL DROP TABLE dbo.raw_world_bank;
CREATE TABLE dbo.raw_world_bank (
    id                          INT IDENTITY(1,1) PRIMARY KEY,
    country                     NVARCHAR(100),
    country_code                CHAR(3),
    year                        INT,
    oop_pct_current_health_exp  FLOAT,
    current_health_exp_pct_gdp  FLOAT,
    current_health_exp_per_capita_usd FLOAT,
    under5_mortality_per_1000   FLOAT,
    infant_mortality_per_1000   FLOAT,
    maternal_mortality_per_100k FLOAT,
    hospital_beds_per_1000      FLOAT,
    physicians_per_1000         FLOAT,
    gdp_per_capita_usd          FLOAT,
    poverty_headcount_ratio     FLOAT,
    created_at                  DATETIME DEFAULT GETDATE()
);
GO

-- Raw WHO data
IF OBJECT_ID('dbo.raw_who', 'U') IS NOT NULL DROP TABLE dbo.raw_who;
CREATE TABLE dbo.raw_who (
    id                    INT IDENTITY(1,1) PRIMARY KEY,
    country               NVARCHAR(100),
    year                  INT,
    ncd_mortality_30_70   FLOAT,
    pm25_exposure         FLOAT,
    basic_water_access    FLOAT,
    safe_sanitation       FLOAT,
    created_at            DATETIME DEFAULT GETDATE()
);
GO

-- Cleaned merged analytical table
IF OBJECT_ID('dbo.pakistan_health_clean', 'U') IS NOT NULL DROP TABLE dbo.pakistan_health_clean;
CREATE TABLE dbo.pakistan_health_clean (
    year                        INT PRIMARY KEY,
    oop_pct_health_exp          FLOAT,
    health_exp_pct_gdp          FLOAT,
    health_exp_per_capita       FLOAT,
    under5_mortality            FLOAT,
    infant_mortality            FLOAT,
    maternal_mortality          FLOAT,
    hospital_beds               FLOAT,
    physicians                  FLOAT,
    gdp_per_capita              FLOAT,
    poverty_ratio               FLOAT,
    ncd_mortality               FLOAT,
    pm25                        FLOAT,
    -- Engineered features
    oop_above_who_threshold     BIT,       -- 1 if OOP > 40% (WHO threshold)
    health_crisis_period        BIT,       -- 1 = 2018-2023 (post-NHSRC reforms)
    oop_gdp_ratio               FLOAT,
    disease_burden_index        FLOAT
);
GO

PRINT 'Schema created successfully';
