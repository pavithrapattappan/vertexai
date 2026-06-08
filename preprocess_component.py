# src/pipeline/components/preprocess_in_bigquery.py
import os
from kfp import dsl

IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE",

)

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=["google-cloud-bigquery>=3.14.0"],
)
def preprocess_in_bigquery(
    bq_project: str,
    bq_dataset: str,
    bq_features_table: str,      # input
    bq_preprocessed_table: str,  # output
) -> str:
    """
    Builds the preprocessed table in BigQuery, carefully qualifying columns to
    avoid ambiguity (especially TAM_TIER_SOURCE when joining threshold table).
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=bq_project)

    ds  = f"{bq_project}.{bq_dataset}"
    src = f"{ds}.{bq_features_table}"
    dst = f"{ds}.{bq_preprocessed_table}"

    sql = f"""
    -- 1) Average annual spend per customer
    CREATE TEMP TABLE tmp_avg_spend AS
    SELECT
      CUSTOMER_ID,
      AVG(YEARLY_GSAR) AS AVG_ANNUAL_SPEND
    FROM (
      SELECT
        CUSTOMER_ID,
        EXTRACT(YEAR FROM DATE(ORDER_DATE)) AS ORDER_YEAR,
        SUM(GSAR) AS YEARLY_GSAR
      FROM `{src}`
      GROUP BY CUSTOMER_ID, ORDER_YEAR
    )
    GROUP BY CUSTOMER_ID;

    -- 2) Add tier numeric + avg spend
    CREATE TEMP TABLE tmp_features AS
    SELECT
      f.*,
      CASE f.TIER
        WHEN 'Tier 0' THEN 0 WHEN 'Tier 1' THEN 1 WHEN 'Tier 2' THEN 2
        WHEN 'Tier 3' THEN 3 WHEN 'Other' THEN 4 ELSE NULL
      END AS TIER_NUMERIC,
      a.AVG_ANNUAL_SPEND
    FROM `{src}` AS f
    LEFT JOIN tmp_avg_spend AS a
      ON a.CUSTOMER_ID = f.CUSTOMER_ID;

    -- 3) Choose value & source (TAM > TIER_NUMERIC > AVG_SPEND)
    CREATE TEMP TABLE tmp_value AS
    SELECT
      tf.*,
      COALESCE(tf.TAM, tf.TIER_NUMERIC, tf.AVG_ANNUAL_SPEND) AS TAM_TIER_OR_AVGSPEND,
      CASE
        WHEN tf.TAM IS NOT NULL THEN 'TAM'
        WHEN tf.TIER_NUMERIC IS NOT NULL THEN 'TIER'
        WHEN tf.AVG_ANNUAL_SPEND IS NOT NULL THEN 'AVG_ANNUAL_SPEND'
        ELSE 'UNKNOWN'
      END AS TAM_TIER_SOURCE
    FROM tmp_features AS tf;

    -- 4) Per (industry, source) quantiles (25/50/75) for non-null values
    CREATE TEMP TABLE tmp_thresh AS
    SELECT
      INDUSTRY_CODE_DESC,
      TAM_TIER_SOURCE,
      q[OFFSET(1)] AS q1,
      q[OFFSET(2)] AS q2,
      q[OFFSET(3)] AS q3
    FROM (
      SELECT
        v.INDUSTRY_CODE_DESC,
        v.TAM_TIER_SOURCE,
        APPROX_QUANTILES(v.TAM_TIER_OR_AVGSPEND, 4) AS q
      FROM tmp_value AS v
      WHERE v.TAM_TIER_SOURCE IN ('TAM','AVG_ANNUAL_SPEND')
        AND v.TAM_TIER_OR_AVGSPEND IS NOT NULL
      GROUP BY v.INDUSTRY_CODE_DESC, v.TAM_TIER_SOURCE
    );

    -- 5) Assign TAM_TIER_BIN (QUALIFY ALL references from v.* vs t.*)
    CREATE TEMP TABLE tmp_bin AS
    SELECT
      v.*,
      CASE
        WHEN v.TAM_TIER_SOURCE = 'TIER' THEN v.TIER
        WHEN v.TAM_TIER_OR_AVGSPEND IS NULL THEN 'NA'
        WHEN v.TAM_TIER_OR_AVGSPEND <= t.q1 THEN 'Low'
        WHEN v.TAM_TIER_OR_AVGSPEND <= t.q2 THEN 'Medium-Low'
        WHEN v.TAM_TIER_OR_AVGSPEND <= t.q3 THEN 'Medium-High'
        ELSE 'High'
      END AS TAM_TIER_BIN
    FROM tmp_value AS v
    LEFT JOIN tmp_thresh AS t
      ON t.INDUSTRY_CODE_DESC = v.INDUSTRY_CODE_DESC
     AND t.TAM_TIER_SOURCE   = v.TAM_TIER_SOURCE;

    -- 6) Build segment strings
    CREATE TEMP TABLE tmp_seg AS
    SELECT
      b.*,
      CONCAT(
        COALESCE(b.INDUSTRY_CODE_DESC,'NA'),'_',
        COALESCE(b.COMPANY_TYPE_DESC,'NA'),'_',
        COALESCE(b.MSA_REGION,'NA'),'_',
        COALESCE(b.MSA,'NA'),'_',
        COALESCE(b.TAM_TIER_SOURCE,'NA'),'_',
        COALESCE(b.TAM_TIER_BIN,'NA')
      ) AS RULE_BASED_SEGMENT,
      CONCAT(
        COALESCE(b.INDUSTRY_CODE_DESC,'NA'),'_',
        COALESCE(b.COMPANY_TYPE_DESC,'NA'),'_',
        COALESCE(b.MSA_REGION,'NA'),'_',
        COALESCE(b.TAM_TIER_SOURCE,'NA'),'_',
        COALESCE(b.TAM_TIER_BIN,'NA')
      ) AS RULE_BASED_SEGMENT_NO_MSA,
      CONCAT(
        COALESCE(b.INDUSTRY_CODE_DESC,'NA'),'_',
        COALESCE(b.COMPANY_TYPE_DESC,'NA'),'_',
        COALESCE(b.TAM_TIER_SOURCE,'NA'),'_',
        COALESCE(b.TAM_TIER_BIN,'NA')
      ) AS RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION
    FROM tmp_bin AS b;

    -- 7) Counts per segment & >60 fallback (qualify & avoid ambiguity)
    CREATE TEMP TABLE tmp_counts AS
    WITH c1 AS (
      SELECT RULE_BASED_SEGMENT, COUNT(DISTINCT CUSTOMER_ID) AS CNT
      FROM tmp_seg GROUP BY RULE_BASED_SEGMENT
    ),
    c2 AS (
      SELECT RULE_BASED_SEGMENT_NO_MSA, COUNT(DISTINCT CUSTOMER_ID) AS CNT
      FROM tmp_seg GROUP BY RULE_BASED_SEGMENT_NO_MSA
    ),
    c3 AS (
      SELECT RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION, COUNT(DISTINCT CUSTOMER_ID) AS CNT
      FROM tmp_seg GROUP BY RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION
    )
    SELECT
      s.*,
      c1.CNT AS COUNT_SEG_1,
      c2.CNT AS COUNT_SEG_2,
      c3.CNT AS COUNT_SEG_3
    FROM tmp_seg AS s
    LEFT JOIN c1 ON c1.RULE_BASED_SEGMENT = s.RULE_BASED_SEGMENT
    LEFT JOIN c2 ON c2.RULE_BASED_SEGMENT_NO_MSA = s.RULE_BASED_SEGMENT_NO_MSA
    LEFT JOIN c3 ON c3.RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION = s.RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION;

    -- 8) Final table
    CREATE OR REPLACE TABLE `{dst}` AS
    SELECT
      s.*,
      CASE
        WHEN s.COUNT_SEG_1 > 60 THEN s.RULE_BASED_SEGMENT
        WHEN s.COUNT_SEG_2 > 60 THEN s.RULE_BASED_SEGMENT_NO_MSA
        ELSE s.RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION
      END AS SEGMENT
    FROM tmp_counts AS s;
    """

    client.query(sql).result()
    return f"Preprocessed ? `{dst}`"
