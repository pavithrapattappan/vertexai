# SOW ML Pipeline — README

This document explains what the pipeline does, how to run it, and what to look for when something goes wrong.
---

## Overview 

This pipeline connects to **Snowflake**, extracts raw features into Snowflake tables, runs a minimal **data-quality** sanity check, **preprocesses** the features into a modeling-ready table, validates against a **baseline**, **trains** per-segment clustering artifacts, performs a light **model validation** (silhouette), and—if that passes—**registers** the bundle in **Vertex AI Model Registry**.

```mermaid
flowchart TD
  A[Connection Check] --> B[Extract<br/>Snowflake ? Snowflake]
  B --> C[Data Quality (Minimal)]
  C --> D[Preprocess]
  D --> E[Feature Validation (Baseline)]
  E -->|ok or warn| F[Train Cluster Artifacts]
  F --> G[Model Validation (Silhouette Gate)]
  G -->|ok| H[Register in Vertex AI]
  E -->|fail| X[Stop]
  G -->|fail| X
```

---

## Repository Map

```
src/pipeline/components/
  +- connection_check_component.py
  +- extract_snowflake_to_snowflake_component.py
  +- data_quality_check_minimal_component.py
  +- preprocess_snowflake_component.py
  +- feature_threshold_check_component.py
  +- train_cluster_component.py
  +- model_validation_component.py
  +- register_model_component.py

src/pipeline/
  +- pipeline_definition_full.py            # connection ? extract ? dq ? preprocess ? feature-check ? train ? validate ? register
  +- run_full_pipeline.py                   # compiles & submits the full pipeline
```

---

## Inputs & Outputs (high level)

**Snowflake (inputs/outputs)**  
- Inputs: existing tables your DS code reads inside `extract` and `preprocess`.  
- Outputs:
  - `EDP_ML_DEV.SOW.SOW_HISTORY_<RUN_ID>` — extraction “history/stamp” table (best-effort).
  - `EDP_ML_DEV.SOW.SOW_FEATURES_PIPELINETEST_<RUN_ID>` — features table from extract.
  - `EDP_ML_DEV.SOW.SOW_FEATURES_PREPROCESSED_PIPELINETEST_<RUN_ID>` — preprocessed features table (used for training).

**GCS (outputs)**  
- Reports to `gs://…/reports/<RUN_ID>/…` (DQ, feature-check, model-validation).  
- Training bundle to `gs://…/models/sow-clusters/<RUN_ID>_<RUN_TIMESTAMP>/` containing:
  - `manifest.json` (per-segment summary + k/score results),
  - `score.py` (portable scoring helper),
  - `segment_<NAME>.json` (one per segment).

**Vertex AI (optional output)**  
- A **Model Registry** entry that points at the training bundle above.

---

## Components (what each one does)

### 1) `connection_check_component.py`
**Purpose:** sanity-check Snowflake connectivity with the same login flow DS code uses.  
**Input (important):** project/env/warehouse creds + `snowflake_password_secret_name`.  
**Output:** logs “Snowflake OK” with server version.

**How it works:** writes `.env.<app_env>`, imports your `SnowflakeConnector` from `src/data/connect.py`, opens a Snowpark session, runs `select current_version()`.

---

### 2) `extract_snowflake_to_snowflake_component.py`
**Purpose:** run the DS extraction inside Snowflake and persist results back to Snowflake tables.  
**Key inputs:** Snowflake creds, output database/schema, `history_table`, `features_table`, `run_id`.  
**Output:** two tables (history + features) in Snowflake; returns a status string.

**Notes:**
- It imports your DS extraction (`create_data`/`CreateData`) tolerantly and calls the standard stage methods if present.
- It writes fully qualified tables with `CREATE OR REPLACE …` fallback if `save_as_table` isn’t available.

---

### 3) `data_quality_check_minimal_component.py`
**Purpose:**  “is the table there and sane?” check.  
**Checks:** table exists, has columns, `COUNT(*) >= min_rows`, quick null-rate snapshot.  
**Output:** small JSON report to GCS (best-effort) and a status string.

---

### 4) `preprocess_snowflake_component.py`
**Purpose:** run your DS preprocessing class to produce the *preprocessed* table used for training.  
**Inputs:** Snowflake creds + `input_database/schema/table`, `preprocessed_table`, `pipeline_suffix=run_id`.  
**Output:** writes `EDP_ML_DEV.SOW.<preprocessed_table>_<RUN_ID>` in Snowflake; returns a short summary.

**Important detail:** We **append the sanitized `run_id` as a suffix once**. This prevents accidental doubling like `…_RUN_20250929_…_RUN_20250929_…`.

---

### 5) `feature_threshold_check_component.py`
**Purpose:** compare distribution stats vs a baseline JSON in GCS.  
**Inputs:** `features_table_fq` (typically the *preprocessed* table), `baseline_gcs_path`, `flex_pct`.  
**Output:** JSON report to GCS; status `ok|warn|fail`.  
**Behavior:** missing baseline features ? marked as `fail`. On exceptions, writes an error artifact to GCS.

---

### 6) `train_cluster_component.py` ? `train_cluster_artifacts`
**Purpose:** per-segment clustering (KMeans/MiniBatchKMeans) and artifact bundle creation.  
**Inputs:** preprocessed Snowflake table, feature columns, k range, min customers, max rows, industries.  
**Output (to GCS):**
```
gs://…/models/sow-clusters/<RUN_ID>_<RUN_TS>/
  +- manifest.json
  +- score.py
  +- segment_<SEG_A>.json
  +- segment_<SEG_B>.json (etc.)
```
**Notes:** converts Snowpark ? pandas after optional industry filter and row cap; tries multiple k; keeps best by silhouette.

---

### 7) `model_validation_component.py`
**Purpose:** minimal acceptance test on the training bundle.  
**Inputs:** `bundle_gcs_uri`, `silhouette_threshold` (e.g., 0.20).  
**Logic:** read `manifest.json`, compute the average of `best_silhouette` over **clustered** segments;  
`avg = threshold ? ok`, `avg = 0.8 * threshold ? warn`, else `fail`. Writes report to GCS.  
**Output:** a status string `ok|warn|fail` and a JSON report (best-effort).

---

### 8) `register_model_component.py`
**Purpose:** register the GCS bundle in Vertex AI Model Registry.  
**Inputs:** project/location, `artifact_gcs_uri` (or `model_gcs_prefix + run_id + run_timestamp_utc`), optional labels/description.  
**Output:** Model resource name (string).

---

## End-to-End Pipeline: `pipeline_definition_full.py`

The pipeline wires components in this order:

1. **Connection Check**  
2. **Extract** ? writes `SOW_FEATURES_PIPELINETEST_<RUN_ID>`  
3. **Data Quality (Minimal)** on the extracted features table  
4. **Preprocess** ? writes `SOW_FEATURES_PREPROCESSED_PIPELINETEST_<RUN_ID>`  
5. **Feature Validation** on the **preprocessed** table vs baseline  
6. **Train Cluster Artifacts** if feature validation is not `fail`  
7. **Model Validation** (silhouette gate)  
8. **Register Model** **only if** validation is `ok`

**Key gating/conditions**  
- Feature validation `fail` ? training is skipped.  
- Model validation `ok` ? proceed to registration; `warn`/`fail` ? stop.

---

## How to Run

1) **Compile & submit**  
Use `src/pipeline/run_full_pipeline.py`. It compiles the pipeline, initializes Vertex, and submits a `PipelineJob` with parameters.

2) **Required parameters** (typical)
- Snowflake: `app_env`, `gcp_project_id`, `snowflake_*`, `snowflake_password_secret_name`  
- Tables: database/schema + base names (features/preprocessed)  
- Run metadata: `run_id`, `run_timestamp_utc`  
- Baseline path: `baseline_gcs_path`  
- GCS: `gcs_report_path`, `model_gcs_prefix`  
- Training knobs: `industries_json`, `feature_cols_json`, `clusters_min/max`, `min_customers_for_cluster`, `training_max_rows`

3) **Service account**  
Use an SA with access to:
- Secret Manager (to read Snowflake password),
- GCS buckets (read baseline / write reports / write model bundle).

---

## Artifacts Checklist

- **Snowflake**  
  - Features: `EDP_ML_DEV.SOW.SOW_FEATURES_PIPELINETEST_<RUN_ID>`  
  - Preprocessed: `EDP_ML_DEV.SOW.SOW_FEATURES_PREPROCESSED_PIPELINETEST_<RUN_ID>`

- **GCS**  
  - `reports/<RUN_ID>/…` (dq, feature-check, model-validation)  
  - `models/sow-clusters/<RUN_ID>_<RUN_TS>/manifest.json`, `score.py`, `segment_*.json`

- **Vertex AI (optional)**  
  - Model named like `sow-clusters-<RUN_ID>` (display name) pointing to the bundle above.

---



