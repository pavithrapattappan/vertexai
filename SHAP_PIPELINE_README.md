# README — SHAP Explain Pipeline

This pipeline runs SHAP explanations against an already **preprocessed + clustered** dataset using your model bundle artifacts in GCS.

---

## What it runs

- **Component:** `shap_explain_component`
- **Pipeline wrapper:** `pipeline_definition_shap.py`
- **Runner:** `run_shap_pipeline.py`

Given:
- a **preprocessed table** in Snowflake
- a **model bundle folder** in GCS (from your training step) that contains `manifest.json` and `segment_*.json`

…it will:
1. Read `manifest.json` and resolve which segments to explain.
2. Load data from the Snowflake preprocessed table.
3. Call your project class `src.data.shap_explain` (method `run_shap_explain`) to compute SHAP.
4. Write results into a Snowflake table (default `EDP_ML_DEV.SOW.SOW_SHAP_EXPLAIN_PIPELINETEST`).
5. (Optional) Upload PNG plots + JSON per segment to a debug GCS bucket.

---

## Prerequisites

- The pipeline image has your code under `/app/src` (so it can import `src.data.shap_explain` and `src.data.connect`).
- The **model bundle** exists at `gs://…/models/sow-clusters/<run_id>_<timestamp>/` with:
  - `manifest.json`
  - `segment_*.json`
- Vertex runner SA has access to:
  - **Secret Manager** (to read Snowflake password)
  - **GCS** (to read model bundle, write debug outputs if enabled)
- Snowflake network policy allows Vertex/KFP egress IP (or you use NAT/PrivateLink).

---

## Configure

Edit `src/pipeline/run_shap_pipeline.py` parameter values:

```python
"app_env": "np",
"gcp_project_id": "prj-hds-np-data",
"snowflake_account": "HDSUPPLY-DATA",
"snowflake_user": "INTERFACE_VERTEX_DEV",
"snowflake_database": "EDP_ML_DEV",
"snowflake_schema": "SOW",
"snowflake_role": "HDS-EDP-IT-MLOPS-DEVELOPER-U0",
"snowflake_warehouse": "MLOPS_DEV_WH1",
"snowflake_password_secret_name": "snowflake-password",

# Tables
"input_preprocessed_table": "SOW_FEATURES_PREPROCESSED_PIPELINETEST_<YOUR_VARIANT>",
"input_clusters_table": "SOW_CUSTOMER_LEVEL_CLUSTERS_PIPELINETEST_<YOUR_VARIANT>",
"output_table": "EDP_ML_DEV.SOW.SOW_SHAP_EXPLAIN_PIPELINETEST",

# Bundle
"bundle_gcs_uri": "gs://<bucket>/models/sow-clusters/<run_id>_<timestamp>/",
"model_bundle_gcs_uri": "gs://<bucket>/models/sow-clusters/<run_id>_<timestamp>/",
"run_id": "<run_id>",
"run_timestamp_utc": "<iso-timestamp>",

# Segment
"segment_name": "CONSUMER",   # "" runs all segments
"max_rows_to_explain": 2000,
"sample_frac_for_background": 0.05,

# Optional debug
"debug_gcs_bucket": "gs://<bucket>/debug"
```

---

## Run

```bash
# compile and submit
python -m src.pipeline.run_shap_pipeline
```

---

## Outputs

1. **Snowflake table** (default)
   - `EDP_ML_DEV.SOW.SOW_SHAP_EXPLAIN_PIPELINETEST`
   - Typical cols: `SEGMENT`, `CLUSTER`, `CLUSTER_SHAP`, `CLUSTER_SHAP_PLOT`, `RUNID`, `RUNTIMESTAMP`

2. **Optional GCS artifacts** (if `debug_gcs_bucket` set)
   - `gs://<debug-bucket>/shap_plots/<run_id>/<SEGMENT>/<CLUSTER>.png`
   - `gs://<debug-bucket>/shap_outputs/<run_id>/<SEGMENT>.json`

---

## Segment selection behavior

- Reads segment keys from the bundle’s `manifest.json`.
- `segment_name` is a prefix filter:
  - "CONSUMER" ? runs only CONSUMER segments
  - "" ? runs all segments

---


