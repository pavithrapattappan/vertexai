import os
from datetime import datetime, timezone
from kfp import compiler
from google.cloud import aiplatform
from src.pipeline.pipeline_definition_train_validate_register import (
    sow_train_validate_register_pipeline,
)

# Image for components (your repo image)
os.environ["SOW_PIPELINE_IMAGE"] = "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:de50c50"

PROJECT = "prj-hds-np-data"
REGION = "us-central1"
STAGING_BUCKET = "gs://gcs-mlops-setup-prj-hds-np-data-unique"
PIPELINE_ROOT = STAGING_BUCKET + "/pipeline_root/train_validate_register"
SERVICE_ACCOUNT = "vertex-pipeline-runner@prj-hds-np-data.iam.gserviceaccount.com"

OUT_JSON = "sow_train_validate_register.json"
DISPLAY_NAME = "sow-train-validate-register"

if __name__ == "__main__":
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"CONN_EXTRACT_RUN_{run_ts}"

    compiler.Compiler().compile(
        pipeline_func=sow_train_validate_register_pipeline,
        package_path=OUT_JSON,
    )
    print(f"Compiled pipeline -> {OUT_JSON}")

    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=STAGING_BUCKET)

    params = {
        # Snowflake / secret
        "app_env": "np",
        "gcp_project_id": PROJECT,
        "snowflake_account": "HDSUPPLY-DATA",
        "snowflake_user": "INTERFACE_VERTEX_DEV",
        "snowflake_database": "EDP_ML_DEV",
        "snowflake_schema": "SOW",
        "snowflake_role": "HDS-EDP-IT-MLOPS-DEVELOPER-U0",
        "snowflake_warehouse": "MLOPS_DEV_WH1",
        "snowflake_password_secret_name": "snowflake-password",

        # table bases
        "output_database": "EDP_ML_DEV",
        "output_schema": "SOW",
        "history_table_base": "SOW_HISTORY",
        "features_table_base": "SOW_FEATURES_PIPELINETEST",
        "preprocessed_table_base": "SOW_FEATURES_PREPROCESSED_PIPELINETEST",
        "segments_table_base": "SOW_SEGMENTS",

        # run metadata
        "run_id": run_id,
        "run_timestamp_utc": run_ts,

        # extract knobs
        "row_limit": 0,
        "start_date": "",
        "end_date": "",

        # feature-threshold baseline
        "baseline_gcs_path": "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500/baseline.json",
        "feature_flex_pct": 0.5,

        # training
        "industries_json": '["COMMERCIAL","CONSUMER"]',
        "feature_cols_json": '["RECENCY_DAYS","ORDER_FREQUENCY","CUSTOMER_AGE_YEARS","AVG_ORDER_VALUE"]',
        "clusters_min": 2,
        "clusters_max": 10,
        "min_customers_for_cluster": 60,
        "training_max_rows": 100000,
        "model_gcs_prefix": "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters",

        # validation
        "silhouette_threshold": 0.20,
        "min_segments_required": 1,

        # registry
        "project": PROJECT,
        "location": REGION,
        "model_display_name": "sow-clusters-bundle",
        "serving_container_image_uri": "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:de50c50",
        "labels_json": '{"project":"sow","stage":"np"}',
        "description": "Cluster bundle with per-segment artifacts",

        # reports
        "gcs_report_path": STAGING_BUCKET + "/reports",
    }

    job = aiplatform.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path=OUT_JSON,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=params,
        enable_caching=False,
    )
    job.run(service_account=SERVICE_ACCOUNT)
    print("Submitted PipelineJob")
