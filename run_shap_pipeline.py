# src/pipeline/run_shap_pipeline.py
import os
from kfp import compiler
from google.cloud import aiplatform
from src.pipeline.pipeline_definition_shap import sow_shap_explain_pipeline

# set the image tag you built/pushed that includes /app/src and required deps
os.environ["SOW_PIPELINE_IMAGE"] = "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:de50c50"

PROJECT = "prj-hds-np-data"
REGION = "us-central1"
STAGING_BUCKET = "gs://gcs-mlops-setup-prj-hds-np-data-unique"
PIPELINE_ROOT = STAGING_BUCKET + "/pipeline_root/shap_explain"
SERVICE_ACCOUNT = "vertex-pipeline-runner@prj-hds-np-data.iam.gserviceaccount.com"

OUT_JSON = "sow_shap_explain.json"
DISPLAY_NAME = "sow-shap-explain"

if __name__ == "__main__":
    # compile
    compiler.Compiler().compile(
        pipeline_func=sow_shap_explain_pipeline,
        package_path=OUT_JSON,
    )

    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=STAGING_BUCKET)

    job = aiplatform.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path=OUT_JSON,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "app_env": "np",
            "gcp_project_id": PROJECT,
            "snowflake_account": "HDSUPPLY-DATA",
            "snowflake_user": "INTERFACE_VERTEX_DEV",
            "snowflake_database": "EDP_ML_DEV",
            "snowflake_schema": "SOW",
            "snowflake_role": "HDS-EDP-IT-MLOPS-DEVELOPER-U0",
            "snowflake_warehouse": "MLOPS_DEV_WH1",
            "snowflake_password_secret_name": "snowflake-password",
            # input tables (your CONSUMER example)
            "input_preprocessed_table": "SOW_FEATURES_PREPROCESSED_PIPELINETEST_CONSUMER_PIPELINETEST",
            "input_clusters_table": "SOW_CUSTOMER_LEVEL_CLUSTERS_PIPELINETEST_CONSUMER_PIPELINETEST",
            "output_table": "EDP_ML_DEV.SOW.SOW_SHAP_EXPLAIN_PIPELINETEST",
           # bundle
            "bundle_gcs_uri": "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500/",
            "model_bundle_gcs_uri": "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500/",
            "run_id": "run_20250913_203500_20250913_203500",
            "run_timestamp_utc": "2025-09-13T20:35:00Z",
            "segment_name": "CONSUMER",
            "max_rows_to_explain": 2000,  # tune for speed/memory
            "sample_frac_for_background": 0.05,
        },
        enable_caching=False,
    )
    job.run(service_account=SERVICE_ACCOUNT)
