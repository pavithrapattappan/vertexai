# src/pipeline/pipeline_definition_shap.py
from kfp import dsl
from src.pipeline.components.shap_explain_component import shap_explain_component

@dsl.pipeline(
    name="sow-shap-explain-pipeline",
    description="Compute SHAP explanations for preprocessed + clustered customers (per-segment).",
)
def sow_shap_explain_pipeline(
    # Snowflake/secret inputs
    app_env: str = "np",
    gcp_project_id: str = "prj-hds-np-data",
    snowflake_account: str = "HDSUPPLY-DATA",
    snowflake_user: str = "INTERFACE_VERTEX_DEV",
    snowflake_database: str = "EDP_ML_DEV",
    snowflake_schema: str = "SOW",
    snowflake_role: str = "HDS-EDP-IT-MLOPS-DEVELOPER-U0",
    snowflake_warehouse: str = "MLOPS_DEV_WH1",
    snowflake_password_secret_name: str = "snowflake-password",

    # Inputs
    input_preprocessed_table: str = "SOW_FEATURES_PREPROCESSED_PIPELINETEST_CONSUMER_PIPELINETEST",
    input_clusters_table: str = "SOW_CUSTOMER_LEVEL_CLUSTERS_PIPELINETEST_CONSUMER_PIPELINETEST",

    # Where to write results in Snowflake (default as requested)
    output_table: str = "EDP_ML_DEV.SOW.SOW_SHAP_EXPLAIN_PIPELINETEST",

    # GCS debug bucket for plots/json (optional; set to gs://... to enable uploads)
    debug_gcs_bucket: str = "",

    # Bundle and run params (declare once)
    bundle_gcs_uri: str = "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500/",
    model_bundle_gcs_uri: str = "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500/",
    run_id: str = "run_20250913_203500_20250913_203500",
    run_timestamp_utc: str = "2025-09-13T20:35:00Z",

    # default to CONSUMER (focus on consumer segment)
    segment_name: str = "CONSUMER",
    max_rows_to_explain: int = 5000,
    sample_frac_for_background: float = 0.05,
):
    """
    Pipeline wrapper that calls the SHAP explain component.
    """

    shap = shap_explain_component(
        app_env=app_env,
        gcp_project_id=gcp_project_id,
        snowflake_account=snowflake_account,
        snowflake_user=snowflake_user,
        snowflake_database=snowflake_database,
        snowflake_schema=snowflake_schema,
        snowflake_warehouse=snowflake_warehouse,
        snowflake_role=snowflake_role,
        snowflake_password_secret_name=snowflake_password_secret_name,

        input_preprocessed_table=input_preprocessed_table,
        input_clusters_table=input_clusters_table,

        # pass table + debug bucket through to component
        output_table=output_table,
        debug_gcs_bucket=debug_gcs_bucket,

        # bundles / run metadata
        model_bundle_gcs_uri=model_bundle_gcs_uri,
        bundle_gcs_uri=bundle_gcs_uri,
        run_id=run_id,
        run_timestamp_utc=run_timestamp_utc,

        segment_name=segment_name,
        max_rows_to_explain=max_rows_to_explain,
        sample_frac_for_background=sample_frac_for_background,
    )

    # resource hints (adjust if needed)
    shap.set_cpu_request("4")
    shap.set_cpu_limit("8")
    shap.set_memory_request("16G")
    shap.set_memory_limit("32G")
