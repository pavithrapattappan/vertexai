from kfp import dsl

# components you already have
from src.pipeline.components.connection_check_component import connection_check_component
from src.pipeline.components.extract_snowflake_to_snowflake_component import extract_snowflake_to_snowflake
from src.pipeline.components.data_quality_check_minimal_component import data_quality_check_minimal_component
from src.pipeline.components.preprocess_snowflake_component import preprocess_snowflake_component
from src.pipeline.components.feature_threshold_check_component import feature_threshold_check_component
from src.pipeline.components.train_cluster_component import train_cluster_artifacts
from src.pipeline.components.model_validation_component import model_validation_component
from src.pipeline.components.register_model_component import register_model_component


@dsl.pipeline(
    name="sow-train-validate-register",
    description="SF connect ? extract ? DQ ? preprocess ? feature-check ? train ? validate (silhouette) ? register",
)
def sow_train_validate_register_pipeline(
    # Snowflake / secret
    app_env: str = "np",
    gcp_project_id: str = "prj-hds-np-data",
    snowflake_account: str = "HDSUPPLY-DATA",
    snowflake_user: str = "INTERFACE_VERTEX_DEV",
    snowflake_database: str = "EDP_ML_DEV",
    snowflake_schema: str = "SOW",
    snowflake_role: str = "HDS-EDP-IT-MLOPS-DEVELOPER-U0",
    snowflake_warehouse: str = "MLOPS_DEV_WH1",
    snowflake_password_secret_name: str = "snowflake-password",

    # output / table bases
    output_database: str = "EDP_ML_DEV",
    output_schema: str = "SOW",
    history_table_base: str = "SOW_HISTORY",
    features_table_base: str = "SOW_FEATURES_PIPELINETEST",
    preprocessed_table_base: str = "SOW_FEATURES_PREPROCESSED_PIPELINETEST",
    segments_table_base: str = "SOW_SEGMENTS",

    # run metadata
    run_id: str = "RUN_YYYYMMDD_HHMMSS",
    run_timestamp_utc: str = "YYYYMMDD_HHMMSS",

    # extract options
    row_limit: int = 0,
    start_date: str = "",
    end_date: str = "",

    # feature-threshold baseline + knobs
    baseline_gcs_path: str = "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500/baseline.json",
    feature_flex_pct: float = 0.5,

    # training options
    industries_json: str = '["COMMERCIAL","CONSUMER"]',
    feature_cols_json: str = '["RECENCY_DAYS","ORDER_FREQUENCY","CUSTOMER_AGE_YEARS","AVG_ORDER_VALUE"]',
    clusters_min: int = 2,
    clusters_max: int = 10,
    min_customers_for_cluster: int = 60,
    training_max_rows: int = 100000,
    model_gcs_prefix: str = "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters",

    # model validation
    silhouette_threshold: float = 0.20,
    min_segments_required: int = 1,

    # registration
    project: str = "prj-hds-np-data",
    location: str = "us-central1",
    model_display_name: str = "sow-clusters-bundle",
    serving_container_image_uri: str = "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:de50c50",
    labels_json: str = '{"project":"sow","stage":"np"}',
    description: str = "Cluster bundle with per-segment artifacts",

    # reporting
    gcs_report_path: str = "gs://gcs-mlops-setup-prj-hds-np-data-unique/reports",
):
    """End-to-end pipeline (all KFP components, no CustomJob)."""

    # 1) Connection smoke test
    conn = connection_check_component(
        app_env=app_env,
        gcp_project_id=gcp_project_id,
        snowflake_account=snowflake_account,
        snowflake_user=snowflake_user,
        snowflake_database=snowflake_database,
        snowflake_schema=snowflake_schema,
        snowflake_role=snowflake_role,
        snowflake_warehouse=snowflake_warehouse,
        snowflake_password_secret_name=snowflake_password_secret_name,
    )
    conn.set_cpu_request("0.5"); conn.set_cpu_limit("1")
    conn.set_memory_request("1G"); conn.set_memory_limit("2G")

    # 2) Extract ? write features/history tables (names include run_id)
    history_table = f"{history_table_base}_{run_id}"
    features_table = f"{features_table_base}_{run_id}"

    extract = extract_snowflake_to_snowflake(
        app_env=app_env,
        gcp_project_id=gcp_project_id,
        snowflake_account=snowflake_account,
        snowflake_user=snowflake_user,
        snowflake_database=snowflake_database,
        snowflake_schema=snowflake_schema,
        snowflake_role=snowflake_role,
        snowflake_warehouse=snowflake_warehouse,
        snowflake_password_secret_name=snowflake_password_secret_name,

        output_database=output_database,
        output_schema=output_schema,
        history_table=history_table,
        features_table=features_table,

        run_id=run_id,
        run_timestamp_utc=run_timestamp_utc,
        row_limit=row_limit,
        start_date=start_date,
        end_date=end_date,
    )
    extract.set_cpu_request("2"); extract.set_cpu_limit("4")
    extract.set_memory_request("8G"); extract.set_memory_limit("16G")
    extract.after(conn)

    # FQ for features table
    feat_fq = f"{output_database}.{output_schema}.{features_table}"

    # 3) Minimal DQ on the features table
    dq = data_quality_check_minimal_component(
        app_env=app_env,
        gcp_project_id=gcp_project_id,
        snowflake_account=snowflake_account,
        snowflake_user=snowflake_user,
        snowflake_database=snowflake_database,
        snowflake_schema=snowflake_schema,
        snowflake_role=snowflake_role,
        snowflake_warehouse=snowflake_warehouse,
        snowflake_password_secret_name=snowflake_password_secret_name,

        features_table_fq=feat_fq,
        min_rows=100,

        run_id=run_id,
        run_timestamp_utc=run_timestamp_utc,
        gcs_report_path=gcs_report_path,
    )
    dq.set_cpu_request("2"); dq.set_cpu_limit("4")
    dq.set_memory_request("4G"); dq.set_memory_limit("8G")
    dq.after(extract)

    # 4) Preprocess (write preprocessed table). Pass base name; component appends suffix once.
    preprocessed_table = preprocessed_table_base
    segments_table = segments_table_base

    pre = preprocess_snowflake_component(
        app_env=app_env,
        gcp_project_id=gcp_project_id,
        snowflake_account=snowflake_account,
        snowflake_user=snowflake_user,
        snowflake_database=snowflake_database,
        snowflake_schema=snowflake_schema,
        snowflake_role=snowflake_role,
        snowflake_warehouse=snowflake_warehouse,
        snowflake_password_secret_name=snowflake_password_secret_name,

        input_database=output_database,
        input_schema=output_schema,
        input_table=features_table,          # features table name created above (no FQ)

        preprocessed_table=preprocessed_table,
        segments_table=segments_table,

        pipeline_suffix=run_id,              # component will append once (no duplication)
        min_segment_size=60,
        use_msa_in_primary_segment=True,

        run_id=run_id,
        run_timestamp_utc=run_timestamp_utc,
    )
    pre.set_cpu_request("2"); pre.set_cpu_limit("4")
    pre.set_memory_request("8G"); pre.set_memory_limit("16G")
    pre.after(dq)

    # Build FQ name of the preprocessed table exactly as the component wrote it:
    preproc_fq = f"{snowflake_database}.{snowflake_schema}.{preprocessed_table_base}_{run_id}"

    # 5) Feature-threshold check on the PREPROCESSED table
    ft = feature_threshold_check_component(
        app_env=app_env,
        gcp_project_id=gcp_project_id,
        snowflake_account=snowflake_account,
        snowflake_user=snowflake_user,
        snowflake_database=snowflake_database,
        snowflake_schema=snowflake_schema,
        snowflake_role=snowflake_role,
        snowflake_warehouse=snowflake_warehouse,
        snowflake_password_secret_name=snowflake_password_secret_name,

        features_table_fq=preproc_fq,
        baseline_gcs_path=baseline_gcs_path,
        run_id=run_id,
        run_timestamp_utc=run_timestamp_utc,

        flex_pct=feature_flex_pct,
        gcs_report_path=gcs_report_path,
        sample_rows_for_fail=10,
    )
    ft.set_cpu_request("2"); ft.set_cpu_limit("4")
    ft.set_memory_request("4G"); ft.set_memory_limit("8G")
    ft.after(pre)

    # 6) Train (only if feature-threshold != "fail")
    with dsl.Condition(ft.output != "fail"):
        train = train_cluster_artifacts(
            app_env=app_env,
            gcp_project_id=gcp_project_id,
            snowflake_account=snowflake_account,
            snowflake_user=snowflake_user,
            snowflake_database=snowflake_database,
            snowflake_schema=snowflake_schema,
            snowflake_role=snowflake_role,
            snowflake_warehouse=snowflake_warehouse,
            snowflake_password_secret_name=snowflake_password_secret_name,

            input_table_fq=preproc_fq,
            industries_json=industries_json,
            feature_cols_json=feature_cols_json,
            clusters_min=clusters_min,
            clusters_max=clusters_max,
            min_customers_for_cluster=min_customers_for_cluster,
            training_max_rows=training_max_rows,

            model_gcs_prefix=model_gcs_prefix,
            run_id=run_id,
            run_timestamp_utc=run_timestamp_utc,
        )
        train.set_cpu_request("4"); train.set_cpu_limit("8")
        train.set_memory_request("16G"); train.set_memory_limit("32G")
        train.after(ft)

        # 7) Model validation (silhouette) on training output
        mv = model_validation_component(
            project=gcp_project_id,
            location=location,
            bundle_gcs_uri=train.output,          # train returns the gs:// bundle URI
            silhouette_threshold=silhouette_threshold,
            min_segments_required=min_segments_required,
            run_id=run_id,
            gcs_report_path=gcs_report_path,
        )
        mv.set_cpu_request("1"); mv.set_cpu_limit("2")
        mv.set_memory_request("1G"); mv.set_memory_limit("2G")
        mv.after(train)

        # 8) Register ONLY if validation == "ok"
        with dsl.Condition(mv.output == "ok"):
            reg = register_model_component(
                project=project,
                location=location,
                model_display_name=f"{model_display_name}-{run_id}",
                artifact_gcs_uri=train.output,     # use exact bundle from training
                serving_container_image_uri=serving_container_image_uri,
                labels_json=labels_json,
                description=description,
            )
            reg.after(mv)
