import os
from kfp import compiler
from google.cloud import aiplatform
from src.pipeline.pipeline_inference import sow_batch_infer_pipeline

# Use the same image tag as training
os.environ["SOW_PIPELINE_IMAGE"] = (
    "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:5541009"
)

PROJECT        = "prj-hds-np-data"
REGION         = "us-central1"
STAGING_BUCKET = "gs://gcs-mlops-setup-prj-hds-np-data-unique"  # bucket only
PIPELINE_ROOT  = "gs://gcs-mlops-setup-prj-hds-np-data-unique/pipeline_root/831685804345"
SERVICE_ACCT   = "vertex-pipeline-runner@prj-hds-np-data.iam.gserviceaccount.com"

OUT_JSON       = "sow_batch_infer.json"
DISPLAY_NAME   = "sow-batch-infer"

# BigQuery I/O
BQ_DATASET           = "SOW_TEST"
BQ_FEATURES_TABLE    = "SOW_FEATURES_TEST"
BQ_PREPROCESSED      = "SOW_FEATURES_PREP_TEST"
BQ_SCORED_OUTPUT     = "SOW_CLUSTERED_DATA_TEST"

# Model artifacts root
MODEL_DIR_BASE_GCS   = "gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500"
MODEL_RUN_ID         = "latest"   # or e.g., "run_20250902_171436"

if __name__ == "__main__":
    # 1) Compile
    compiler.Compiler().compile(
        pipeline_func=sow_batch_infer_pipeline,
        package_path=OUT_JSON,
    )

    # 2) Init Vertex
    aiplatform.init(
        project=PROJECT,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
    )

    # 3) Submit the job
    job = aiplatform.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path=OUT_JSON,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=dict(
            bq_project=PROJECT,
            bq_dataset=BQ_DATASET,
            bq_features_table=BQ_FEATURES_TABLE,
            bq_preprocessed_table=BQ_PREPROCESSED,
            bq_scored_table=BQ_SCORED_OUTPUT,
            model_dir_base_gcs=MODEL_DIR_BASE_GCS,
            model_run_id=MODEL_RUN_ID,
            refresh_preprocess=True,
        ),
        enable_caching=False,
    )
    job.run(service_account=SERVICE_ACCT)
