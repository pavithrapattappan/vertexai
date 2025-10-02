import os
from datetime import datetime, timezone
from kfp import compiler
from google.cloud import aiplatform
from src.pipeline.pipeline_definition_drift import sow_drift_pipeline

# Use the same image you’ve been using
os.environ["SOW_PIPELINE_IMAGE"] = "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:de50c50"

PROJECT = "prj-hds-np-data"
REGION  = "us-central1"
STAGING_BUCKET = "gs://gcs-mlops-setup-prj-hds-np-data-unique"
PIPELINE_ROOT  = f"{STAGING_BUCKET}/pipeline_root/drift_minimal"
SERVICE_ACCOUNT = "vertex-pipeline-runner@prj-hds-np-data.iam.gserviceaccount.com"

OUT_JSON = "sow_drift_minimal.json"
DISPLAY_NAME = "sow-drift-minimal"

if __name__ == "__main__":
    run_id = f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    compiler.Compiler().compile(
        pipeline_func=sow_drift_pipeline,
        package_path=OUT_JSON,
    )

    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=STAGING_BUCKET)

    params = {
        "baseline_gcs_path": "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500/baseline.json",
        "model_bundle_gcs_uri": "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500/",
        "current_stats_gcs_path": "",                 # leave empty
        "use_baseline_as_current": True,              # test mode => OK
        "gcs_report_path": f"{STAGING_BUCKET}/reports",
        "run_id": run_id,
    }

    job = aiplatform.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path=OUT_JSON,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=params,
        enable_caching=False,
    )
    job.run(service_account=SERVICE_ACCOUNT)
