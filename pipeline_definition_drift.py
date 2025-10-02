from kfp import dsl
from src.pipeline.components.drift_check_component import drift_check_component

@dsl.pipeline(
    name="sow-drift-minimal",
    description="Minimal drift check using baseline + model manifest (supports test mode).",
)
def sow_drift_pipeline(
    baseline_gcs_path: str = "gs://YOUR_BUCKET/path/to/feature_baseline.json",
    model_bundle_gcs_uri: str = "gs://YOUR_BUCKET/models/sow-clusters/run_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS/",
    current_stats_gcs_path: str = "",      # leave empty for test mode
    use_baseline_as_current: bool = True,  # test mode
    mean_delta_warn: float = 0.25,
    mean_delta_fail: float = 0.50,
    null_delta_warn: float = 0.10,
    null_delta_fail: float = 0.30,
    gcs_report_path: str = "gs://YOUR_BUCKET/reports",
    run_id: str = "drift_test_run",
):
    task = drift_check_component(
        baseline_gcs_path=baseline_gcs_path,
        model_bundle_gcs_uri=model_bundle_gcs_uri,
        current_stats_gcs_path=current_stats_gcs_path,
        use_baseline_as_current=use_baseline_as_current,
        mean_delta_warn=mean_delta_warn,
        mean_delta_fail=mean_delta_fail,
        null_delta_warn=null_delta_warn,
        null_delta_fail=null_delta_fail,
        gcs_report_path=gcs_report_path,
        run_id=run_id,
    )
    task.set_cpu_request("1"); task.set_cpu_limit("2")
    task.set_memory_request("2G"); task.set_memory_limit("4G")
