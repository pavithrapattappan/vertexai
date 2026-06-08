import os
from kfp import dsl

IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE",
    )

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "google-cloud-storage>=2.8.0",
        "python-dotenv>=1.0.1",
        "jsonschema>=4.0.0",
    ],
)
def model_validation_component(
    # inputs
    project: str,
    location: str,
    bundle_gcs_uri: str,           # gs://.../<artifact_folder>/
    silhouette_threshold: float = 0.20,  # average silhouette threshold to pass
    min_segments_required: int = 1,
    run_id: str = "",
    gcs_report_path: str = "",     # optional path to write validation report
) -> str:
    """
    Simple model validation:
      - loads manifest.json from bundle_gcs_uri/manifest.json
      - computes average of per-segment `best_silhouette` for segments with status CLUSTERED
      - returns "ok" if avg >= silhouette_threshold, "warn" if slightly below, otherwise "fail"
      - writes a small json report to gcs_report_path/<run_id>/model_validation_{run_id}.json (best-effort)
    Returns primary output: status string ("ok"|"warn"|"fail")
    """
    import os
    import json, time, traceback
    from pathlib import Path

    def _p(m: str):
        print(f"[MODEL-VAL] {m}", flush=True)

    try:
        from google.cloud import storage
    except Exception as e:
        _p(f"Failed to import google-cloud-storage: {e}")
        raise

    # normalize bundle path
    if not bundle_gcs_uri:
        raise ValueError("bundle_gcs_uri must be provided")

    b = bundle_gcs_uri.rstrip("/")
    if not b.startswith("gs://"):
        raise ValueError("bundle_gcs_uri must be a gs:// path")

    try:
        _, rest = b.split("gs://", 1)
        bucket_name, prefix = (rest.split("/", 1) + [""])[:2]
        prefix = prefix.rstrip("/") if prefix else ""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        manifest_path = f"{prefix}/manifest.json" if prefix else "manifest.json"
        _p(f"Loading manifest from gs://{bucket_name}/{manifest_path}")
        blob = bucket.blob(manifest_path)
        if not blob.exists():
            raise FileNotFoundError(f"manifest.json not found at {bundle_gcs_uri}/manifest.json")
        manifest = json.loads(blob.download_as_text())
    except Exception as e:
        tb = traceback.format_exc()
        _p(f"Failed to load manifest: {e}\n{tb}")
        raise

    # compute average silhouette across clustered segments
    segs = manifest.get("segments", {})
    sils = []
    details = {}
    for seg, info in segs.items():
        try:
            status = info.get("status")
            best_sil = info.get("best_silhouette")
            if status == "CLUSTERED" and best_sil is not None:
                sils.append(float(best_sil))
                details[seg] = {"status": status, "best_silhouette": float(best_sil)}
            else:
                details[seg] = {"status": status, "best_silhouette": best_sil}
        except Exception:
            details[seg] = {"status": info.get("status"), "best_silhouette": info.get("best_silhouette")}

    seg_count = len([s for s in segs.keys()])
    clustered_count = len(sils)

    report = {
        "run_id": run_id,
        "bundle_gcs_uri": bundle_gcs_uri,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segments_total": seg_count,
        "clustered_segments": clustered_count,
        "per_segment": details,
        "avg_silhouette": None,
        "silhouette_threshold": float(silhouette_threshold),
        "status": "fail",
    }

    if clustered_count >= max(1, min_segments_required) and sils:
        avg = float(sum(sils) / len(sils))
        report["avg_silhouette"] = avg
        # rules:
        #   avg >= threshold -> ok
        #   avg >= (threshold * 0.8) -> warn
        #   else -> fail
        if avg >= float(silhouette_threshold):
            report["status"] = "ok"
        elif avg >= float(silhouette_threshold) * 0.8:
            report["status"] = "warn"
        else:
            report["status"] = "fail"
    else:
        # not enough clustered segments -> fail
        report["status"] = "fail"

    # write report to gcs_report_path (best-effort)
    if gcs_report_path and gcs_report_path.startswith("gs://"):
        try:
            _, rest = gcs_report_path.split("gs://", 1)
            bucket_name2, prefix2 = (rest.split("/", 1) + [""])[:2]
            prefix2 = prefix2.rstrip("/") if prefix2 else ""
            client2 = storage.Client()
            bucket2 = client2.bucket(bucket_name2)
            blob_path = f"{prefix2}/{run_id}/model_validation_{run_id}.json" if prefix2 else f"{run_id}/model_validation_{run_id}.json"
            bucket2.blob(blob_path).upload_from_string(json.dumps(report, default=str), content_type="application/json")
            _p(f"Wrote validation report to gs://{bucket_name2}/{blob_path}")
        except Exception as e:
            _p(f"Non-fatal: failed to write validation report to GCS: {e}")

    _p(f"Validation status: {report['status']}")
    return report["status"]
