# src/pipeline/components/register_model_component.py
import os
from kfp import dsl

IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE"
)

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "google-cloud-aiplatform>=1.28.0",
        "python-dotenv>=1.0.1",
    ],
)
def register_model_component(
    project: str,
    location: str,
    model_display_name: str,

    # EITHER pass artifact_gcs_uri directly...
    artifact_gcs_uri: str = "",

    # ...OR let the component build it from these:
    model_gcs_prefix: str = "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters",
    run_id: str = "",
    run_timestamp_utc: str = "",

    serving_container_image_uri: str = "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:de50c50",
    labels_json: str = "{}",
    description: str = "",
) -> str:
    """
    Registers the artifact bundle in Vertex Model Registry and returns the model resource name.
    If artifact_gcs_uri is empty, we build:
      gs://.../<model_gcs_prefix>/<run_id>_<run_timestamp_utc>/
    """
    import os
    import json, traceback
    from google.cloud import aiplatform

    def _p(m: str):
        print(f"[REGISTER] {m}", flush=True)

    try:
        # Build the bundle URI here if not provided
        bundle_uri = artifact_gcs_uri or f"{model_gcs_prefix.rstrip('/')}/{run_id}_{run_timestamp_utc}/"

        _p(f"Init Vertex: project={project}, location={location}")
        aiplatform.init(project=project, location=location)

        labels = json.loads(labels_json) if labels_json else {}

        _p(f"Uploading model metadata to Vertex (artifact_uri={bundle_uri})")
        _p(f"Serving container image: {serving_container_image_uri}")

        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=bundle_uri,
            serving_container_image_uri=serving_container_image_uri,
            labels=labels,
            description=description,
        )

        model_resource_name = getattr(model, "resource_name", None) or getattr(model, "name", None)
        _p(f"Model uploaded: {model_resource_name}")
        return str(model_resource_name)

    except Exception:
        _p("FATAL ERROR registering model:")
        traceback.print_exc()
        raise
