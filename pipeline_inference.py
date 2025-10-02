from kfp import dsl

@dsl.pipeline(name="sow-batch-infer")
def sow_batch_infer_pipeline(
    # BigQuery
    bq_project: str = "prj-hds-np-data",
    bq_dataset: str = "SOW_TEST",
    bq_features_table: str = "SOW_FEATURES_TEST",
    bq_preprocessed_table: str = "SOW_FEATURES_PREP_TEST",
    bq_scored_table: str = "SOW_CLUSTERED_DATA_TEST",

    # Model artifacts
    model_dir_base_gcs: str = "gs://gcs-mlops-setup-prj-hds-np-data-unique/models",
    model_run_id: str = "latest",      # "latest" or an explicit run_id string

    # Refresh preprocessing first?
    refresh_preprocess: bool = True,
):
    from src.pipeline.components.preprocess_component import preprocess_in_bigquery
    from src.pipeline.components.score_clusters_component import score_clusters_from_bq

    if refresh_preprocess:
        prep = preprocess_in_bigquery(
            bq_project=bq_project,
            bq_dataset=bq_dataset,
            bq_features_table=bq_features_table,
            bq_preprocessed_table=bq_preprocessed_table,
        )

    score = score_clusters_from_bq(
        bq_project=bq_project,
        bq_dataset=bq_dataset,
        bq_preprocessed_table=bq_preprocessed_table,
        model_dir_base_gcs=model_dir_base_gcs,
        model_run_id=model_run_id,
        out_clustered_table=bq_scored_table,
    )

    if refresh_preprocess:
        score.after(prep)
