# README: Training Artifacts, Manifest, and Segment Files

## 1. What Happens During Training
When the **training pipeline** runs (`train_cluster_artifacts` component):

- Data is pulled from the **preprocessed Snowflake table** (per industry/segment).
- For each segment (e.g., `CONSUMER`, `COMMERCIAL`, etc.):
  - A clustering model is trained (KMeans/MiniBatchKMeans).
  - A **segment JSON file** is written containing:
    - Chosen number of clusters (`best_k`).
    - Best silhouette score.
    - Cluster centroids.
    - Feature list and scaler info.
- A **manifest.json** is written at the run folder root:
  - Summarizes the run (`run_id`, timestamp, config).
  - Lists all segments and references their artifact filenames.
  - Stores training metadata.

Additionally, a helper `score.py` file is created for local/inference scoring.

---

## 2. Folder Structure in GCS
After training completes, artifacts are uploaded to:

```
gs://<bucket>/models/sow-clusters/<run_id>_<timestamp>/
¦
+-- manifest.json
+-- score.py
+-- segment_CONSUMER.json
+-- segment_COMMERCIAL.json
+-- segment_HEALTHCARE.json
+-- ...
```

### Example: `manifest.json`
```json
{
  "run_id": "run_20250930_101500",
  "run_timestamp": "2025-09-30T10:15:00Z",
  "industries": ["CONSUMER", "COMMERCIAL"],
  "feature_cols": ["RECENCY_DAYS", "ORDER_FREQUENCY", "CUSTOMER_AGE_YEARS"],
  "segments": {
    "CONSUMER": {
      "status": "CLUSTERED",
      "customer_count": 5000,
      "best_k": 4,
      "best_silhouette": 0.31,
      "artifact": "segment_CONSUMER.json"
    },
    "COMMERCIAL": {
      "status": "CLUSTERED",
      "customer_count": 3000,
      "best_k": 3,
      "best_silhouette": 0.27,
      "artifact": "segment_COMMERCIAL.json"
    }
  }
}
```

---

## 3. How Inference Uses These Files
- **Inference components** (`score_clusters_from_bq`) load `manifest.json` from the GCS run folder.
- For each row in the input data:
  - Look up its segment.
  - Load the corresponding `segment_<SEG>.json`.
  - Apply stored `scaler` and `centroids` to assign a cluster label.
- Output is written to **BigQuery** or **Snowflake** with added `CLUSTER_LABEL`.

---

## 4. How SHAP Uses These Files
- **SHAP pipeline** (`shap_explain_component`) also loads the `manifest.json` and per-segment files.
- It uses the **features + centroids** info to compute SHAP values for clusters.
- Results are saved back to Snowflake + optionally plots/JSON are uploaded to GCS.

---

## 5. Why Manifest + Segment Separation?
- **Manifest** = the index / catalog of the run.
- **Segment files** = detailed per-segment clustering artifacts.
- Together they allow:
  - Easy validation (`model_validation_component` reads silhouette scores).
  - Efficient inference (only load segment file needed).
  - Explainability (SHAP runs per segment using features list).

---
