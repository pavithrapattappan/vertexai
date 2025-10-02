# src/pipeline/components/train_cluster_component.py
import os
from kfp import dsl

IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE",
    "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:de50c50",
)

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "snowflake-snowpark-python[pandas]>=1.15.0",
        "pandas>=2.2.2",
        "numpy>=1.25",
        "scikit-learn>=1.2.2",
        "google-cloud-storage>=2.10.0",
        "python-dotenv>=1.0.1",
    ],
)
def train_cluster_artifacts(
    # Snowflake / env
    app_env: str,
    gcp_project_id: str,
    snowflake_account: str,
    snowflake_user: str,
    snowflake_database: str,
    snowflake_schema: str,
    snowflake_role: str,
    snowflake_warehouse: str,
    snowflake_password_secret_name: str,

    # input (preprocessed table, fully-qualified)
    input_table_fq: str,  # e.g. "EDP_ML_DEV.SOW.SOW_FEATURES_PREPROCESSED_..."

    # options
    industries_json: str = '["COMMERCIAL","CONSUMER","HEALTHCARE","HOSPITALITY","MULTIFAMILY","PUBLIC SECTOR","INDUSTRIAL"]',
    feature_cols_json: str = '["RECENCY_DAYS","ORDER_FREQUENCY","CUSTOMER_AGE_YEARS","AVG_ORDER_VALUE"]',
    clusters_min: int = 2,
    clusters_max: int = 10,
    min_customers_for_cluster: int = 60,
    training_max_rows: int = 0,   # 0 = all rows (cap is applied PER industry)

    # output
    model_gcs_prefix: str = "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters",

    # run metadata
    run_id: str = "",
    run_timestamp_utc: str = "",
) -> str:
    """
    Train per-segment clustering artifacts and upload a bundle to GCS.
    - Streams data PER INDUSTRY to avoid pulling the full table to memory at once.
    - For each segment in each industry:
        * scales features (RobustScaler)
        * searches k in [clusters_min..clusters_max] (bounded by data size)
        * chooses best_k via silhouette score
        * saves segment artifact json + updates manifest
    Returns: bundle_uri (gs://...)
    """
    import os
    import json, time, traceback
    from pathlib import Path

    def _p(msg: str):
        print(f"[TRAIN] {msg}", flush=True)

    # 1) write connector env so existing connect.py can find variables
    env_path = f".env.{app_env}"
    with open(env_path, "w") as f:
        f.write(
            f"GCP_PROJECT_ID={gcp_project_id}\n"
            f"SNOWFLAKE_PASSWORD_SECRET_NAME={snowflake_password_secret_name}\n"
            f"SNOWFLAKE_ACCOUNT={snowflake_account}\n"
            f"SNOWFLAKE_USER={snowflake_user}\n"
            f"SNOWFLAKE_DATABASE={snowflake_database}\n"
            f"SNOWFLAKE_SCHEMA={snowflake_schema}\n"
            f"SNOWFLAKE_ROLE={snowflake_role}\n"
            f"SNOWFLAKE_WAREHOUSE={snowflake_warehouse}\n"
        )
    _p(f"Wrote env: {env_path}")

    # 2) ensure repo paths are importable
    import sys, os
    for p in ("/app", "/app/src", os.getcwd(), str(Path(os.getcwd()).parent)):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # 3) import SnowflakeConnector
    try:
        from src.data.connect import SnowflakeConnector
    except Exception:
        try:
            from data.connect import SnowflakeConnector
        except Exception as e:
            _p(f"Failed to import SnowflakeConnector: {e}")
            raise

    # ML & storage imports
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import RobustScaler
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    from google.cloud import storage

    # parse JSON params
    industries = json.loads(industries_json)
    feature_cols = json.loads(feature_cols_json)

    # artifact local folder
    tmp_root = Path("/tmp/train_cluster_artifacts")
    tmp_root.mkdir(parents=True, exist_ok=True)

    run_stamp = run_timestamp_utc or time.strftime("%Y%m%d_%H%M%S")
    run_name = run_id or f"run_{run_stamp}"
    artifact_folder = tmp_root / f"{run_name}_{run_stamp}"
    artifact_folder.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_name,
        "run_timestamp": run_stamp,
        "industries": industries,
        "feature_cols": feature_cols,
        "clusters_min": clusters_min,
        "clusters_max": clusters_max,
        "min_customers_for_cluster": min_customers_for_cluster,
        "segments": {},  # key: segment -> info
    }

    session = None
    try:
        conn = SnowflakeConnector(app_env=app_env)
        session = conn.create_snowflake_session()
        _p("Snowpark session created")
        _p(f"Training from: {input_table_fq}")

        from snowflake.snowpark.functions import col as F_col

        # Process ONE INDUSTRY AT A TIME to keep memory low
        for ind in industries:
            ind_u = ind.upper()
            _p(f"--- Processing industry: {ind_u} ---")

            # Pull just this industry's rows & selected columns
            sel_cols = ["CUSTOMER_ID", "SEGMENT", "MOST_RECENT_INDUSTRY_CODE_DESC"] + feature_cols
            sf_df = (
                session.table(input_table_fq)
                .filter(F_col("MOST_RECENT_INDUSTRY_CODE_DESC") == ind_u)
                .select(*sel_cols)
            )

            if training_max_rows and int(training_max_rows) > 0:
                _p(f"[{ind_u}] Row cap applied: {training_max_rows}")
                sf_df = sf_df.limit(int(training_max_rows))

            _p(f"[{ind_u}] Converting to pandas (streamed subset).")
            pdf = sf_df.to_pandas()
            _p(f"[{ind_u}] Rows loaded: {len(pdf):,}")

            if pdf is None or len(pdf) == 0:
                _p(f"[{ind_u}] No data, skipping industry.")
                continue

            # Ensure types
            if "SEGMENT" in pdf.columns:
                pdf["SEGMENT"] = pdf["SEGMENT"].astype(str)

            segments = pdf["SEGMENT"].dropna().unique().tolist() if "SEGMENT" in pdf.columns else []
            _p(f"[{ind_u}] Segments found: {len(segments)}")

            # Iterate segments (still small enough per industry)
            for seg in segments:
                seg_df = pdf[pdf["SEGMENT"] == seg].copy()
                n = len(seg_df)
                key = f"{ind_u}::{seg}"
                manifest["segments"][key] = {"customer_count": int(n)}
                _p(f"[SEG] {key} customers={n}")

                if n < min_customers_for_cluster:
                    manifest["segments"][key].update({"status": "SKIPPED_TOO_SMALL"})
                    _p(f"Skipping {key}: below min customers {min_customers_for_cluster}")
                    continue

                # Prepare feature matrix
                if not all(c in seg_df.columns for c in feature_cols):
                    manifest["segments"][key].update({"status": "MISSING_FEATURES"})
                    _p(f"Skipping {key}: missing requested features")
                    continue

                X = seg_df[feature_cols].copy()
                X = X.replace([np.inf, -np.inf], np.nan).dropna()
                if len(X) < min_customers_for_cluster:
                    manifest["segments"][key].update({"status": "SKIPPED_AFTER_DROPOFF"})
                    _p(f"Skipping {key} after dropping NA rows (now {len(X)})")
                    continue

                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)

                # k candidates bounded by size
                max_k_try = min(clusters_max, max(2, int(len(X_scaled) // 2)))
                k_candidates = list(range(max(clusters_min, 2), max_k_try + 1))
                if not k_candidates:
                    manifest["segments"][key].update({"status": "NO_K_CANDIDATES"})
                    _p(f"No k candidates for {key}")
                    continue

                best_k = None
                best_score = -999.0
                results_k = []

                for k in k_candidates:
                    try:
                        if len(X_scaled) > 20000:
                            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
                        else:
                            km = KMeans(n_clusters=k, random_state=42, n_init=10)

                        labels = km.fit_predict(X_scaled)

                        # Only compute silhouette when it's meaningful
                        if k > 1 and len(np.unique(labels)) > 1:
                            s = silhouette_score(X_scaled, labels)
                        else:
                            s = -1.0

                        results_k.append({"k": int(k), "silhouette": float(s)})

                        if s > best_score:
                            best_score = float(s)
                            best_k = int(k)
                    except Exception as e:
                        _p(f"[{ind_u}] {seg}: error k={k}: {e}")
                        continue

                if best_k is None:
                    manifest["segments"][key].update({"status": "NO_VALID_K", "results_k": results_k})
                    _p(f"[{ind_u}] {seg}: no valid k")
                    continue

                _p(f"[{ind_u}] {seg}: best k => {best_k} (silhouette={best_score:.4f})")

                if len(X_scaled) > 20000:
                    final_km = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=1024)
                else:
                    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)

                final_km.fit(X_scaled)
                centroids = final_km.cluster_centers_.tolist()

                # segment artifact
                seg_art = {
                    "status": "CLUSTERED",
                    "industry": ind_u,
                    "segment": seg,
                    "customer_count": int(n),
                    "best_k": int(best_k),
                    "best_silhouette": float(best_score),
                    "results_k": results_k,
                    "features": feature_cols,
                    "scaler": {
                        "center": (getattr(scaler, "center_", None).tolist() if hasattr(scaler, "center_") else None),
                        "scale": getattr(scaler, "scale_", None).tolist() if hasattr(scaler, "scale_") else None,
                    },
                    "centroids": centroids,
                }

                safe_seg = str(seg).replace(" ", "_")
                seg_fname = artifact_folder / f"segment_{ind_u}_{safe_seg}.json"
                with open(seg_fname, "w") as fh:
                    json.dump(seg_art, fh, indent=2)

                manifest["segments"][key].update({
                    "status": "CLUSTERED",
                    "artifact": seg_fname.name,
                    "best_k": int(best_k),
                    "best_silhouette": float(best_score),
                })

        # write manifest
        manifest_path = artifact_folder / "manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        _p(f"Wrote manifest: {manifest_path}")

        # write inline scorer (score.py) into artifact folder
        score_py_path = artifact_folder / "score.py"
        score_py_code = r'''
import json, numpy as np, pandas as pd

def assign_cluster_for_row(row_features: pd.Series, segment_art: dict):
    feats = segment_art["features"]
    X = np.array([row_features.get(f, 0.0) for f in feats], dtype=float)
    scaler = segment_art.get("scaler", {})
    center = scaler.get("center")
    scale = scaler.get("scale")
    if scale is None:
        scale = [1.0] * len(feats)
    if center is not None:
        X = (X - np.array(center)) / np.array(scale)
    else:
        X = X / np.array(scale)
    centroids = np.array(segment_art["centroids"])
    dists = np.linalg.norm(centroids - X, axis=1)
    cluster_idx = int(np.argmin(dists))
    return cluster_idx

def score_batch(df: pd.DataFrame, manifest_local_folder: str):
    m = json.load(open(manifest_local_folder.rstrip("/") + "/manifest.json"))
    seg_map = {}
    for seg_key, info in m.get("segments", {}).items():
        if "artifact" not in info: 
            continue
        segjson = manifest_local_folder.rstrip("/") + "/" + info["artifact"]
        seg_map[seg_key] = json.load(open(segjson))

    out_rows = []
    for _, row in df.iterrows():
        key = f"{str(row.get('MOST_RECENT_INDUSTRY_CODE_DESC','')).upper()}::{row.get('SEGMENT')}"
        if key not in seg_map:
            out_rows.append({"CUSTOMER_ID": row.get("CUSTOMER_ID"), "SEGMENT_KEY": key, "CLUSTER_LABEL": -1})
            continue
        try:
            cl = assign_cluster_for_row(row, seg_map[key])
        except Exception:
            cl = -1
        out_rows.append({"CUSTOMER_ID": row.get("CUSTOMER_ID"), "SEGMENT_KEY": key, "CLUSTER_LABEL": int(cl)})
    return pd.DataFrame(out_rows)
'''
        with open(score_py_path, "w") as fh:
            fh.write(score_py_code)
        _p(f"Wrote scorer helper to {score_py_path}")

        # upload artifacts to GCS
        client = storage.Client()
        prefix_full = model_gcs_prefix.rstrip("/")
        if not prefix_full.startswith("gs://"):
            raise ValueError("model_gcs_prefix must be a gs:// path")
        bucket_name, *rest = prefix_full.replace("gs://", "").split("/", 1)
        prefix = rest[0] if rest else ""
        blob_prefix = f"{prefix}/{artifact_folder.name}" if prefix else artifact_folder.name
        bucket = client.bucket(bucket_name)

        uploaded = []
        for f in sorted(artifact_folder.iterdir()):
            blob_name = f"{blob_prefix}/{f.name}"
            bucket.blob(blob_name).upload_from_filename(str(f))
            uploaded.append(f"gs://{bucket_name}/{blob_name}")
            _p(f"Uploaded {f.name} -> gs://{bucket_name}/{blob_name}")

        bundle_uri = f"gs://{bucket_name}/{blob_prefix}/"
        summary = {
            "manifest": "manifest.json",
            "files": [p.split(f"gs://{bucket_name}/")[-1] for p in uploaded],
            "bundle_uri": bundle_uri,
        }
        bucket.blob(f"{blob_prefix}/summary.json").upload_from_string(
            json.dumps(summary, indent=2), content_type="application/json"
        )

        _p(f"Artifact bundle uploaded to {bundle_uri}")
        return bundle_uri

    except Exception:
        _p("FATAL ERROR during training")
        traceback.print_exc()
        raise
    finally:
        if session is not None:
            try:
                session.close(); _p("Snowpark session closed")
            except Exception:
                pass
