# src/pipeline/components/score_clusters_from_bq.py
import os
from kfp import dsl

IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE",
    "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:5541009",
)

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "google-cloud-bigquery>=3.14.0",
        "google-cloud-storage>=2.16.0",
        "pandas>=2.2.2",
        "numpy>=1.26.4",
        "db-dtypes>=1.2.0",
        "scikit-learn>=1.4.2",
        "joblib>=1.3.2",
    ],
)
def score_clusters_from_bq(
    bq_project: str,
    bq_dataset: str,
    # accept both names so it fits any caller
    preprocessed_table: str = "",
    bq_preprocessed_table: str = "",
    # model location: accept either gs://... or bucket/path...
    model_dir_base_gcs: str = "gs://gcs-mlops-setup-prj-hds-np-data-unique/models/sow-clusters/run_20250913_203500_20250913_203500",
    model_run_id: str = "latest",  # "latest" = pick latest run folder under model_dir_base_gcs
    # output
    out_clustered_table: str = "SOW_CLUSTERED_DATA_TEST",
) -> str:
    """
    Scores preprocessed BQ table with clustering artifacts.
    Accepts model bundles saved as either:
      - model.joblib (contains {'scaler':..., 'kmeans':...}) OR
      - manifest.json + segment_<seg>.json with centroids + scaler info
    Writes output to BigQuery table (WRITE_TRUNCATE).
    """
    import json, pathlib, time, traceback
    from datetime import datetime, timezone
    import os

    import numpy as np
    import pandas as pd
    from google.cloud import bigquery, storage

    def _p(m: str):
        print(f"[SCORE] {m}", flush=True)

    # ---------- Validate input ----------
    source_table = (preprocessed_table or bq_preprocessed_table or "").strip()
    if not source_table:
        raise ValueError("Provide preprocessed_table or bq_preprocessed_table.")

    bq = bigquery.Client(project=bq_project)
    ds = f"{bq_project}.{bq_dataset}"
    src = f"{ds}.{source_table}"
    dst = f"{ds}.{out_clustered_table}"
    _p(f"Scoring from `{src}` -> `{dst}`")

    # ---------- helpers ----------
    def _parse_gs_path(gs_path: str):
        # Accepts "gs://bucket/prefix" or "bucket/prefix" or "bucket" or "/bucket/prefix"
        s = gs_path.strip()
        if s.startswith("gs://"):
            s = s[len("gs://"):]
        s = s.lstrip("/")
        bucket, sep, prefix = s.partition("/")
        prefix = prefix.rstrip("/")
        return bucket, prefix

    def _exists_blob(bucket_obj, name: str):
        try:
            return bucket_obj.blob(name).exists()
        except Exception:
            return False

    try:
        # ---------- normalize base path ----------
        base = model_dir_base_gcs.rstrip("/") if model_dir_base_gcs else ""
        bucket, base_prefix = _parse_gs_path(base)
        _p(f"Model base -> bucket: '{bucket}' prefix: '{base_prefix}'")

        gcs = storage.Client(project=bq_project)
        bkt = gcs.bucket(bucket)

        # ---------- locate artifact run folder ----------
        if model_run_id and model_run_id.lower() != "latest":
            run_id = model_run_id
            artifact_prefix = f"{base_prefix}/{run_id}".strip("/")

            # quick check: does manifest or model.joblib exist here?
            if not (_exists_blob(bkt, f"{artifact_prefix}/manifest.json") or _exists_blob(bkt, f"{artifact_prefix}/model.joblib")):
                # if not, maybe user passed a full path including the run folder already; try base as run folder
                if _exists_blob(bkt, f"{base_prefix}/manifest.json") or _exists_blob(bkt, f"{base_prefix}/model.joblib"):
                    artifact_prefix = base_prefix
                    run_id = artifact_prefix.split("/")[-1]
                else:
                    raise ValueError(f"Requested run_id '{model_run_id}' not found under gs://{bucket}/{base_prefix}")
        else:
            # find run_* style folders under base_prefix (delimiter="/")
            iterator = gcs.list_blobs(bucket, prefix=(base_prefix + "/") if base_prefix else "", delimiter="/")
            prefixes = []
            # collect page prefixes
            try:
                for page in iterator.pages:
                    # page.prefixes is an iterable of directory-like prefixes
                    pfxs = list(page.prefixes)
                    prefixes.extend(pfxs)
            except Exception:
                # older client behavior: try listing blobs and parse
                blobs = list(gcs.list_blobs(bucket, prefix=(base_prefix + "/") if base_prefix else ""))
                # extract run-like prefixes
                for b in blobs:
                    # take everything before the last slash for each blob name
                    if "/" in b.name:
                        pfx = b.name.split("/")[0] + "/"
                        prefixes.append(pfx)
                prefixes = list(set(prefixes))

            prefixes = sorted([p.rstrip("/") for p in prefixes if p and p.strip()])
            if not prefixes:
                # maybe base_prefix itself is a run folder (contains manifest.json)
                if _exists_blob(bkt, f"{base_prefix}/manifest.json") or _exists_blob(bkt, f"{base_prefix}/model.joblib"):
                    artifact_prefix = base_prefix
                    run_id = artifact_prefix.split("/")[-1]
                else:
                    raise ValueError(f"No model run folders found under gs://{bucket}/{base_prefix}")
            else:
                chosen = prefixes[-1]  # lexicographically latest
                # chosen might be like 'models/sow-clusters/run_2025...'
                artifact_prefix = chosen.rstrip("/")
                run_id = artifact_prefix.split("/")[-1]

        _p(f"Selected artifact prefix: gs://{bucket}/{artifact_prefix} (run_id={run_id})")

        # ---------- detect artifact type ----------
        model_joblib_blob = f"{artifact_prefix}/model.joblib"
        manifest_blob = f"{artifact_prefix}/manifest.json"
        has_joblib = _exists_blob(bkt, model_joblib_blob)
        has_manifest = _exists_blob(bkt, manifest_blob)

        model_type = None
        seg_map = None
        scaler = None
        kmeans = None
        features = None

        tmpdir = "/tmp/score_bundle"
        pathlib.Path(tmpdir).mkdir(parents=True, exist_ok=True)

        if has_joblib:
            # download joblib
            _p(f"Found model.joblib at {model_joblib_blob}. Downloading...")
            bkt.blob(model_joblib_blob).download_to_filename(tmpdir + "/model.joblib")
            import joblib
            artifacts = joblib.load(tmpdir + "/model.joblib")
            scaler = artifacts.get("scaler")
            kmeans = artifacts.get("kmeans")
            # optional features.json
            try:
                feats_txt = bkt.blob(f"{artifact_prefix}/features.json").download_as_text()
                features = json.loads(feats_txt).get("features")
            except Exception:
                features = None
            model_type = "joblib_kmeans"
            _p("Loaded joblib artifacts.")
        elif has_manifest:
            _p(f"Found manifest.json at {manifest_blob}. Downloading manifest and segment artifacts...")
            bkt.blob(manifest_blob).download_to_filename(tmpdir + "/manifest.json")
            manifest = json.load(open(tmpdir + "/manifest.json"))
            seg_map = {}
            for seg, info in manifest.get("segments", {}).items():
                art = info.get("artifact")
                if not art:
                    continue
                seg_blob_name = f"{artifact_prefix}/{art}"
                if _exists_blob(bkt, seg_blob_name):
                    local_path = os.path.join(tmpdir, art)
                    bkt.blob(seg_blob_name).download_to_filename(local_path)
                    seg_map[seg] = json.load(open(local_path))
            model_type = "json_centroids"
            _p(f"Loaded manifest and {len(seg_map)} segment artifacts.")
        else:
            raise ValueError(f"No model.joblib or manifest.json found under gs://{bucket}/{artifact_prefix}")

        # ---------- load BQ data ----------
        _p("Loading preprocessed data from BigQuery...")
        df = bq.query(f"SELECT * FROM `{src}`").result().to_dataframe(create_bqstorage_client=True)
        if df is None:
            df = pd.DataFrame()
        df.columns = [c.upper() for c in df.columns]
        _p(f"Loaded {len(df)} rows from {src}")

        # ---------- scoring ----------
        scored_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if model_type == "joblib_kmeans":
            _p("Scoring using joblib KMeans artifacts.")
            if features is None:
                raise ValueError("features.json not found in joblib bundle; cannot determine feature order")
            features_up = [f.upper() for f in features]
            # ensure columns exist & numeric (fill missing with median)
            for c in features_up:
                if c not in df.columns:
                    df[c] = np.nan
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median(skipna=True))

            X = df[features_up].to_numpy(dtype=float)
            if scaler is not None:
                try:
                    Xs = scaler.transform(X)
                except Exception:
                    # scaler may be dict -> fallback to elementwise cent/scale if present
                    if isinstance(scaler, dict):
                        center = np.array(scaler.get("center") or [0.0]*len(features_up))
                        scale = np.array(scaler.get("scale") or [1.0]*len(features_up))
                        Xs = (X - center) / scale
                    else:
                        Xs = X
            else:
                Xs = X

            if kmeans is None:
                raise ValueError("kmeans object not found inside joblib artifact")
            try:
                labels = kmeans.predict(Xs)
            except Exception as e:
                # If kmeans is a dict with centroids, fallback to nearest-centroid
                if isinstance(kmeans, dict) and "centroids" in kmeans:
                    centroids = np.array(kmeans["centroids"])
                    dists = np.linalg.norm(centroids[None, :] - Xs[:, None, :], axis=2)
                    labels = np.argmin(dists, axis=1)
                else:
                    raise

            out = df.copy()
            out["CLUSTER_LABEL"] = labels.astype("int64")
            out["RUNID"] = run_id
            out["SCORED_AT_UTC"] = scored_ts

        elif model_type == "json_centroids":
            _p("Scoring using JSON centroids (per-segment artifacts).")
            # For JSON path, segments may use same feature list (stored in seg_art["features"])
            # We'll build per-row label by applying per-segment centroid nearest neighbor.
            out_rows = []
            # ensure we have CUSTOMER_ID column if present
            cid_col = "CUSTOMER_ID" if "CUSTOMER_ID" in df.columns else None

            # helper to compute label for a row and a segment artifact
            def _assign_for_row(row, segment_art):
                feats = segment_art.get("features", [])
                feats_up = [f.upper() for f in feats]
                X = np.array([row.get(f, 0.0) for f in feats_up], dtype=float)
                scaler_j = segment_art.get("scaler", {})
                center = scaler_j.get("center")
                scale = scaler_j.get("scale")
                if scale is None:
                    scale = [1.0] * len(feats_up)
                scale = np.array(scale, dtype=float)
                if center is not None:
                    X = (X - np.array(center)) / scale
                else:
                    X = X / scale
                centroids = np.array(segment_art["centroids"])
                dists = np.linalg.norm(centroids - X, axis=1)
                return int(np.argmin(dists))

            # uppercase df columns for feature lookups
            # (already uppercased above)
            for idx, row in df.iterrows():
                seg = row.get("SEGMENT")
                cid = row.get("CUSTOMER_ID") if "CUSTOMER_ID" in row.index else None
                if seg not in seg_map:
                    out_rows.append({"CUSTOMER_ID": cid, "SEGMENT": seg, "CLUSTER_LABEL": -1})
                    continue
                try:
                    cl = _assign_for_row(row, seg_map[seg])
                except Exception:
                    cl = -1
                out_rows.append({"CUSTOMER_ID": cid, "SEGMENT": seg, "CLUSTER_LABEL": int(cl)})

            out = pd.DataFrame(out_rows)
            # add run metadata
            out["RUNID"] = run_id
            out["SCORED_AT_UTC"] = scored_ts

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # ---------- write results to BigQuery ----------
        _p(f"Writing {len(out)} rows to {dst}")
        job_cfg = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        bq.load_table_from_dataframe(out, dst, job_config=job_cfg).result()
        t = bq.get_table(dst)
        _p(f"Wrote {t.num_rows:,} rows to {dst}")

        return f"Scored with run_id={run_id} -> {dst} ({t.num_rows:,} rows)"

    except Exception:
        _p("FATAL ERROR during scoring:")
        traceback.print_exc()
        raise
