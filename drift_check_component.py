import os
from kfp import dsl

IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE",
    )

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "snowflake-snowpark-python[pandas]>=1.15.0",
        "python-dotenv>=1.0.1",
        "pandas>=2.2.2",
        "numpy>=1.26.4",
        "google-cloud-storage>=2.16.0",
    ],
)
def drift_check_component(
    # Snowflake / env (same style as other components)
    app_env: str,
    gcp_project_id: str,
    snowflake_account: str,
    snowflake_user: str,
    snowflake_database: str,
    snowflake_schema: str,
    snowflake_role: str,
    snowflake_warehouse: str,
    snowflake_password_secret_name: str,

    # Current table to evaluate (fully-qualified: DB.SCHEMA.TABLE)
    current_table_fq: str,

    # (Preferred) Baseline table to compare against (DB.SCHEMA.TABLE)
    baseline_table_fq: str = "",

    # Optional: baseline stats JSON (fallback mode; same shape as your feature-threshold baseline)
    baseline_gcs_path: str = "",

    # Feature list to evaluate (JSON string array) – if empty, we will infer ‘reasonable’ numeric columns
    feature_list_json: str = "",

    # Thresholds
    psi_warn_threshold: float = 0.10,
    psi_fail_threshold: float = 0.25,
    cluster_kl_warn: float = 0.05,
    cluster_kl_fail: float = 0.10,

    # Sampling caps (per table, to avoid memory blowups)
    max_rows_per_table: int = 200_000,

    # Outputs
    run_id: str = "",
    run_timestamp_utc: str = "",
    gcs_report_path: str = "gs://xxxxxxxxxxxxx/reports",
) -> str:
    """
    Simple drift monitor.

    Mode A (recommended): compare CURRENT Snowflake table vs BASELINE Snowflake table.
      - For each numeric feature:
          * build 10 quantile bins from the *baseline* distribution,
          * compute PSI(current || baseline)
      - If `CLUSTER_LABEL` exists in both tables, compute cluster share KL divergence.

    Mode B (fallback): baseline_gcs_path only (no baseline table).
      - Runs lightweight checks vs baseline stats JSON (mean / null deltas). PSI is skipped.

    Writes a JSON report to GCS:
      gs://.../<reports_prefix>/<run_id>/drift_report_<run_id>.json

    Returns: status string "ok" | "warn" | "fail"
    """
    import json, time, traceback, math
    from pathlib import Path
    import numpy as np
    import pandas as pd

    def _p(m: str):
        print(f"[DRIFT] {m}", flush=True)

    # -----------------------------
    # Write connector .env like the others
    # -----------------------------
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
    _p(f"Wrote env file: {env_path}")

    # -----------------------------
    # Import Snowflake connector (robust path search)
    # -----------------------------
    import sys, importlib, os as _os
    from pathlib import Path as _Path
    for p in ("/app", "/app/src", _os.getcwd(), str(_Path(_os.getcwd()).parent)):
        if p and _os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    def _import_attr(cands):
        for mod, attr in cands:
            try:
                m = importlib.import_module(mod)
                return getattr(m, attr)
            except Exception:
                continue
        raise ImportError(f"Could not import any of: {cands}; sys.path[:6]={sys.path[:6]}")

    try:
        SnowflakeConnector = _import_attr([
            ("src.data.connect", "SnowflakeConnector"),
            ("data.connect", "SnowflakeConnector"),
            ("connect", "SnowflakeConnector"),
        ])
    except Exception as e:
        _p(f"Import SnowflakeConnector failed: {e}")
        traceback.print_exc()
        raise

    # -----------------------------
    # Helpers
    # -----------------------------
    def _load_table_sample(session, table_fq: str, cols: list, limit: int) -> pd.DataFrame:
        sel = ", ".join(cols)
        q = f"SELECT {sel} FROM {table_fq}"
        if limit and limit > 0:
            q += f" LIMIT {int(limit)}"
        _p(f"Sampling: {q}")
        try:
            return session.sql(q).to_pandas()
        except Exception as e:
            _p(f"Sampling failed for {table_fq}: {e}")
            raise

    def _infer_numeric_columns(session, table_fq: str, candidate_limit: int = 2000) -> list:
        # read a small sample and detect numeric columns
        try:
            pdf = _load_table_sample(session, table_fq, ["*"], candidate_limit)
            nums = pdf.select_dtypes(include=[np.number]).columns.tolist()
            # remove obvious non-features
            drop_like = {"CLUSTER_LABEL"}
            nums = [c for c in nums if c.upper() not in drop_like]
            return nums
        except Exception:
            return []

    def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        expected = np.clip(expected, eps, 1.0)
        actual = np.clip(actual, eps, 1.0)
        return float(np.sum((actual - expected) * np.log(actual / expected)))

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        eps = 1e-8
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        return float(np.sum(p * np.log(p / q)))

    def _write_gcs_json(bucket_path: str, name: str, obj: dict):
        if not bucket_path or not bucket_path.startswith("gs://"):
            return None
        try:
            from google.cloud import storage
            _, rest = bucket_path.split("gs://", 1)
            bucket_name, *prefix_parts = rest.split("/", 1)
            prefix = prefix_parts[0].rstrip("/") if prefix_parts else ""
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob_path = f"{prefix}/{run_id}/{name}" if prefix else f"{run_id}/{name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(json.dumps(obj, default=str, indent=2), content_type="application/json")
            return f"gs://{bucket_name}/{blob_path}"
        except Exception as e:
            _p(f"GCS write failed: {e}")
            return None

    # -----------------------------
    # Connect to Snowflake
    # -----------------------------
    session = None
    try:
        conn = SnowflakeConnector(app_env=app_env)
        session = conn.create_snowflake_session()
        _p("Snowpark session created")
    except Exception as e:
        _p(f"Cannot create Snowflake session: {e}")
        raise

    # -----------------------------
    # Determine features
    # -----------------------------
    import os
    import json as _json
    try:
        feats = _json.loads(feature_list_json) if feature_list_json else []
        if feats and not isinstance(feats, list):
            feats = []
    except Exception:
        feats = []

    # If no explicit features, infer from current table
    if not feats:
        feats = _infer_numeric_columns(session, current_table_fq, 2000)
        _p(f"Inferred numeric features: {feats}")

    # Always consider cluster share drift if present
    want_cluster_drift = True

    report = {
        "run_id": run_id,
        "timestamp_utc": run_timestamp_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "baseline_table" if baseline_table_fq else ("baseline_json" if baseline_gcs_path else "unknown"),
        "current_table": current_table_fq,
        "baseline_table": baseline_table_fq or None,
        "status": "ok",
        "psi": {},                # per-feature PSI (if baseline_table)
        "feature_notes": {},      # notes for skipped/missing
        "cluster_share": {},      # cluster share distributions + KL
        "thresholds": {
            "psi_warn": float(psi_warn_threshold),
            "psi_fail": float(psi_fail_threshold),
            "cluster_kl_warn": float(cluster_kl_warn),
            "cluster_kl_fail": float(cluster_kl_fail),
        }
    }

    overall = "ok"

    try:
        if baseline_table_fq:
            # --------- Mode A: table vs table (PSI + cluster share KL) ----------
            # Build sample columns
            cols_needed = list(set(feats + (["CLUSTER_LABEL"] if want_cluster_drift else [])))
            # sample both
            cur_df = _load_table_sample(session, current_table_fq, cols_needed, max_rows_per_table)
            base_df = _load_table_sample(session, baseline_table_fq, cols_needed, max_rows_per_table)

            # Ensure numeric for feature columns
            for c in feats:
                cur_df[c] = pd.to_numeric(cur_df.get(c), errors="coerce")
                base_df[c] = pd.to_numeric(base_df.get(c), errors="coerce")

            # Per-feature PSI using 10 bins determined by BASELINE quantiles
            for f in feats:
                try:
                    base_series = base_df[f].dropna()
                    cur_series = cur_df[f].dropna()
                    if len(base_series) < 50 or len(cur_series) < 50:
                        report["feature_notes"][f] = "insufficient_samples"
                        continue

                    # bin edges from baseline deciles
                    qs = np.linspace(0.0, 1.0, 11)
                    edges = np.unique(np.quantile(base_series.values, qs))
                    if len(edges) < 3:
                        report["feature_notes"][f] = "low_variance_baseline"
                        continue

                    # histogram proportions
                    # np.histogram returns counts; use density to get proportions
                    base_hist, _ = np.histogram(base_series.values, bins=edges)
                    cur_hist, _  = np.histogram(cur_series.values,  bins=edges)
                    # convert to proportions
                    base_p = base_hist / max(1, base_hist.sum())
                    cur_p  = cur_hist / max(1, cur_hist.sum())

                    psi_val = _psi(base_p, cur_p)
                    report["psi"][f] = float(psi_val)

                    if psi_val >= psi_fail_threshold:
                        overall = "fail"
                    elif psi_val >= psi_warn_threshold and overall != "fail":
                        overall = "warn"

                except Exception as e:
                    report["feature_notes"][f] = f"error:{e}"

            # Cluster share drift (if CLUSTER_LABEL present in both)
            if "CLUSTER_LABEL" in cur_df.columns and "CLUSTER_LABEL" in base_df.columns:
                try:
                    cur_counts = cur_df["CLUSTER_LABEL"].value_counts(dropna=False).sort_index()
                    base_counts = base_df["CLUSTER_LABEL"].value_counts(dropna=False).sort_index()
                    # align indexes
                    all_idx = sorted(set(cur_counts.index).union(set(base_counts.index)))
                    cur_vec = np.array([cur_counts.get(i, 0) for i in all_idx], dtype=float)
                    base_vec = np.array([base_counts.get(i, 0) for i in all_idx], dtype=float)
                    cur_p = cur_vec / max(1.0, cur_vec.sum())
                    base_p = base_vec / max(1.0, base_vec.sum())
                    kl = _kl(cur_p, base_p)
                    report["cluster_share"] = {
                        "labels": [int(i) if (isinstance(i, (int, np.integer)) or (isinstance(i, float) and i.is_integer())) else str(i) for i in all_idx],
                        "current": cur_p.tolist(),
                        "baseline": base_p.tolist(),
                        "kl_divergence": float(kl),
                    }
                    if kl >= cluster_kl_fail:
                        overall = "fail"
                    elif kl >= cluster_kl_warn and overall != "fail":
                        overall = "warn"
                except Exception as e:
                    report["cluster_share"] = {"error": str(e)}
            else:
                report["cluster_share"] = {"note": "CLUSTER_LABEL not available in both tables"}

        elif baseline_gcs_path:
            # --------- Mode B: baseline JSON fallback (no PSI) ----------
            from google.cloud import storage
            try:
                _, rest = baseline_gcs_path.split("gs://", 1)
                bucket_name, *prefix_parts = rest.split("/", 1)
                prefix = prefix_parts[0] if prefix_parts else ""
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(prefix)
                baseline = json.loads(blob.download_as_text())
            except Exception as e:
                raise RuntimeError(f"Failed to load baseline JSON: {e}")

            # Pull a sample of current table and compute mean/null deltas
            cur_df = _load_table_sample(session, current_table_fq, ["*"], max_rows_per_table)

            notes = {}
            checks = {}
            for f, b in baseline.items():
                if f not in cur_df.columns:
                    notes[f] = "missing_in_current"
                    overall = "fail"
                    continue
                s = pd.to_numeric(cur_df[f], errors="coerce")
                cur_mean = float(np.nanmean(s.values)) if np.isfinite(np.nanmean(s.values)) else None
                cur_null = float(np.mean(pd.isna(s.values)))
                base_mean = b.get("mean")
                base_null = b.get("null_pct", 0.0)
                mean_dev = None if (cur_mean is None or base_mean is None) else (0.0 if base_mean == 0 else abs(cur_mean - base_mean) / (abs(base_mean) + 1e-9))
                null_delta = cur_null - (base_null or 0.0)

                status = "ok"
                reason = []
                if mean_dev is not None and mean_dev > 0.5:   # loose 50% relative change => warn
                    status = "warn"; reason.append({"metric":"mean_dev","value":mean_dev,"thresh":0.5})
                if null_delta > 0.30:                          # +30pp nulls => fail
                    status = "fail"; reason.append({"metric":"null_delta","value":null_delta,"thresh":0.30})

                if status == "fail": overall = "fail"
                elif status == "warn" and overall != "fail": overall = "warn"

                checks[f] = {
                    "baseline_mean": base_mean,
                    "current_mean": cur_mean,
                    "baseline_null_pct": base_null,
                    "current_null_pct": cur_null,
                    "mean_dev_rel": mean_dev,
                    "null_delta": null_delta,
                    "status": status,
                    "reasons": reason,
                }

            report["fallback_checks"] = checks
            report["feature_notes"] = notes

        else:
            raise ValueError("Provide either baseline_table_fq (preferred) or baseline_gcs_path.")

    except Exception as e:
        report["error"] = f"{e}"
        overall = "fail"
        traceback.print_exc()

    report["status"] = overall

    # Write report to GCS
    out_uri = _write_gcs_json(gcs_report_path, f"drift_report_{run_id}.json", report)
    if out_uri:
        _p(f"Wrote drift report: {out_uri}")
    else:
        _p("Drift report not written to GCS (no/invalid gcs_report_path).")

    _p(f"Overall drift status: {overall}")
    return overall
