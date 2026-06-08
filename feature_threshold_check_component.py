# src/pipeline/components/feature_threshold_check_component.py
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
        "pandas>=2.0.0",
        "google-cloud-storage>=2.8.0",
    ],
)
def feature_threshold_check_component(
    # Snowflake connector/env inputs (same pattern as other components)
    app_env: str,
    gcp_project_id: str,
    snowflake_account: str,
    snowflake_user: str,
    snowflake_database: str,
    snowflake_schema: str,
    snowflake_role: str,
    snowflake_warehouse: str,
    snowflake_password_secret_name: str,

    # table and baseline
    features_table_fq: str,               # e.g. "EDP_ML_DEV.SOW.SOW_FEATURES_..."
    baseline_gcs_path: str,               # e.g. "gs://.../run_.../baseline.json"
    run_id: str,
    run_timestamp_utc: str,

    # behavior / thresholds
    flex_pct: float = 0.5,                # default 50% multiplicative tolerance
    per_feature_overrides: str = "",      # optional JSON string to override flex_pct per feature
    null_warn_delta: float = 0.10,        # absolute pp increase -> warn (10%)
    null_fail_delta: float = 0.30,        # absolute pp increase -> fail (30%)

    # output
    gcs_report_path: str = "",            # e.g. "gs://bucket/pipeline-reports"
    sample_rows_for_fail: int = 10,       # include up to N sample rows for failing features
) -> str:
    """
    Compare run feature stats against baseline ranges with multiplicative flexibility.
    Returns JSON string with {"status": "ok|warn|fail", "report_gcs_path": "gs://..."} and writes a report to GCS.

    Instrumentation added (non-intrusive):
      - logs features_table_fq & baseline_gcs_path at start
      - writes started_{run_id}.txt to gcs_report_path/{run_id}/ if gcs_report_path set
      - on exception writes error_{run_id}.txt with traceback to same GCS folder
    """

    import os
    import json, sys, time, traceback
    from pathlib import Path

    def _p(msg: str):
        print(f"[FTCHK] {msg}", flush=True)

    # small helper to write debug files to GCS (best-effort; errors are logged but do not mask original exception)
    def _write_to_gcs(gcs_path: str, content: str, name: str):
        try:
            from google.cloud import storage
            if not gcs_path or not gcs_path.startswith("gs://"):
                _p(f"_gcs_write skipped, invalid gcs_report_path: {gcs_path}")
                return None
            _, rest = gcs_path.split("gs://", 1)
            bucket_name, *prefix_parts = rest.split("/", 1)
            prefix = prefix_parts[0].rstrip("/") if prefix_parts else ""
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob_path = f"{prefix}/{run_id}/{name}" if prefix else f"{run_id}/{name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content, content_type="text/plain")
            gpath = f"gs://{bucket_name}/{blob_path}"
            _p(f"Wrote GCS debug file: {gpath}")
            return gpath
        except Exception as e:
            _p(f"Failed to write debug file to GCS: {e}")
            return None

    # --- write connector env file (keeps SnowflakeConnector usage unchanged) ---
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

    # make repo importable
    for p in ("/app", "/app/src", os.getcwd(), str(Path(os.getcwd()).parent)):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # Log the key inputs (instrumentation)
    _p(f"features_table_fq: {features_table_fq}")
    _p(f"baseline_gcs_path: {baseline_gcs_path}")
    if gcs_report_path:
        # write a small started marker so you can confirm the component started and had GCS access
        try:
            _write_to_gcs(gcs_report_path, f"started {run_id} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", f"started_{run_id}.txt")
        except Exception as e:
            _p(f"Non-fatal: could not write started marker to GCS: {e}")

    # import SnowflakeConnector
    import importlib
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

    # --- load baseline from GCS ---
    baseline = {}
    try:
        from google.cloud import storage
        if not baseline_gcs_path or not baseline_gcs_path.startswith("gs://"):
            raise ValueError("baseline_gcs_path must be a gs:// path")
        _, rest = baseline_gcs_path.split("gs://", 1)
        bucket_name, *prefix_parts = rest.split("/", 1)
        prefix = prefix_parts[0] if prefix_parts else ""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(prefix)
        if not blob.exists():
            raise FileNotFoundError(f"Baseline not found at {baseline_gcs_path}")
        baseline = json.loads(blob.download_as_text())
        _p(f"Loaded baseline from {baseline_gcs_path}; features: {list(baseline.keys())}")
    except Exception as e:
        _p(f"Failed to load baseline from GCS: {e}")
        traceback.print_exc()
        # write error artifact to GCS if possible, then re-raise
        try:
            if gcs_report_path:
                _write_to_gcs(gcs_report_path, f"Failed to load baseline: {e}\n\n{traceback.format_exc()}", f"error_{run_id}.txt")
        except Exception:
            pass
        raise

    # parse overrides
    overrides = {}
    if per_feature_overrides:
        try:
            overrides = json.loads(per_feature_overrides)
        except Exception:
            _p("per_feature_overrides is not valid JSON; ignoring overrides")

    # connect to Snowflake
    session = None
    try:
        conn = SnowflakeConnector(app_env=app_env)
        session = conn.create_snowflake_session()
        _p("Snowpark session created")
    except Exception as e:
        _p(f"Could not create Snowflake session: {e}")
        traceback.print_exc()
        # write error artifact to GCS if available, then re-raise
        try:
            if gcs_report_path:
                _write_to_gcs(gcs_report_path, f"Snowflake session creation failed: {e}\n\n{traceback.format_exc()}", f"error_{run_id}.txt")
        except Exception:
            pass
        raise

    report = {
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp_utc,
        "features_table": features_table_fq,
        "status": "ok",
        "feature_checks": {},
        "meta": {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }

    try:
        # parse table fq
        parts = features_table_fq.split(".")
        if len(parts) != 3:
            raise ValueError("features_table_fq must be fully-qualified DB.SCHEMA.TABLE")
        db, schema, table = parts[0], parts[1], parts[2]

        # determine which baseline features exist in the table
        q_cols = f"""
            SELECT column_name
            FROM {db}.information_schema.columns
            WHERE table_schema = '{schema}' AND table_name = '{table}'
        """
        cols_meta = [r["COLUMN_NAME"] for r in session.sql(q_cols).collect()]
        baseline_feats = [f for f in baseline.keys() if f in cols_meta]
        missing_feats = [f for f in baseline.keys() if f not in cols_meta]
        for m in missing_feats:
            report["feature_checks"][m] = {"present": False, "reason": "missing_in_table"}
            # mark missing as fail by default
            report["status"] = "fail"
            report.setdefault("failed_features", []).append(m)
            _p(f"Baseline feature missing in table: {m}")

        if not baseline_feats:
            _p("No baseline features are present in the table; exiting with fail")
            # write report and return
            raise RuntimeError("No baseline features present in table")

        # Build a single aggregate SQL to compute p1 (approx), p99 (approx), avg, nulls
        select_parts = ["COUNT(*) AS total_rows"]
        for f in baseline_feats:
            select_parts.append(f"APPROX_PERCENTILE({f}, 0.01) AS {f}__p1")
            select_parts.append(f"APPROX_PERCENTILE({f}, 0.99) AS {f}__p99")
            select_parts.append(f"AVG({f}) AS {f}__mean")
            select_parts.append(f"SUM(CASE WHEN {f} IS NULL THEN 1 ELSE 0 END) AS {f}__nulls")
        select_sql = "SELECT\n  " + ",\n  ".join(select_parts) + f"\nFROM {db}.{schema}.{table}"
        _p("Running aggregate stats SQL (pushdown to Snowflake)...")
        agg_row = session.sql(select_sql).collect()[0]

        total_rows = int(agg_row["TOTAL_ROWS"])
        report["total_rows"] = total_rows

        overall_status = "ok"

        # --- Robust row lookup helper (handles Snowpark Row variations) ---
        def _row_get(row, col_name):
            # try direct indexing
            try:
                return row[col_name]
            except Exception:
                pass
            # try uppercase
            try:
                return row[col_name.upper()]
            except Exception:
                pass
            # try lowercase
            try:
                return row[col_name.lower()]
            except Exception:
                pass
            # try attribute access
            try:
                return getattr(row, col_name)
            except Exception:
                pass
            try:
                return getattr(row, col_name.upper())
            except Exception:
                pass
            return None

        for f in baseline_feats:
            # use robust getter (column aliases were created as f"{f}__p1" etc.)
            cur_p1 = _row_get(agg_row, f"{f}__p1")
            cur_p99 = _row_get(agg_row, f"{f}__p99")
            cur_mean = _row_get(agg_row, f"{f}__mean")
            cur_nulls = _row_get(agg_row, f"{f}__nulls")
            cur_nulls = int(cur_nulls) if cur_nulls is not None else 0
            cur_null_pct = (cur_nulls / total_rows) if total_rows > 0 else None

            b = baseline[f]
            # choose flex for this feature (override if provided)
            fflex = overrides.get(f, {}).get("flex_pct", overrides.get(f, {}).get("flex", flex_pct))

            # helper to compare a value against baseline multiplicatively (handles near-zero baseline)
            def _within_bounds(metric_name, baseline_val, cur_val, flex):
                note = {"baseline": baseline_val, "current": cur_val, "flex_pct": flex}
                if baseline_val is None or cur_val is None:
                    return {"result": "missing", "note": note}
                # if baseline is very small (close to 0), use additive tolerance = baseline_abs_tol
                if abs(baseline_val) < 1e-6:
                    # use a relative absolute tolerance (e.g., baseline + small absolute delta)
                    abs_tol = max(1.0, abs(baseline_val) * flex)
                    lower = baseline_val - abs_tol
                    upper = baseline_val + abs_tol
                else:
                    lower = baseline_val * (1.0 - flex)
                    upper = baseline_val * (1.0 + flex)
                note.update({"lower": lower, "upper": upper})
                if cur_val < lower or cur_val > upper:
                    return {"result": "out", "note": note}
                return {"result": "in", "note": note}

            # compare p1, p99, mean
            p1_cmp = _within_bounds("p1", b.get("p1"), cur_p1, fflex)
            p99_cmp = _within_bounds("p99", b.get("p99"), cur_p99, fflex)
            mean_cmp = _within_bounds("mean", b.get("mean"), cur_mean, fflex)

            # null pct comparison: baseline null pct (default 0 if absent)
            baseline_null = float(b.get("null_pct", 0.0))
            null_delta = (cur_null_pct - baseline_null) if (cur_null_pct is not None) else None
            null_result = "ok"
            if null_delta is not None:
                if null_delta > float(null_fail_delta):
                    null_result = "fail"
                elif null_delta > float(null_warn_delta):
                    null_result = "warn"

            # decide per-feature status
            feature_status = "ok"
            reasons = []
            if p1_cmp["result"] == "out":
                reasons.append({"metric": "p1", "detail": p1_cmp["note"]})
                feature_status = "warn"
            if p99_cmp["result"] == "out":
                reasons.append({"metric": "p99", "detail": p99_cmp["note"]})
                feature_status = "warn"
            if mean_cmp["result"] == "out":
                reasons.append({"metric": "mean", "detail": mean_cmp["note"]})
                feature_status = "warn"
            if null_result == "warn":
                reasons.append({"metric": "null_pct", "detail": {"baseline": baseline_null, "current": cur_null_pct, "delta": null_delta}, "level": "warn"})
                if feature_status != "fail":
                    feature_status = "warn"
            if null_result == "fail":
                reasons.append({"metric": "null_pct", "detail": {"baseline": baseline_null, "current": cur_null_pct, "delta": null_delta}, "level": "fail"})
                feature_status = "fail"

            # escalate: if any comparison outside bounds and deviation is extreme (e.g., > 2x flex) mark fail.
            # compute a crude severity factor for p99 deviation:
            def _severity(bv, cv):
                if bv is None or cv is None:
                    return 0.0
                if abs(bv) < 1e-6:
                    return abs(cv - bv)
                return abs(cv - bv) / (abs(bv) + 1e-9)

            if p99_cmp["result"] == "out":
                sev = _severity(b.get("p99"), cur_p99)
                if sev > 1.0:  # >100% relative change -> escalate to fail
                    feature_status = "fail"
                    reasons.append({"metric": "p99_severity", "value": sev, "explain": "relative change >100%, escalate to fail"})

            # update overall status
            if feature_status == "fail":
                overall_status = "fail"
            elif feature_status == "warn" and overall_status != "fail":
                overall_status = "warn"

            # store feature result
            report["feature_checks"][f] = {
                "present": True,
                "current": {"p1": cur_p1, "p99": cur_p99, "mean": cur_mean, "null_pct": cur_null_pct},
                "baseline": b,
                "flex_pct": fflex,
                "status": feature_status,
                "reasons": reasons
            }

        report["status"] = overall_status

        # for any failing features, collect up to sample_rows_for_fail sample rows to help debugging
        failing = [fname for fname, v in report["feature_checks"].items() if v.get("status") == "fail"]
        if failing and sample_rows_for_fail > 0:
            # sample up to N rows where feature value is extreme (p99 > baseline upper or p1 < baseline lower)
            samples = {}
            for f in failing:
                fb = report["feature_checks"][f]
                b = fb["baseline"]
                fflex = fb["flex_pct"]
                # build simple filter: value > upper OR value < lower (using p99 baseline as pivot)
                upper = b.get("p99") * (1 + fflex) if b.get("p99") is not None else None
                lower = b.get("p1") * (1 - fflex) if b.get("p1") is not None else None
                cond_parts = []
                if upper is not None:
                    cond_parts.append(f"{f} > {upper}")
                if lower is not None:
                    cond_parts.append(f"{f} < {lower}")
                cond = " OR ".join(cond_parts) if cond_parts else "1=0"
                q = f"SELECT * FROM {db}.{schema}.{table} WHERE {cond} LIMIT {sample_rows_for_fail}"
                try:
                    rows = [dict(r) for r in session.sql(q).collect()]
                except Exception:
                    rows = []
                samples[f] = rows
            report["samples"] = samples

        # write report to GCS
        report_blob_path = None
        if gcs_report_path and gcs_report_path.startswith("gs://"):
            _, rest = gcs_report_path.split("gs://", 1)
            bucket_name, *prefix_parts = rest.split("/", 1)
            prefix = prefix_parts[0] if prefix_parts else ""
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"{prefix.rstrip('/')}/{run_id}/feature_threshold_report_{run_id}.json" if prefix else f"{run_id}/feature_threshold_report_{run_id}.json"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(json.dumps(report, default=str), content_type="application/json")
            report_blob_path = f"gs://{bucket_name}/{blob_path}"
            _p(f"Wrote feature threshold report to {report_blob_path}")

        # close session
        try:
            session.close()
        except Exception:
            pass

        out = {"status": report["status"], "report_gcs_path": report_blob_path, "report": report}
        return json.dumps(out, default=str)

    except Exception as e:
        # Write traceback to GCS for easier debugging (non-intrusive)
        tb = traceback.format_exc()
        _p(f"Feature threshold check failed: {e}")
        _p(tb)
        try:
            if gcs_report_path:
                _write_to_gcs(gcs_report_path, tb, f"error_{run_id}.txt")
                # also write a tiny json summary so callers can quickly inspect
                try:
                    summary = {"error": str(e), "run_id": run_id, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    _write_to_gcs(gcs_report_path, json.dumps(summary), f"error_summary_{run_id}.json")
                except Exception:
                    pass
        except Exception:
            pass
        traceback.print_exc()
        try:
            if session:
                session.close()
        except Exception:
            pass
        # re-raise original exception to preserve existing behavior
        raise
