import os
from kfp import dsl

IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE",
    "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:latest",
)

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "snowflake-snowpark-python[pandas]>=1.15.0",
        "python-dotenv>=1.0.1",
        "google-cloud-storage>=2.8.0",
    ],
)
def data_quality_check_minimal_component(
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

    # target table (FQN)
    features_table_fq: str,

    # basic thresholds
    min_rows: int = 100,

    # run meta + reporting
    run_id: str = "",
    run_timestamp_utc: str = "",
    gcs_report_path: str = "",
) -> str:
    """
    Minimal DQ:
      - Write .env.<env>
      - Connect to Snowflake via your src.data.connect.SnowflakeConnector
      - Validate FQN parse
      - COUNT(*) and simple null checks on a few key columns (if present)
      - Write 'started', 'report', and 'error' files to GCS
    Returns "ok" | "fail"
    """
    import os
    import json, time, traceback, importlib, sys
    from pathlib import Path

    def _p(msg: str):
        print(f"[DQ-MIN] {msg}", flush=True)

    def _gcs_write(name: str, content: str, content_type: str = "text/plain"):
        try:
            if not gcs_report_path or not gcs_report_path.startswith("gs://"):
                _p(f"Skip GCS write (invalid path): {gcs_report_path}")
                return None
            from google.cloud import storage
            _, rest = gcs_report_path.split("gs://", 1)
            bucket_name, *prefix_parts = rest.split("/", 1)
            prefix = (prefix_parts[0] if prefix_parts else "").rstrip("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob_path = f"{prefix}/{run_id}/{name}" if prefix else f"{run_id}/{name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content, content_type=content_type)
            gpath = f"gs://{bucket_name}/{blob_path}"
            _p(f"Wrote GCS: {gpath}")
            return gpath
        except Exception as e:
            _p(f"GCS write failed: {e}")
            return None

    # 1) Write .env so your connector picks the same vars
    try:
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
        _gcs_write(f"started_{run_id}.txt", f"started {run_id} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    except Exception as e:
        tb = traceback.format_exc()
        _p(f"Failed writing env: {e}\n{tb}")
        _gcs_write(f"error_env_{run_id}.txt", f"{e}\n\n{tb}")
        raise

    # 2) Make repo importable & import connector robustly
    for p in ("/app", "/app/src", os.getcwd(), str(Path(os.getcwd()).parent)):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    def _import_connector():
        try:
            m = importlib.import_module("src.data.connect")
            return getattr(m, "SnowflakeConnector")
        except Exception:
            pass
        try:
            m = importlib.import_module("data.connect")
            return getattr(m, "SnowflakeConnector")
        except Exception:
            pass
        # fallback by path
        for cand in ("/app/src/data/connect.py", "/app/data/connect.py", "/app/connect.py", "connect.py"):
            if Path(cand).exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("tmp_connect", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "SnowflakeConnector"):
                    return getattr(mod, "SnowflakeConnector")
        raise ImportError("SnowflakeConnector not found")

    try:
        SnowflakeConnector = _import_connector()
        _p("Imported SnowflakeConnector")
    except Exception as e:
        tb = traceback.format_exc()
        _p(f"Connector import failed: {e}\n{tb}")
        _gcs_write(f"error_import_{run_id}.txt", f"{e}\n\n{tb}")
        raise

    # 3) Connect
    session = None
    try:
        conn = SnowflakeConnector(app_env=app_env)
        session = conn.create_snowflake_session()
        _p("Snowpark session created")
    except Exception as e:
        tb = traceback.format_exc()
        _p(f"Snowflake session failed: {e}\n{tb}")
        _gcs_write(f"error_session_{run_id}.txt", f"{e}\n\n{tb}")
        raise

    report = {
        "run_id": run_id,
        "table": features_table_fq,
        "status": "ok",
        "meta": {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "checks": {},
    }

    try:
        # 4) Parse FQN
        parts = features_table_fq.split(".")
        if len(parts) != 3:
            raise ValueError("features_table_fq must be DB.SCHEMA.TABLE")
        db, schema, table = parts[0], parts[1], parts[2]
        report["checks"]["parsed"] = {"db": db, "schema": schema, "table": table}

        # 5) COUNT(*)
        cnt_row = session.sql(f"SELECT COUNT(*) AS TOTAL_ROWS FROM {db}.{schema}.{table}").collect()[0]
        total_rows = int(cnt_row["TOTAL_ROWS"] if "TOTAL_ROWS" in cnt_row else list(cnt_row)[0])
        report["checks"]["total_rows"] = total_rows
        if total_rows < int(min_rows):
            report["status"] = "fail"
            report["checks"]["reason"] = f"total_rows {total_rows} < min_rows {min_rows}"

        # 6) Minimal column presence + null probe (best-effort)
        col_rows = session.sql(
            f"SELECT column_name FROM {db}.information_schema.columns "
            f"WHERE table_schema='{schema}' AND table_name='{table}'"
        ).collect()
        cols = [r["COLUMN_NAME"] for r in col_rows] if col_rows else []
        report["checks"]["columns_head"] = cols[:25]

        for probe in ("CUSTOMER_ID", "ORDER_NUMBER"):
            if probe in cols:
                r = session.sql(
                    f"SELECT SUM(CASE WHEN {probe} IS NULL THEN 1 ELSE 0 END) AS NULLS "
                    f"FROM {db}.{schema}.{table}"
                ).collect()[0]
                nnull = int(r["NULLS"] if "NULLS" in r else list(r)[0])
                report["checks"][f"{probe}_nulls"] = nnull

        # 7) Write report
        _gcs_write(f"dq_minimal_report_{run_id}.json", json.dumps(report, default=str), "application/json")

        try:
            session.close()
        except Exception:
            pass
        return report["status"]

    except Exception as e:
        tb = traceback.format_exc()
        _p(f"DQ failed: {e}\n{tb}")
        _gcs_write(f"dq_minimal_error_{run_id}.txt", f"{e}\n\n{tb}")
        try:
            if session:
                session.close()
        except Exception:
            pass
        raise
