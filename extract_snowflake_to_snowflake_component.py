# src/pipeline/components/extract_snowflake_to_snowflake_component.py
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
        "pandas>=2.2.2",
    ],
)
def extract_snowflake_to_snowflake(
    # connector inputs (passed through to .env.<app_env>)
    app_env: str,
    gcp_project_id: str,
    snowflake_account: str,
    snowflake_user: str,
    snowflake_database: str,
    snowflake_schema: str,
    snowflake_role: str,
    snowflake_warehouse: str,
    snowflake_password_secret_name: str,
    # output table names (in Snowflake)
    output_database: str,   # e.g. "EDP_ML_DEV" (or same as snowflake_database)
    output_schema: str,     # e.g. "SOW"
    history_table: str,     # e.g. "SOW_HISTORY_PIPELINETEST"
    features_table: str,    # e.g. "SOW_FEATURES_PIPELINETEST"
    # run metadata
    run_id: str = "",
    run_timestamp_utc: str = "",
    # caps / optional behavior
    row_limit: int = 0,
    start_date: str = "",
    end_date: str = "",
) -> str:
    """
    Wrapper that runs the DS extraction inside Snowflake and saves outputs back to Snowflake tables.
    Returns a status string listing tables written.
    """
    
    import os
    import sys, importlib, traceback, time
    from pathlib import Path

    def _p(msg: str):
        print(f"[EXTRACT-SF] {msg}", flush=True)

    # 1) write env file for DS connector (keep DS login logic unchanged)
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
    _p(f"Wrote connector env: {env_path}")

    # 2) ensure repo path is importable inside the container (image must include /app/src)
    def _add_paths():
        for p in ("/app", "/app/src", os.getcwd(), str(Path(os.getcwd()).parent)):
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        maybe_src = Path(os.getcwd()) / "src"
        if maybe_src.is_dir():
            parent = str(maybe_src.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
    _add_paths()
    _p(f"sys.path head: {sys.path[:6]}")

    # 3) tolerant import helpers
    def _import_attr(cands):
        for mod, attr in cands:
            try:
                m = importlib.import_module(mod)
                return getattr(m, attr)
            except Exception:
                continue
        raise ImportError(f"Could not import any of: {cands}; sys.path[:6]={sys.path[:6]}")

    # import SnowflakeConnector (do NOT change DS login)
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

    # import extractor (create_data / CreateData) from common locations
    try:
        create_data_cls_or_fn = _import_attr([
            ("src.data.extraction", "create_data"),
            ("data.extraction", "create_data"),
            ("extraction", "create_data"),
            ("src.data.create_data", "CreateData"),
            ("data.create_data", "CreateData"),
            ("create_data", "CreateData"),
        ])
    except Exception as e:
        _p(f"Import create_data/CreateData failed: {e}")
        traceback.print_exc()
        raise

    session = None
    try:
        # 4) create Snowpark session using DS connector
        conn = SnowflakeConnector(app_env=app_env)
        session = conn.create_snowflake_session()
        _p("Snowpark session created")

        # tolerant constructor for create_data
        def _new_cd(sess, rid, rts):
            cd = create_data_cls_or_fn
            try: return cd(session=sess, runid=rid, runtimestamp=rts)
            except TypeError: pass
            try: return cd(session=sess, run_id=rid, run_timestamp=rts)
            except TypeError: pass
            try: return cd(sess, rid, rts)
            except TypeError: pass
            try: return cd(session=sess)
            except TypeError as e:
                raise TypeError("Cannot construct CreateData/create_data; tried multiple signatures. " + str(e))

        job = _new_cd(session, run_id, run_timestamp_utc)
        _p("Created extractor job object")

        # run common extraction stage methods if present (safe)
        for m in [
            "get_carn","partition","get_sales","get_attributions",
            "carn_sales_join","exclude_customers","get_agreement","build_final",
            "build_combined_tam","get_master_tam","clean_values","get_sale_date",
            "get_tiers","enrich_data","add_stamps",
        ]:
            if hasattr(job, m):
                try:
                    getattr(job, m)()
                    _p(f"Ran method: {m}")
                except Exception as e:
                    _p(f"Method {m} failed: {e}; continuing")

        # feature steps
        def _safe_call(name):
            if hasattr(job, name):
                try:
                    getattr(job, name)()
                    _p(f"Ran feature step: {name}")
                    return True
                except Exception as e:
                    _p(f"Feature step {name} failed: {e}")
                    return False
            return False

        _safe_call("compute_behavioral_metrics")
        _safe_call("compute_customer_age")
        _safe_call("consolidate_tam_values")
        _safe_call("derive_most_recent_channel")
        _safe_call("format_columns")

        # locate Snowpark DataFrames from extractor (best-effort)
        history_src = getattr(job, "stamp_df", None) or getattr(job, "final_df", None) or getattr(job, "enriched_df", None)
        features_src = getattr(job, "consolidate_df", None) or getattr(job, "features_df", None) or getattr(job, "enriched_df", None)

        # optional date window (if Snowpark DF)
        if (start_date or end_date) and history_src is not None:
            try:
                from snowflake.snowpark.functions import to_date, lit, col as sf_col
                def _date_window(df_like):
                    if not df_like: return df_like
                    conds = []
                    if start_date: conds.append(sf_col("ORDER_DATE") >= to_date(lit(start_date)))
                    if end_date:   conds.append(sf_col("ORDER_DATE") <  to_date(lit(end_date)))
                    return df_like.filter(conds[0] if len(conds)==1 else (conds[0] & conds[1]))
                history_src  = _date_window(history_src)
                features_src = _date_window(features_src)
                _p(f"Applied date window: {start_date or 'None'} .. {end_date or 'None'}")
            except Exception as e:
                _p(f"Date window not applied: {e}")

        # cap rows if requested
        def _cap(df_like):
            if not row_limit or row_limit <= 0: return df_like
            return df_like.limit(int(row_limit)) if hasattr(df_like, "limit") else df_like
        history_src, features_src = _cap(history_src), _cap(features_src)
        if row_limit and row_limit > 0: _p(f"Row cap applied: {row_limit}")

        # ---------- write Snowpark DF -> Snowflake table ----------
        written = []
        def _save_snowpark_df(df, fq_table: str):
            """
            Try df.write.save_as_table; fallback to create_or_replace_table via session.register_or_replace?
            fq_table expected as 'DATABASE.SCHEMA.TABLE'
            """
            if df is None:
                _p(f"Skipping write: {fq_table} (no dataframe)")
                return False
            try:
                # prefer fully-qualified SAVE (may depend on Snowpark version)
                try:
                    df.write.save_as_table(fq_table, mode="overwrite")
                    _p(f"Wrote (save_as_table) -> {fq_table}")
                    return True
                except Exception as e:
                    _p(f"save_as_table failed ({e}), trying create_or_replace_table fallback")
                # fallback: create or replace table by using df.create_or_replace_temp_view + session.sql
                tmp_view = f"TEMP_VIEW_{int(time.time())}"
                df.create_or_replace_temp_view(tmp_view)
                session.sql(f"CREATE OR REPLACE TABLE {fq_table} AS SELECT * FROM {tmp_view}").collect()
                _p(f"Wrote (CREATE OR REPLACE TABLE AS SELECT) -> {fq_table}")
                return True
            except Exception as e:
                _p(f"Failed writing dataframe to {fq_table}: {e}")
                traceback.print_exc()
                return False

        hist_fq = f"{output_database}.{output_schema}.{history_table}"
        feat_fq = f"{output_database}.{output_schema}.{features_table}"

        ok_hist = _save_snowpark_df(history_src, hist_fq)
        ok_feat = _save_snowpark_df(features_src, feat_fq)
        if ok_hist: written.append(hist_fq)
        else: written.append(f"{hist_fq} (skipped/no rows/failed)")
        if ok_feat: written.append(feat_fq)
        else: written.append(f"{feat_fq} (skipped/no rows/failed)")

        _p("Extraction finished; returning status")
        return "Wrote to Snowflake: " + ", ".join(written)

    except Exception:
        _p("FATAL ERROR — full traceback follows")
        traceback.print_exc()
        raise

    finally:
        if session is not None:
            try:
                session.close(); _p("Snowpark session closed")
            except Exception:
                pass
