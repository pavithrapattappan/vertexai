# src/pipeline/components/preprocess_snowflake_component.py
import os
from kfp import dsl

IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE"
)

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "snowflake-snowpark-python[pandas]>=1.15.0",
        "python-dotenv>=1.0.1",
        "pandas>=2.2.2",
        "pyarrow>=14.0.0",
        "db-dtypes>=1.2.0",
    ],
)
def preprocess_snowflake_component(
    app_env: str,
    gcp_project_id: str,
    snowflake_account: str,
    snowflake_user: str,
    snowflake_database: str,
    snowflake_schema: str,
    snowflake_role: str,
    snowflake_warehouse: str,
    snowflake_password_secret_name: str,

    input_database: str,
    input_schema: str,
    input_table: str,

    preprocessed_table: str,
    segments_table: str,            # kept for API compatibility though DS writes preprocessed_table

    # If you pass the run id or any suffix here, we will append it only if it's NOT already present.
    pipeline_suffix: str = "",

    min_segment_size: int = 60,
    use_msa_in_primary_segment: bool = True,

    run_id: str = "",
    run_timestamp_utc: str = "",
) -> str:
    """
    Minimal wrapper that runs your src/data/preprocess.py::preprocess_data class.
    It:
      - writes .env.<app_env> for your connector,
      - creates a Snowpark session via SnowflakeConnector from src.data.connect,
      - loads input table as a Snowpark DataFrame,
      - instantiates preprocess_data(input_df),
      - calls create_segment_freq(<preprocessed_table_fqn>) so DS code writes the table.
    Returns a short summary string.
    """
    # runtime imports
    import os
    import sys, importlib, importlib.util, traceback, re
    from pathlib import Path

    def _p(msg: str):
        print(f"[PREPROCESS] {msg}", flush=True)

    # ---------------------------
    # Helpers for table name fix
    # ---------------------------
    def _sanitize_suffix(s: str) -> str:
        """Make a safe suffix like '_ABC_123' (or '' if empty)."""
        if not s:
            return ""
        s2 = re.sub(r"[^A-Za-z0-9_]+", "_", s)
        s2 = s2[:48].rstrip("_")
        return f"_{s2}" if s2 and not s2.startswith("_") else s2

    def _append_suffix_once(base: str, suffix: str) -> str:
        """
        Append sanitized suffix to base only if it's not already present.
        Also collapse accidental duplicate trailing run-chunks like:
          _CONN_EXTRACT_RUN_YYYYMMDD_HHMMSS_CONNN_EXTRACT_RUN_YYYYMMDD_HHMMSS
        """
        if not suffix:
            name = base
        else:
            if base.endswith(suffix):
                name = base
            else:
                name = base + suffix

        # Collapse duplicated trailing RUN chunks (common ones we use)
        patterns = [
            r"(.*?)(CONN_EXTRACT_RUN_\d{8}_\d{6})_(?:CONN_EXTRACT_RUN_\d{8}_\d{6})$",
            r"(.*?)(PREPROCESS_RUN_\d{8}_\d{6})_(?:PREPROCESS_RUN_\d{8}_\d{6})$",
            r"(.*?)(EXTRACT_RUN_\d{8}_\d{6})_(?:EXTRACT_RUN_\d{8}_\d{6})$",
            r"(.*?)(RUN_\d{8}_\d{6})_(?:RUN_\d{8}_\d{6})$",
            # generic: duplicated timestamp at end "..._YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS"
            r"(.*?)(_?\d{8}_\d{6})_(?:\d{8}_\d{6})$",
        ]
        for pat in patterns:
            name = re.sub(pat, r"\1\2", name, flags=re.IGNORECASE)

        return name

    # 1) write connector env file so DS connect uses same credentials
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

    # 2) ensure /app and /app/src are on sys.path to allow imports
    for p in ("/app", "/app/src", os.getcwd(), str(Path(os.getcwd()).parent)):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    _p(f"sys.path head: {sys.path[:6]}")

    # 3) robust import SnowflakeConnector (try common module locations)
    def _import_symbol(candidates):
        for mod, name in candidates:
            try:
                m = importlib.import_module(mod)
                if hasattr(m, name):
                    return getattr(m, name)
            except Exception:
                continue
        # fallback: try import by file path if present
        paths = [
            "/app/src/data/connect.py",
            "/app/data/connect.py",
            "/app/src/connect.py",
            "/app/connect.py",
        ]
        for p in paths:
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location("tmp_connect_mod", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "SnowflakeConnector"):
                    return getattr(mod, "SnowflakeConnector")
        raise ImportError(f"Could not import SnowflakeConnector from {candidates} or fallback paths")

    try:
        SnowflakeConnector = _import_symbol([
            ("src.data.connect", "SnowflakeConnector"),
            ("data.connect", "SnowflakeConnector"),
            ("connect", "SnowflakeConnector"),
        ])
    except Exception as e:
        _p(f"Failed to import SnowflakeConnector: {e}")
        traceback.print_exc()
        raise

    # 4) import your preprocess_data class specifically
    def _import_preprocess_class():
        # try the canonical locations where your file exists
        try_cands = [
            ("src.data.preprocess", "preprocess_data"),
            ("data.preprocess", "preprocess_data"),
            ("src.model.preprocess", "preprocess_data"),
            ("model.preprocess", "preprocess_data"),
            ("src.data.preprocess", "preprocess"),
            ("data.preprocess", "preprocess"),
        ]
        for mod, name in try_cands:
            try:
                m = importlib.import_module(mod)
                if hasattr(m, name):
                    return getattr(m, name)
            except Exception:
                continue
        # fallback load by filename
        fallback_paths = [
            "/app/src/data/preprocess.py",
            "/app/src/model/preprocess.py",
            "/app/data/preprocess.py",
            "/app/model/preprocess.py",
        ]
        for p in fallback_paths:
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location("tmp_preproc_mod", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for name in ("preprocess_data", "preprocess", "Preprocess"):
                    if hasattr(mod, name):
                        return getattr(mod, name)
        raise ImportError("Could not import preprocess class/function from known locations")

    try:
        PreprocessClass = _import_preprocess_class()
    except Exception as e:
        _p(f"Failed to import preprocess implementation: {e}")
        traceback.print_exc()
        raise

    # Build target FQNs with safe, de-duplicated suffix handling
    safe_suffix = _sanitize_suffix(pipeline_suffix)
    pre_name = _append_suffix_once(preprocessed_table, safe_suffix)
    seg_name = _append_suffix_once(segments_table, safe_suffix)

    input_fq = f"{input_database}.{input_schema}.{input_table}"
    pre_fq = f"{snowflake_database}.{snowflake_schema}.{pre_name}"
    seg_fq = f"{snowflake_database}.{snowflake_schema}.{seg_name}"

    _p(f"Input FQ: {input_fq}")
    _p(f"Target preprocessed table: {pre_fq}")
    _p(f"Target segments table: {seg_fq}")

    session = None
    try:
        # create Snowpark session using your connector (this uses the .env file we wrote)
        conn = SnowflakeConnector(app_env=app_env)
        session = conn.create_snowflake_session()
        _p("Snowpark session created")

        # load input table as Snowpark DataFrame
        try:
            input_df = session.table(input_fq)
            _p("Loaded input table into Snowpark DataFrame")
        except Exception as e:
            _p(f"Failed to read input table {input_fq}: {e}")
            traceback.print_exc()
            raise

        # instantiate DS class with the Snowpark DataFrame
        try:
            preprocessor = PreprocessClass(input_df)
            _p(f"Instantiated preprocessor class: {type(preprocessor).__name__}")
        except TypeError:
            # try alternate constructors just in case
            try:
                preprocessor = PreprocessClass(session, input_df)
                _p("Instantiated preprocessor(session,input_df)")
            except Exception as e:
                _p(f"Failed to instantiate preprocess class: {e}")
                traceback.print_exc()
                raise

        # call the writer method your class provides: create_segment_freq(table)
        written = []
        try:
            if hasattr(preprocessor, "create_segment_freq"):
                _p(f"Calling preprocessor.create_segment_freq({pre_fq})")
                preprocessor.create_segment_freq(pre_fq)
                written.append(pre_fq)
                _p(f"create_segment_freq wrote {pre_fq}")
            elif hasattr(preprocessor, "write_prep_data"):
                _p(f"Calling preprocessor.write_prep_data({pre_fq})")
                preprocessor.write_prep_data(pre_fq)
                written.append(pre_fq)
                _p(f"write_prep_data wrote {pre_fq}")
            else:
                # if no writer method, try to see if preprocessor exposed a DF attribute to write
                for cand in ("data_segs_cnt", "data_segs", "data_seg", "processed_df", "preprocessed_df"):
                    if hasattr(preprocessor, cand):
                        df = getattr(preprocessor, cand)
                        if df is not None and hasattr(df, "write"):
                            _p(f"Found attribute {cand}; writing it to {pre_fq}")
                            df.write.save_as_table(pre_fq, mode="overwrite")
                            written.append(pre_fq)
                            _p(f"WROTE attribute {cand} -> {pre_fq}")
                            break

        except Exception as e:
            _p(f"DS writer call failed: {e}")
            traceback.print_exc()
            raise

        summary = ", ".join([f"{t}=OK" for t in written]) if written else "No tables written by DS preprocess (check logs)"
        _p("Preprocess component finished: " + summary)
        return summary

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
