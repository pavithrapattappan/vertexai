# src/pipeline/components/shap_explain_component.py
import os
from kfp import dsl
import importlib
import inspect

IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE"
)

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "google-cloud-storage>=2.10.0",
        "google-cloud-secret-manager>=2.20.0",
        "snowflake-snowpark-python[pandas]>=1.15.0",
        "python-dotenv>=1.0.1",
        "pandas>=2.0.0",
        "numpy>=1.23.0",
        "shap>=0.41.0",
        "matplotlib>=3.6.0",
    ],
)
def shap_explain_component(
    # connector / env
    app_env: str,
    gcp_project_id: str,
    snowflake_account: str,
    snowflake_user: str,
    snowflake_database: str,
    snowflake_schema: str,
    snowflake_warehouse: str,
    snowflake_role: str,
    snowflake_password_secret_name: str,

    # input tables & bundle
    input_preprocessed_table: str,
    input_clusters_table: str,

    # pipeline-level output table (fallback if manifest doesn't provide per-segment table)
    output_table: str = "",
    model_bundle_gcs_uri: str = "",
    bundle_gcs_uri: str = "",
    # run metadata / controls
    run_id: str = "",
    run_timestamp_utc: str = "",
    # default to CONSUMER per your requirement; override at runtime if needed
    segment_name: str = "CONSUMER",
    max_rows_to_explain: int = 5000,
    sample_frac_for_background: float = 0.05,
    debug_gcs_bucket: str = "",
) -> str:
    """
    Component that runs the DS shap_explain class for segments, writes results to Snowflake,
    and optionally uploads plots + JSON to GCS for inspection.
    """

    # local imports for function scope
    import json, sys, traceback, base64, importlib as _importlib, inspect as _inspect
    from pathlib import Path
    from google.cloud import storage

    # ensure the project src path is available inside the container
    sys.path.insert(0, "/app")
    sys.path.insert(0, str(Path("/app/src")))

    def _p(msg: str):
        print(f"[SHAP-EXPLAIN] {msg}", flush=True)

    # -------------------------------------------------------------------
    # parse model bundle URI (prefer model_bundle_gcs_uri then bundle_gcs_uri)
    # -------------------------------------------------------------------
    uri = (model_bundle_gcs_uri or bundle_gcs_uri or "").rstrip("/")
    if not uri:
        raise ValueError("model_bundle_gcs_uri or bundle_gcs_uri must be provided")
    try:
        bucket_name, prefix = uri.replace("gs://", "").split("/", 1)
    except Exception:
        bucket_name = uri.replace("gs://", "")
        prefix = ""

    # -------------------------------------------------------------------
    # write env file used by Snowflake connector (some connectors rely on this)
    # -------------------------------------------------------------------
    env_path = f".env.{app_env}"
    try:
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
        _p(f"Env file written: {env_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to write env file {env_path}: {e}")

    # -------------------------------------------------------------------
    # Import Snowflake connector (project module paths), fallback included
    # -------------------------------------------------------------------
    def _import_connector():
        candidates = [
            ("src.data.connect", "SnowflakeConnector"),
            ("data.connect", "SnowflakeConnector"),
            ("connect", "SnowflakeConnector"),
        ]
        for mod, attr in candidates:
            try:
                m = _importlib.import_module(mod)
                _p(f"Imported SnowflakeConnector from {mod}.{attr}")
                return getattr(m, attr)
            except Exception:
                continue

        # Fallback Snowflake Connector that uses Secret Manager + Snowpark
        _p("Project SnowflakeConnector not found; using builtin fallback SnowflakeConnector")

        class FallbackSnowflakeConnector:
            def __init__(self, app_env=None):
                self.env = {}
                env_path_local = f".env.{app_env}" if app_env else None
                if env_path_local and os.path.exists(env_path_local):
                    try:
                        with open(env_path_local, "r") as fh:
                            for line in fh:
                                if "=" in line:
                                    k, v = line.strip().split("=", 1)
                                    self.env[k] = v
                        _p(f"Loaded env file {env_path_local}")
                    except Exception as e:
                        _p(f"Failed to read env file {env_path_local}: {e}")

            def _get(self, key, fallback=None):
                return self.env.get(key) or os.environ.get(key) or fallback

            def create_snowflake_session(self):
                try:
                    from google.cloud import secretmanager
                    from snowflake.snowpark import Session
                except Exception as e:
                    raise RuntimeError(f"Missing required packages for Snowpark/Secret Manager: {e}")

                account = self._get("SNOWFLAKE_ACCOUNT") or snowflake_account
                user = self._get("SNOWFLAKE_USER") or snowflake_user
                database = self._get("SNOWFLAKE_DATABASE") or snowflake_database
                schema = self._get("SNOWFLAKE_SCHEMA") or snowflake_schema
                role = self._get("SNOWFLAKE_ROLE") or snowflake_role
                warehouse = self._get("SNOWFLAKE_WAREHOUSE") or snowflake_warehouse
                secret_name = self._get("SNOWFLAKE_PASSWORD_SECRET_NAME") or snowflake_password_secret_name
                gcp_project = self._get("GCP_PROJECT_ID") or gcp_project_id

                if not (account and user and database and schema and secret_name and gcp_project):
                    raise RuntimeError("Missing Snowflake connection parameters: account,user,database,schema,secret and GCP project are required")

                # fetch password from Secret Manager
                try:
                    sm_client = secretmanager.SecretManagerServiceClient()
                    secret_path = f"projects/{gcp_project}/secrets/{secret_name}/versions/latest"
                    response = sm_client.access_secret_version(request={"name": secret_path})
                    password = response.payload.data.decode("utf-8")
                except Exception as e:
                    raise RuntimeError(f"Failed to access secret {secret_name} in project {gcp_project}: {e}")

                conn_props = {
                    "account": account,
                    "user": user,
                    "password": password,
                    "role": role,
                    "warehouse": warehouse,
                    "database": database,
                    "schema": schema,
                }

                _p(f"Creating Snowpark session to account={account}, user={user}, database={database}, schema={schema}, warehouse={warehouse}, role={role}")
                try:
                    session = Session.builder.configs(conn_props).create()
                    return session
                except Exception as e:
                    raise RuntimeError(f"Failed to create Snowpark Session: {e}")

        return FallbackSnowflakeConnector

    SnowflakeConnector = _import_connector()

    # -------------------------------------------------------------------
    # create Snowflake session (fallback connector will create it)
    # -------------------------------------------------------------------
    try:
        conn = SnowflakeConnector(app_env=app_env)
        session = conn.create_snowflake_session()
        _p("Snowflake session created successfully")
    except FileNotFoundError as e:
        _p(f"Env file missing during session creation, regenerating: {e}")
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
        conn = SnowflakeConnector(app_env=app_env)
        session = conn.create_snowflake_session()
        _p("Snowflake session created on retry")
    except Exception as e:
        _p(f"ERROR creating Snowflake session: {e}")
        raise

    # -------------------------------------------------------------------
    # load manifest.json from the model bundle in GCS
    # -------------------------------------------------------------------
    client = storage.Client(project=gcp_project_id)
    manifest_blob = client.bucket(bucket_name).blob(prefix + "/manifest.json")
    try:
        manifest = json.loads(manifest_blob.download_as_text())
    except Exception as e:
        raise RuntimeError(f"Failed to load manifest from {uri}/manifest.json: {e}")

    _p(f"Manifest loaded with {len(manifest.get('segments', {}))} segments")

    # -------------------------------------------------------------------
    # resolve segments to run (supports prefix)
    # -------------------------------------------------------------------
    def _resolve_segment_names(manifest, seg):
        seg_keys = list(manifest.get("segments", {}).keys())
        if not seg:
            return seg_keys
        seg_lower = seg.lower()
        return [k for k in seg_keys if k.lower().startswith(seg_lower)]

    segments_to_run = _resolve_segment_names(manifest, segment_name)
    _p(f"Resolved segments: {segments_to_run}")

    # -------------------------------------------------------------------
    # find DS class shap_explain (expects class with run_shap_explain)
    # -------------------------------------------------------------------
    def _find_shap_class():
        try:
            mod = _importlib.import_module("src.data.shap_explain")
        except Exception as e:
            _p(f"Failed to import src.data.shap_explain: {e}")
            raise

        members = [m for m in dir(mod) if not m.startswith("_")]
        _p(f"src.data.shap_explain members: {members}")

        # prefer top-level class named shap_explain
        if hasattr(mod, "shap_explain"):
            candidate = getattr(mod, "shap_explain")
            if _inspect.isclass(candidate):
                _p("Found shap_explain as a class at top-level")
                return candidate
            # search inside attribute if it is a container
            try:
                inner_members = [getattr(candidate, n) for n in dir(candidate) if not n.startswith("_")]
                for it in inner_members:
                    if _inspect.isclass(it) and "shap" in it.__name__.lower():
                        _p(f"Found inner class {it} inside shap_explain attribute")
                        return it
            except Exception:
                pass

        # fallback: any class that exposes run_shap_explain
        for name in members:
            try:
                attr = getattr(mod, name)
            except Exception:
                continue
            if _inspect.isclass(attr) and "run_shap_explain" in dir(attr):
                _p(f"Using class {name} with run_shap_explain")
                return attr

        raise ImportError("Could not locate shap_explain class with run_shap_explain in src.data.shap_explain")

    ShapClass = _find_shap_class()
    _p(f"Using Shap class: {ShapClass}")

    # -------------------------------------------------------------------
    # helper to upload PNGs and a JSON copy to GCS (if debug_gcs_bucket provided)
    # -------------------------------------------------------------------
    def _upload_plot_and_json(results_df, seg_key):
        if not debug_gcs_bucket:
            return []

        dest = debug_gcs_bucket.replace("gs://", "").rstrip("/")
        if "/" in dest:
            bname, prefix2 = dest.split("/", 1)
        else:
            bname, prefix2 = dest, ""

        uploaded_paths = []
        bucket = client.bucket(bname)

        for i, row in results_df.iterrows():
            seg = row.get("SEGMENT", seg_key)
            cluster = row.get("CLUSTER", i)
            plot_html = row.get("CLUSTER_SHAP_PLOT")
            if not plot_html:
                continue
            try:
                marker = "base64,"
                b64 = plot_html.split(marker, 1)[1].split('"', 1)[0]
                data = base64.b64decode(b64)
                object_path = f"{prefix2.rstrip('/')}/shap_plots/{run_id}/{seg}/{cluster}.png" if prefix2 else f"shap_plots/{run_id}/{seg}/{cluster}.png"
                blob = bucket.blob(object_path.lstrip("/"))
                blob.upload_from_string(data, content_type="image/png")
                uploaded_paths.append(f"gs://{bname}/{object_path}")
                _p(f"Uploaded plot for {seg}/{cluster} to gs://{bname}/{object_path}")
            except Exception as e:
                _p(f"Failed to extract/upload plot for {seg}/{cluster}: {e}")
                continue

        try:
            df_copy = results_df.copy()
            if "CLUSTER_SHAP_PLOT" in df_copy.columns:
                df_copy = df_copy.drop(columns=["CLUSTER_SHAP_PLOT"])
            json_str = df_copy.to_json(orient="records", date_format="iso")
            object_path = f"{prefix2.rstrip('/')}/shap_outputs/{run_id}/{seg_key}.json" if prefix2 else f"shap_outputs/{run_id}/{seg_key}.json"
            blob = bucket.blob(object_path.lstrip("/"))
            blob.upload_from_string(json_str, content_type="application/json")
            uploaded_paths.append(f"gs://{bname}/{object_path}")
            _p(f"Uploaded results JSON for {seg_key} to gs://{bname}/{object_path}")
        except Exception as e:
            _p(f"Failed to upload results JSON for {seg_key}: {e}")

        return uploaded_paths

    # -------------------------------------------------------------------
    # iterate segments, call DS class, save to Snowflake (DS code) + optionally upload to GCS
    # -------------------------------------------------------------------
    processed = []
    upload_report = {}
    for seg_key in segments_to_run:
        try:
            seg_info = manifest.get("segments", {}).get(seg_key)
            if seg_info is None:
                _p(f"No seg_info for {seg_key}, skipping")
                continue
            _p(f"seg_info for {seg_key}: {seg_info}")
        except Exception:
            _p(f"No seg_info for {seg_key}, skipping")
            continue

        # load artifact (either seg_info points to artifact blob, or seg_info is the artifact)
        try:
            if isinstance(seg_info, dict) and "artifact" in seg_info:
                artifact_name = seg_info["artifact"]
                blob = client.bucket(bucket_name).blob(prefix + "/" + artifact_name)
                seg_art = json.loads(blob.download_as_text())
            else:
                seg_art = seg_info
        except Exception as e:
            _p(f"Error loading artifact for {seg_key}: {e}")
            traceback.print_exc()
            continue

        _p(f"Segment artifact for {seg_key}: keys = {list(seg_art.keys()) if isinstance(seg_art, dict) else type(seg_art)}")

        try:
            # determine input table (manifest override if present)
            input_table = seg_art.get("input_table") if isinstance(seg_art, dict) else None
            input_table = input_table or (seg_art.get("data_table") if isinstance(seg_art, dict) else None)
            input_table = input_table or input_preprocessed_table

            # construct Snowpark DataFrame (best-effort)
            try:
                snowpark_df = session.table(input_table)
            except Exception:
                try:
                    snowpark_df = session.sql(f"SELECT * FROM {input_table} LIMIT {max_rows_to_explain}")
                except Exception as e:
                    _p(f"Failed to construct Snowpark DataFrame from {input_table}: {e}")
                    raise

            # determine x_cols and y_col (manifest overrides)
            x_cols = seg_art.get("x_cols") if isinstance(seg_art, dict) else None
            y_col = seg_art.get("y_col") if isinstance(seg_art, dict) else None

            if not x_cols:
                # infer column names best-effort and exclude SEGMENT/CLUSTER_LABEL
                try:
                    cols = snowpark_df.schema.fields
                    try:
                        colnames = [c.name for c in cols]
                    except Exception:
                        colnames = snowpark_df.limit(1).to_pandas().columns.tolist()
                except Exception:
                    try:
                        sample_df = session.table(input_preprocessed_table).limit(1).to_pandas()
                        colnames = sample_df.columns.tolist()
                    except Exception:
                        colnames = []
                x_cols = [c for c in colnames if c not in ("SEGMENT", "CLUSTER_LABEL")]

            if not y_col:
                y_col = ["CLUSTER_LABEL"]

            target_table = (seg_art.get("output_table") or seg_art.get("target_table") or output_table)

            _p(f"Calling ShapClass for {seg_key}: input_table={input_table}, target_table={target_table}, x_cols(len)={len(x_cols)}")

            # ensure target table exists with a reasonable schema (create if not)
            try:
                ddl = f"""
                CREATE TABLE IF NOT EXISTS {target_table} (
                  SEGMENT VARCHAR,
                  CLUSTER INTEGER,
                  CLUSTER_SHAP VARIANT,
                  CLUSTER_SHAP_PLOT VARCHAR,
                  RUNID VARCHAR,
                  RUNTIMESTAMP VARCHAR
                )
                """
                # run DDL
                session.sql(ddl).collect()
                _p(f"Ensured Snowflake table exists: {target_table}")
            except Exception as e:
                _p(f"Warning: could not ensure/create table {target_table}: {e}")

            # instantiate and run DS class (this should write to Snowflake)
            try:
                inst = ShapClass(session=session, data=snowpark_df, x_cols=x_cols, y_col=y_col,
                                 runid=run_id, runtimestamp=run_timestamp_utc)
            except Exception as e:
                _p(f"Failed to instantiate ShapClass: {e}")
                traceback.print_exc()
                raise

            try:
                inst.run_shap_explain(target_table)
                _p(f"ShapClass.run_shap_explain completed for {seg_key}; Snowflake table: {target_table}")
            except Exception as e:
                _p(f"Error during inst.run_shap_explain for {seg_key}: {e}")
                traceback.print_exc()
                continue

            # --- diagnostics: inspect in-memory results and count back from Snowflake
            try:
                results_df = getattr(inst, "results_df", None)
                if results_df is None:
                    _p(f"[DIAG] inst.results_df is None for {seg_key}")
                    uploaded = []
                else:
                    try:
                        nrows = len(results_df)
                    except Exception:
                        nrows = None
                    _p(f"[DIAG] inst.results_df rows: {nrows}; columns: {list(results_df.columns)}")
                    try:
                        _p(f"[DIAG] sample rows: {results_df.head(3).to_dict(orient='records')}")
                    except Exception:
                        _p("[DIAG] failed to show sample rows")
                    # upload plots + json copy to GCS if requested
                    uploaded = _upload_plot_and_json(results_df, seg_key)
                upload_report[seg_key] = uploaded
            except Exception as e:
                _p(f"Failed to upload/results diagnostics for {seg_key}: {e}")
                traceback.print_exc()
                upload_report[seg_key] = []

            # check Snowflake table row count (best-effort)
            try:
                check_df = session.sql(f"SELECT COUNT(*) AS CNT FROM {target_table}").to_pandas()
                cnt = int(check_df["CNT"].iloc[0]) if not check_df.empty else -1
                _p(f"[DIAG] Snowflake table {target_table} COUNT = {cnt}")
            except Exception as e:
                _p(f"[DIAG] Could not query back Snowflake table {target_table}: {e}")

            processed.append(seg_key)

        except Exception as e:
            _p(f"Error processing segment {seg_key}: {e}")
            traceback.print_exc()
            continue

    # -------------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------------
    try:
        session.close()
        _p("Snowflake session closed")
    except Exception:
        _p("Warning: error closing Snowflake session (ignored)")

    summary = {"processed_segments": processed, "uploads": upload_report}
    return json.dumps(summary)
