import os
from kfp import dsl

# Pick up the pipeline image (must contain /app/src)
IMAGE_URI = os.getenv(
    "SOW_PIPELINE_IMAGE",
    "us-central1-docker.pkg.dev/prj-hds-np-data/ml-pipeline-images/sow-py:latest",
)


@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "google-cloud-secret-manager>=2.20.0",
        "snowflake-snowpark-python[pandas]>=1.15.0",
        "python-dotenv>=1.0.1",
    ],
)
def connection_check_component(
    app_env: str,
    gcp_project_id: str,
    snowflake_account: str,
    snowflake_user: str,
    snowflake_database: str,
    snowflake_schema: str,
    snowflake_role: str,
    snowflake_warehouse: str,
    snowflake_password_secret_name: str,
) -> str:
    """
    Pipeline wrapper for DS connection logic.
    1. Writes .env.<env> so connect.py can read it.
    2. Imports SnowflakeConnector from src/data/connect.py.
    3. Creates Snowflake session and runs a smoke query.
    """

    import sys
    from pathlib import Path

    # --- Make sure /app/src is on PYTHONPATH ---
    sys.path.insert(0, "/app")
    sys.path.insert(0, str(Path("/app/src")))

    # --- Write connector env file ---
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
    print(f"[CONNECT] Wrote env file: {env_path}", flush=True)

    # --- Import DS connector ---
    from src.data.connect import SnowflakeConnector

    session = None
    try:
        conn = SnowflakeConnector(app_env=app_env)
        session = conn.create_snowflake_session()
        print("[CONNECT] Session created, running smoke query...", flush=True)

        res = session.sql("select current_version() as ver").collect()
        version = res[0]["VER"] if res else "unknown"

        print(f"[CONNECT] Snowflake OK — version: {version}", flush=True)
        return f"Snowflake connection OK — version {version}"
    finally:
        if session is not None:
            session.close()
            print("[CONNECT] Session closed", flush=True)
