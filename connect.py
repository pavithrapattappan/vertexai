# Import necessary functions
import os
from dotenv import load_dotenv
from google.cloud.secretmanager import SecretManagerServiceClient
from google.api_core.exceptions import NotFound, PermissionDenied, GoogleAPICallError
from snowflake.snowpark import Session

class SnowflakeConnector:
    def __init__(self, app_env:str = "np"):
        self.load_env(app_env)
        
    def load_env(self, app_env: str):
        env = app_env.lower()
        env_file = f".env.{env}"
        if not os.path.exists(env_file):
            raise FileNotFoundError(f"? Environment file '{env_file}' not found")
        # Load .env.np file only once
        load_dotenv(dotenv_path=env_file)
        print(f"Loaded config from {env_file}")

        # Snowflake credentials
        self.GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'prj-hds-np-data')
        self.SNOWFLAKE_PASSWORD_SECRET_NAME = os.environ.get('SNOWFLAKE_PASSWORD_SECRET_NAME')
        self.SNOWFLAKE_ACCOUNT = os.environ.get('SNOWFLAKE_ACCOUNT')
        self.SNOWFLAKE_USER = os.environ.get('SNOWFLAKE_USER')
        self.SNOWFLAKE_DATABASE = os.environ.get('SNOWFLAKE_DATABASE')
        self.SNOWFLAKE_SCHEMA = os.environ.get('SNOWFLAKE_SCHEMA')
        self.SNOWFLAKE_ROLE = os.environ.get('SNOWFLAKE_ROLE', '')
        self.SNOWFLAKE_WAREHOUSE = os.environ.get('SNOWFLAKE_WAREHOUSE', '')

    def get_secret_value(self, secret_id: str, version_id: str = "latest") -> str:
        """
        Retrieves the string value of a secret from GCP Secret Manager.

        Args:
            project_id (str): The GCP project ID.
            secret_id (str): The name of the secret.
            version_id (str): The secret version (e.g., "latest").

        Returns:
            str: The secret payload as a string.

        Raises:
            Exception: If the secret cannot be accessed or found.
        """
        try:
            client = SecretManagerServiceClient()
            name = f"projects/{self.GCP_PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except NotFound:
            raise Exception(f"Secret '{secret_id}' or version '{version_id}' not found in project '{self.GCP_PROJECT_ID}'.")
        except PermissionDenied:
            raise Exception(f"Permission denied to access secret '{secret_id}' in project '{self.GCP_PROJECT_ID}'. Ensure the service account has 'Secret Manager Secret Accessor' role.")
        except GoogleAPICallError as e:
            raise Exception(f"Google API error accessing secret '{secret_id}': {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred accessing secret '{secret_id}': {e}")


    def create_snowflake_session(self) -> Session:
        """Create and return Snowflake session."""
        snowflake_password = self.get_secret_value(self.SNOWFLAKE_PASSWORD_SECRET_NAME)
        connection_parameters = {
            "account": self.SNOWFLAKE_ACCOUNT,
            "user": self.SNOWFLAKE_USER,
            "role": self.SNOWFLAKE_ROLE,  # optional
            "warehouse": self.SNOWFLAKE_WAREHOUSE,
            "password": snowflake_password,
            "database": self.SNOWFLAKE_DATABASE,
            "schema": self.SNOWFLAKE_SCHEMA
        }
        return Session.builder.configs(connection_parameters).create()