import os
from typing import Optional, Union

from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

from .configurations import logger_from_file
from .exception_handeling import extract_error_from_exceptions

load_dotenv()


class CloudStorage(storage.Client):
    """
    A custom class for interacting with Google Cloud Storage, enhanced with logging capabilities.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket to interact with.
        logger_config (str, optional): Path to the logger configuration file. Default is 'configs/logger.yaml'.
        service_key (str, optional): Path to the service account key file. Default is 'configs/agent-dnd-b7fc8efe8bdd.json'.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = os.environ["BUCKET"],
        logger_config: Optional[str] = "configs/logger.yaml",
        service_key: Optional[str] = "configs/agent-dnd-b7fc8efe8bdd.json",
    ):
        """
        Initialize the Google Cloud Storage client with Compute Engine credentials.
        """
        credentials = service_account.Credentials.from_service_account_file(service_key)
        super().__init__(credentials=credentials)
        self.logger = logger_from_file(logger_config)
        self.bucket_name = bucket_name
        self._bucket_exists()

    def _bucket_exists(self):
        """ """
        bucket = self.bucket(self.bucket_name)
        if not bucket.exists():
            raise Exception(f"GCS {bucket} does not exist.")

    def upload_to_bucket(self, content: Union[bytes, str], filename: str):
        """
        Uploads a file to the specified bucket in Google Cloud Storage.

        Args:
            content (str): The content to upload as a string.
            filename (str): The name of the file in the bucket.
        """
        try:
            bucket = self.get_bucket(self.bucket_name)
            blob = bucket.blob(filename)
            blob.upload_from_string(content)
            self.logger.info(f"{filename} successfully uploaded to {self.bucket_name}.")

        except Exception as e:
            exception = extract_error_from_exceptions(str(e))
            self.logger.exception(f"Error uploading {filename} to bucket: {exception}")

    def download_from_bucket(self, filename: str, destination_path: Optional[str] = ""):
        """
        Downloads a file from the specified bucket in Google Cloud Storage.

        Args:
            filename (str): The name of the file in the bucket.
            destination_path (str, optional): The local path where the file will be saved. If not provided,
                                               it will be saved in the current working directory.
        """
        try:
            bucket = self.get_bucket(self.bucket_name)
            blob = bucket.blob(filename)

            if not destination_path:
                destination_path = os.path.join(os.getcwd(), filename)

            blob.download_to_filename(destination_path)
            self.logger.info(
                f"{filename} successfully downloaded to {destination_path}."
            )

        except Exception as e:
            exception = extract_error_from_exceptions(str(e))
            self.logger.exception(
                f"Error downloading {filename} from bucket: {exception}"
            )

    def read_from_bucket(self, filename: str) -> bytes:
        """
        Reads a file from the specified bucket in Google Cloud Storage.

        Args:
            filename (str): The name of the file in the bucket.

        Returns:
            bytes: The content of the file as bytes.
        """
        try:
            bucket = self.get_bucket(self.bucket_name)
            blob = bucket.blob(filename)
            with blob.open("rb") as byte_data:
                byte_content = byte_data.read()

            return byte_content

        except Exception as e:
            exception = extract_error_from_exceptions(str(e))
            self.logger.exception(
                f"Error reading {filename} from {self.bucket_name}: {exception}"
            )
            return None

    def list_files_from_bucket(self, folder: Optional[str] = "downloads/"):
        """
        Lists files from the specified folder in the bucket.

        Args:
            folder (str, optional): The folder path within the bucket. Default is 'downloads/'.

        Returns:
            list: A list of file names within the specified folder.
        """
        try:
            blobs = self.list_blobs(self.bucket_name, prefix=folder)
            self.logger.info(
                f"Successfully listed {self.bucket_name}'s {folder} files."
            )
            return [blob.name for blob in blobs]

        except Exception as e:
            exception = extract_error_from_exceptions(str(e))
            self.logger.exception(
                f"Error listing {self.bucket_name}'s {folder} folder: {exception}"
            )
            return []
