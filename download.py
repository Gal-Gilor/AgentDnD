import argparse
import os
from typing import Optional

import requests
from google.auth import compute_engine
from google.cloud import storage

from utils.configurations import config_from_file, logger_from_file


class DownloadPDF:
    """
    A class for downloading PDF files based on configuration.

    Args:
        downloads_config (str, optional): Path to the YAML configuration file
            containing download URLs and filenames. Default is "configs/downloads.yaml".
        logger_config (str, optional): Path to the YAML configuration file
            containing logger settings. Default is "configs/logger.yaml".

    Attributes:
        config (dict): Configuration loaded from downloads_config.
        logger (logging.Logger): Logger instance configured based on logger_config.
    """

    def __init__(
        self,
        downloads_config: Optional[str] = "configs/downloads.yaml",
        logger_config: Optional[str] = "configs/logger.yaml",
    ) -> None:
        self.config = config_from_file(downloads_config)
        self.logger = logger_from_file(logger_config)

    def _download(self, url: str, filename: str, directory: Optional[str]) -> None:
        """Download file from URL."""
        if directory:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"Folder {directory} created")

        filepath = os.path.join(directory or os.getcwd(), filename)

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for non-200 response
            with open(filepath, "wb") as file:
                file.write(response.content)

        except Exception as e:
            self.logger.exception(f"Error downloading {filename}: {e}")

    def download(self, outfolder: Optional[str] = "") -> None:
        """Download files."""
        if "downloads" not in self.config:
            self.logger.error("Key 'downloads' not found in the config.")
            return

        downloads = self.config["downloads"]
        outfolder = outfolder or self.config["outfolder"] or os.getcwd()

        for num_downloads, (filename, url) in enumerate(downloads.items(), 1):
            self._download(url, filename, outfolder)
            self.logger.info(f"Finished downloading {filename}")

        self.logger.info(f"Downloaded {num_downloads} files.")


class StorageWithLogging(storage.Client):
    """
    A custom class for interacting with Google Cloud Storage, enhanced with logging capabilities.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket to interact with.
        logger_config (str, optional): Path to the logger configuration file. Default is 'configs/logger.yaml'.
        service_key (str, optional): Path to the service account key file. Default is 'configs/agent-dnd-b7fc8efe8bdd.json'.
    """

    def __init__(
        self,
        bucket_name: str,
        logger_config: Optional[str] = "configs/logger.yaml",
        service_key: Optional[str] = "configs/agent-dnd-b7fc8efe8bdd.json",
    ):
        """
        Initialize the Google Cloud Storage client with Compute Engine credentials.
        """
        credentials = compute_engine.Credentials()
        super().__init__(credentials=credentials)
        self.logger = logger_from_file(logger_config)
        self.bucket_name = bucket_name

    def upload_to_bucket(self, filename: str, content: str):
        """
        Uploads a file to the specified bucket in Google Cloud Storage.

        Args:
            filename (str): The name of the file in the bucket.
            content (str): The content to upload as a string.
        """
        try:
            bucket = self.get_bucket(self.bucket_name)
            blob = bucket.blob(filename)
            blob.upload_from_string(content)
            self.logger.info(f"{filename} successfully uploaded to {self.bucket_name}.")

        except Exception as e:
            self.logger.exception(f"Error uploading {filename} to bucket: {e}")

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
            self.logger.exception(f"Error downloading {filename} from bucket: {e}")

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
            self.logger.exception(
                f"Error reading {filename} from {self.bucket_name}: {e}"
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
            self.logger.exception(
                f"Error listing {self.bucket_name}'s {folder} folder: {e}"
            )
            return []


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--downloads_config", type=str, default="configs/downloads.yaml"
    )
    argparser.add_argument("--logger_config", type=str, default="configs/logger.yaml")
    argparser.add_argument("--bucket_name", type=str, default="agent-dnd-storage")
    argparser.add_argument("--outfolder", type=str, default="")
    argparser.add_argument("--blob_name", type=str, default="")

    args = argparser.parse_args()
    logger_config = args.logger_config

    pdf_downloader = DownloadPDF(
        downloads_config=args.downloads_config, logger_config=logger_config
    )
    pdf_downloader.download(outfolder=args.outfolder)

    bucket_name = args.bucket_name
    blob_name = args.blob_name

    if bucket_name and blob_name:
        gcstorage = StorageWithLogging(
            bucket_name=bucket_name, logger_config=logger_config
        )
        gcstorage.download_from_bucket(blob_name, blob_name)
