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


class CustomCloudStorage(storage.Client):
    def __init__(
        self,
        logger_config: Optional[str] = "configs/logger.yaml",
    ):
        """
        Initialize the Google Cloud Storage client with Compute Engine credentials.
        """
        # Use the default credentials associated with the notebook instance
        credentials = compute_engine.Credentials()
        super().__init__(credentials=credentials)
        self.logger = logger_from_file(logger_config)

    def upload_to_bucket(self, bucket_name: str, filename: str, file_content: str):
        """
        Upload file to a specified bucket in Google Cloud Storage.

        Args:
            bucket_name (str): The name of the bucket.
            filename (str): The name of the file in the bucket.
            file_content (str): The content to upload as a string.
        """
        try:
            # Get the bucket object
            bucket = self.bucket(bucket_name)

            # Create a blob object in the bucket
            blob = bucket.blob(filename)

            # Upload the string variable to the blob
            blob.upload_from_string(file_content)

            self.logger.info(f"{filename} successfully uploaded to {bucket_name}.")

        except Exception as e:
            self.logger.exception(f"Error uploading {filename} to bucket: {e}")

    def download_from_bucket(
        self, bucket_name: str, filename: str, destination_path: str
    ):
        """
        Download a file from a specified bucket in Google Cloud Storage.

        Args:
            bucket_name (str): The name of the bucket.
            filename (str): The name of the file in the bucket.
            destination_path (str): The local path where the file will be saved.
        """
        try:
            # Get the bucket object
            bucket = self.bucket(bucket_name)

            # Get the blob object
            blob = bucket.blob(filename)

            # Download the file
            blob.download_to_filename(destination_path)
            self.logger.info(f"{filename} successfully download to {destination_path}.")

        except Exception as e:
            self.logger.exception(f"Error downloading {filename} from bucket: {e}")


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
        gcstorage = CustomCloudStorage(logger_config=logger_config)
        gcstorage.download_from_bucket(bucket_name, blob_name, blob_name)
