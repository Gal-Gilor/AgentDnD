import argparse
import logging
import logging.config
import os
from typing import Optional

import requests
import yaml


class LoadConfigError(Exception):
    """Raised when there's an error loading configuration."""


class DownloadPDF:
    """
    A class for downloading PDF files based on configuration.

    Args:
        downloads_config (str, optional): Path to the YAML configuration file
            containing download URLs and filenames. Default is "downloads.yaml".
        logger_config (str, optional): Path to the YAML configuration file
            containing logger settings. Default is "logger.yaml".

    Attributes:
        config (dict): Configuration loaded from downloads_config.
        logger (logging.Logger): Logger instance configured based on logger_config.
    """

    def __init__(
        self,
        downloads_config: Optional[str] = "downloads.yaml",
        logger_config: Optional[str] = "logger.yaml",
    ) -> None:
        self.logger = self._setup_logger(logger_config)
        self.config = self._load_config(downloads_config)

    def _load_config(self, filename: str) -> dict:
        """Load YAML configuration file."""
        try:
            with open(filename) as file:
                return yaml.safe_load(file)
        except FileNotFoundError as e:

            raise LoadConfigError(f"Error loading config file '{filename}': {e}")
        except Exception as e:

            raise LoadConfigError(f"Unable to load config file '{filename}': {e}")

    def _setup_logger(self, filename: str) -> logging.Logger:
        """Configure logger using YAML configuration file."""
        try:
            with open(filename) as file:
                logging.config.dictConfig(yaml.safe_load(file))
            return logging.getLogger(__name__)

        except FileNotFoundError as e:
            raise LoadConfigError(f"Error loading logger config file '{filename}': {e}")

        except Exception as e:
            raise LoadConfigError(
                f"Unable to load logger config file '{filename}': {e}"
            )

    def _download(self, url: str, filename: str, directory: Optional[str]) -> None:
        """Download file from URL."""
        if directory:
            os.makedirs(directory, exist_ok=True)

        filepath = os.path.join(directory or os.getcwd(), filename)

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for non-200 response
            with open(filepath, "wb") as file:
                file.write(response.content)

        except Exception as e:
            self.logger.exception(f"Error downloading {filename}: {e}")

    def download(self, directory: Optional[str] = "") -> None:
        """Download files."""
        if "downloads" not in self.config:
            self.logger.error("Key 'downloads' not found in the config.")
            return

        downloads = self.config["downloads"]
        directory = directory or os.getcwd()

        for num_downloads, (filename, url) in enumerate(downloads.items(), 1):
            self._download(url, filename, directory)
            self.logger.info(f"Finished downloading {filename}")

        self.logger.info(f"Downloaded {num_downloads} files.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--downloads_config", type=str, default="downloads.yaml")
    argparser.add_argument("--logger_config", type=str, default="logger.yaml")
    argparser.add_argument("--directory", type=str, default="downloads")
    args = argparser.parse_args()

    pdf_downloader = DownloadPDF(
        downloads_config=args.downloads_config, logger_config=args.logger_config
    )
    pdf_downloader.download(directory=args.directory)
