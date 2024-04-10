import argparse
import os
from typing import Optional

import requests

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
            os.makedirs(directory, exist_ok=True)

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
        outfolder = outfolder or os.getcwd()

        for num_downloads, (filename, url) in enumerate(downloads.items(), 1):
            self._download(url, filename, outfolder)
            self.logger.info(f"Finished downloading {filename}")

        self.logger.info(f"Downloaded {num_downloads} files.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--downloads_config", type=str, default="configs/downloads.yaml"
    )
    argparser.add_argument("--logger_config", type=str, default="configs/logger.yaml")
    argparser.add_argument("--outfolder", type=str, default="downloads")
    args = argparser.parse_args()

    pdf_downloader = DownloadPDF(
        downloads_config=args.downloads_config, logger_config=args.logger_config
    )
    pdf_downloader.download(outfolder=args.outfolder)
