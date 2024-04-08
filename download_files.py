import yaml
import requests
import argparse
import os
import logging
import logging.config
from typing import Optional


class DownloadPDF:

    def __init__(
        self,
        downloads_config: Optional[str] = "downloads.yaml",
        logger_config: Optional[str] = "logger.yaml",
    ):
        self.config = self._config_from_path(downloads_config)
        self.logger = self._logger_from_path(logger_config)

    def _config_from_path(self, filename: str) -> dict:
        """ """
        with open(filename) as file:
            config = yaml.safe_load(file)

        return config

    def _logger_from_path(self, filename: str):
        """ """
        with open(filename) as file:
            config = yaml.safe_load(file)

        logging.config.dictConfig(config)
        logger = logging.getLogger(__name__)
        # Configure logging

        return logger

    def _download(self, url: str, filename: str, directory: Optional[str]) -> None:
        """ """
        # create a directory if it doesn't exist
        if directory:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # filepath = os.path.normpath(directory)
        filepath = os.path.join(directory, filename)

        r = requests.get(url)
        with open(filepath, "wb") as file:
            file.write(r.content)

    def download(self, directory: Optional[str] = "") -> None:
        """ """
        config = self.config
        assert "downloads" in config, "Expecting the key 'downloads' in the config"

        downloads = config["downloads"]
        directory = directory if directory else os.getcwd()
        for idx, (filename, url) in enumerate(downloads.items(), 1):

            self._download(url=url, filename=filename, directory=directory)
            self.logger.info(f"Finished downloading {filename}")

        self.logger.info(f"Downloaded {idx} files.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--downloads_config", type=str, default="downloads.yaml")
    argparser.add_argument("--logger_config", type=str, default="logger.yaml")
    argparser.add_argument("--directory", type=str, default="downloads")
    args = argparser.parse_args()

    downloads_config = args.downloads_config
    logger_config = args.logger_config
    directory = args.directory

    pdf_downloader = DownloadPDF(
        downloads_config=downloads_config, logger_config=logger_config
    )
    pdf_downloader.download(directory)
