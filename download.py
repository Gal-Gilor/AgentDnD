import logging
import os

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from utils import CloudStorage
from utils.configurations import config_from_file

load_dotenv()

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


def download_as_bytes(url: str) -> bytes:
    """
    Downloads the content of a URL as bytes.

    Args:
        url (str): The URL to download content from.

    Returns:
        bytes: The content of the URL as bytes.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for non-200 response
        logger.info(f"Successfully downloaded {url}.")

        return response.content

    except requests.HTTPError as e:
        logger.error(f"HTTP error downloading {url}: {e}")
        raise

    except Exception as e:
        logger.exception(f"Error downloading {url}: {e}")


if __name__ == "__main__":
    gcstore = CloudStorage()
    config = config_from_file(os.environ["DOWNLOADS_CONFIG_PATH"])
    to_download = config["downloads"]
    outfolder = config["outfolder"]

    for filename, url in tqdm(to_download.items()):
        byte_content = download_as_bytes(url)

        if not byte_content:
            logger.info(f"Download failed. Skipping {filename}.")
            continue

        destination = os.path.join(outfolder, filename)
        gcstore.upload_to_bucket(byte_content, outfolder)
