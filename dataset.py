import json
import logging
import os
import re

from dotenv import load_dotenv
from pypdf import PdfReader

from utils import CloudStorage, SentenceTextSplitter
from utils.configurations import config_from_file
from utils.preprocessors import convert_bytes_to_text

load_dotenv()

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    config = config_from_file(os.environ["DATASET_CONFIG_PATH"])
    gcstore = CloudStorage()
    splitter = SentenceTextSplitter(
        config_path=os.environ["DATASET_CONFIG_PATH"],
        remove_regex=config["patterns"]["remove"],
    )

    # create the JSONL dataset
    nrows = 0
    files = gcstore.list_files_from_bucket(folder=config["infolder"])
    for file in files:
        logger.debug(file)
        byte_content = gcstore.read_from_bucket(file)
        text = convert_bytes_to_text(byte_content)

        filename = os.path.basename(file)
        # remove the version number from the file name
        filename, _ = os.path.splitext(filename)
        filename = re.sub(r" v\d+\.\d+", "", filename)

        chunks = splitter.chunk_text(filename, text)
        nrows += len(chunks)

        for chunk in chunks:
            with open("test5.json", "a") as f:
                f.write(json.dumps(chunk) + "\n")

    # read the dataset
    logger.info(
        f"The dataset was generated from {len(files)} files and contains {nrows} rows."
    )
    # with open(config["outfile"], "r") as data:
    #     oneshots = data.read()

    # # save the dataset in gcs
    # upload_path = config["outfolder"] + config["outfile"]
    # gcstore.upload_to_bucket(oneshots, upload_path)
