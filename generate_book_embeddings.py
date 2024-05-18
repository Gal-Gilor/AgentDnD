import asyncio
import json
import logging
import os
import re
import time
import uuid
from argparse import ArgumentParser

from dotenv import load_dotenv

from utils import CloudStorage
from utils.configurations import config_from_file
from utils.preprocessors import convert_bytes_to_text
from utils.retrievers import BGERetriever
from utils.splitters import SentenceTextSplitter

load_dotenv()

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


async def create_chunks_async(
    texts: list[str], embeddings: list[float], filename: str, save_as: str
) -> dict:
    """ """

    for string, embedding in zip(texts, embeddings):
        chunk = {
            "id": str(uuid.uuid4()),
            "embedding": embedding.tolist(),
            "text": string,
            "filename": filename,
        }

        with open(save_as, "a") as f:
            json.dump(chunk, f)
            f.write("\n")


async def main(config_path: dict):
    """ """
    config = config_from_file(config_path)
    gcstore = CloudStorage()
    splitter = SentenceTextSplitter(
        config_path=config_path,
        remove_regex=config["patterns"]["remove"],
    )
    embedder = BGERetriever(config_path)
    embeddings_config = config.get("embeddings", {})
    destination, save_as = (
        embeddings_config["outfolder"],
        embeddings_config["filename"],
    )

    # list the files we are parsing from cloud storage
    files = gcstore.list_files_from_bucket(folder=config["downloads"]["outfolder"])
    logger.info(f"Destination folder contains {len(files)} files")

    start = time.time()
    for file in files:

        byte_content = await gcstore.read_from_bucket(file)

        # remove the version number from the file name
        filename = os.path.basename(file)
        filename, _ = os.path.splitext(filename)
        filename = re.sub(r" v\d+\.\d+", "", filename)
        logger.debug(f"Successfully read {filename} from {gcstore.bucket_name}")

        # convert the byte stream to text
        text = await convert_bytes_to_text(byte_content)
        inputs = splitter.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        logger.debug(f"{filename} contains {input_ids.shape[1]} tokens.")

        # chunk the document
        texts = await splitter.split_text(text)
        embedding_start = time.time()
        embeddings = await embedder.generate_embeddings(texts)
        await create_chunks_async(texts, embeddings, filename, save_as)
        logger.info(
            f"Finished embedding {filename} in {time.time() - embedding_start} seconds."
        )

    # upload the json file to gcs
    with open(save_as, "r") as f:
        chunks = f.read()

    upload_path = os.path.join(destination, save_as)
    gcstore.upload_to_bucket(chunks, upload_path)
    logger.info(
        f"Uploaded embeddings to {upload_path}. Total runtime: {time.time() - start}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Agent D&D",
        description="Parse, chunk, and embed D&D PDFs from GCP, and upload the embeddings to GCS.",
    )
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args()
    config_path = args.config_path

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(config_path=config_path))
