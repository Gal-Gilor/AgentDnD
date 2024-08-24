import os
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import asyncio
import itertools
import json
import logging
import re
import time
import uuid
from typing import Dict, Iterable, Iterator, List, Optional

import nest_asyncio
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from tqdm import tqdm

from utils import CloudStorage
from utils.configurations import config_from_file
from utils.preprocessors import convert_bytes_to_text
from utils.retrievers import BGERetriever
from utils.splitters import SentenceTextSplitter

nest_asyncio.apply()
load_dotenv()


config = config_from_file(os.environ["QA_INDEX_CONFIG_PATH"])

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


async def create_chunks_async(filename: str, texts: List[str]) -> Dict:
    """ """

    chunks = []
    for idx, string in enumerate(texts, 1):
        chunk = {
            "id": str(uuid.uuid4()),
            "values": [],
            "sparse_values": {"indecies": [], "values": []},
            "metadata": {
                "chunk": idx,
                "text": string,
                "filename": filename,
            },
        }

        chunks.append(chunk)

    return chunks


def create_pinecone_index(config: dict) -> None:
    """ """

    embedding_config = config.get("embedder", {})
    indexer_config = config.get("indexer", {})
    embedding_dim = embedding_config.get("embedding_dim", 1024)
    cloud = indexer_config.get("cloud", "aws")
    region = indexer_config.get("region", "us-east-1")
    index_name = indexer_config.get("index_qa_name", "agent_dnd_qa")
    metric = indexer_config.get("metric", "dotproduct")

    client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"), pool_threads=30)
    spec = ServerlessSpec(cloud=cloud, region=region)

    if index_name in client.list_indexes().names():
        logger.info(f"Index {index_name} already exists.")

    else:
        client.create_index(
            index_name, dimension=embedding_dim, metric=metric, spec=spec
        )
        # wait for index to be initialized
        while not client.describe_index(index_name).status["ready"]:
            time.sleep(1)

        logger.info(f"Index {index_name} created successfully.")


def batching_chunks(
    iterable: Iterable[Dict], batch_size: Optional[int] = 100
) -> Iterator[Dict]:
    """A helper function to break an iterable into chunks of size batch_size."""

    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


async def prepare_chunks_for_indexing(files, gcstore) -> List[Dict]:
    """ """

    splitter = SentenceTextSplitter(config=config)
    retriever = BGERetriever(config)

    document_chunks = []
    bm25_dataset = []
    for file in files:

        start_time = time.time()
        logger.debug(f"Began processesing {file} from {gcstore.bucket_name}")
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

        # split the text
        split_text = await splitter.split_text(text)
        bm25_dataset.extend(split_text)
        document_chunks.extend(await create_chunks_async(filename, split_text))

    logger.info(
        f"Finished splitting the documents in {time.time() - start_time} seconds."
    )

    # fit a bm25encoder for sparse encoding
    bm25 = BM25Encoder()
    bm25_fit_start = time.time()
    bm25.fit(bm25_dataset)
    logger.info(
        f"Finished fitting the BM25Encoder in {time.time() - bm25_fit_start} seconds."
    )

    # embed the text
    model = retriever.model
    tokenizer = retriever.tokenizer
    embedding_start = time.time()
    for document in tqdm(document_chunks):
        metadata = document["metadata"]
        text = metadata["text"]
        embedding = await retriever.embed(model, tokenizer, text)
        document["sparse_values"] = bm25.encode_documents(text)
        document["values"] = embedding.tolist()

    logger.info(
        f"Finished embedding the documents in {time.time() - embedding_start} seconds."
    )

    return document_chunks


if __name__ == "__main__":

    # identify the files to embed
    gcstore = CloudStorage()
    files = gcstore.list_files_from_bucket(folder=config["downloads"]["outfolder"])
    loop = asyncio.get_event_loop()
    chunks = loop.run_until_complete(prepare_chunks_for_indexing(files, gcstore))

    # create and connect to pinecone
    indexer_config = config.get("indexer", {})
    index_name = indexer_config.get("index_qa_name", "agent-dnd-qa")
    _ = create_pinecone_index(config)

    # upsert the chunks to the index async
    client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"), pool_threads=30)
    with client.Index(index_name, pool_threads=30) as index:
        # Send requests in parallel
        async_results = [
            index.upsert(vectors=ids_vectors_chunk, async_req=True)
            for ids_vectors_chunk in batching_chunks(chunks, batch_size=2)
        ]
        # Wait for and retrieve responses (this raises in case of error)
        [async_result.get() for async_result in async_results]
