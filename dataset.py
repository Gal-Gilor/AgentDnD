import concurrent.futures
import logging
import os
import re
import uuid
from io import BytesIO
from typing import List, Optional, Union

from datasets import load_dataset
from dotenv import load_dotenv
from pypdf import PdfReader
from transformers import AutoTokenizer

from utils import CloudStorage
from utils.configurations import config_from_file
from utils.preprocessor import PDFChunker, Preprocessor

load_dotenv()

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


class GenerateDataset(Preprocessor):
    """
    A class for converting PDFs to chunks, with optional preprocessing of the extracted text.

    This class extends `Preprocessor` to include the functionality to read PDF files, preprocess their textual content,
    and then split this content into manageable chunks based on the configuration provided in a YAML file.

    Attributes:
        preprocess (bool): Determines whether the text extracted from PDFs should be preprocessed.
        skip_book_ver (bool): Determines whether to skip the first page.
        config_path (str): Path to the configuration file that specifies tokenizer settings such as model ID,
                           chunk size, and chunk overlap.
        config (dict): Configuration loaded from the YAML file specified by `config_path`.
        _tokenizer (transformers.AutoTokenizer, optional): Tokenizer loaded based on the 'tokenizer_model'
                                                           specified in the config. Lazy-loaded upon first access.

    Methods:
        tokenizer (property): Returns a tokenizer instance. Initializes the tokenizer if it is not already loaded.
        convert_bytes_to_text(byte_content): Converts byte content of a PDF file to a text string, extracts text from each page,
                                    and optionally preprocesses it.

    Args:
        preprocess (bool, optional): If set to True, the text extracted from PDF files will be preprocessed using
                                     methods inherited from `Preprocessor`. Defaults to True.
        config_path (str, optional): The path to the YAML configuration file that specifies the tokenizer and text splitter
                                     settings. Defaults to "configs/process.yaml".
        min_tokens (Optional[int]): The minimum number of tokens for a merged sentence. Defaults to 400.
        **kwargs: Additional keyword arguments are passed to the `Preprocessor` initializer.

    Raises:
        ValueError: If the configuration file does not contain required keys for tokenizer model or text splitting parameters.
    """

    def __init__(
        self,
        preprocess: bool = True,
        skip_book_cover: bool = True,
        config_path: str = "configs/dataset.yaml",
        min_tokens: int = 400,
        padding: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config_from_file(config_path)
        self.preprocess = preprocess
        self.skip_book_cover = skip_book_cover
        self.min_tokens = min_tokens
        self.padding = padding
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            tokenizer_config = self.config.get("tokenizer", {})
            model_id = tokenizer_config.get("tokenizer_model")
            if model_id is None:
                raise ValueError(
                    "Config must include 'tokenizer_model' with a HF model ID."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        return self._tokenizer

    def split_paragraphs_to_sentences(text: str, pattern: str) -> list:
        """ """
        sentences = re.split(pattern, text)
        logger.debug(f"The original text is split to {len(sentences)} sentences.")

        return sentences

    def convert_bytes_to_text(self, byte_content: bytes) -> str:
        """
        Extracts text content from byte data of a PDF file, preprocesses it if specified,
        and returns the processed text as a string.

        Args:
            byte_content (bytes): Byte content of the PDF file.

        Returns:
            str: Processed text extracted from the PDF file.
        """
        byte_stream = BytesIO(byte_content)

        try:
            reader = PdfReader(byte_stream)
            pages = reader.pages[1:] if self.skip_book_cover else reader.pages
            logger.info(f"PdfReader Succeeded; skip first page {self.skip_book_cover}")

        except Exception as e:
            logger.exception(
                f"Failed to open the PDF. Returning an empty string. Error: {e}"
            )
            return ""

        page_information = []
        for page in pages:

            if self.skip_book_cover and page.page_number == 1:
                continue

            else:
                page_text = page.extract_text() or ""

                # optional text cleaning process
                try:
                    if self.preprocess:
                        page_text = self._preprocess(page_text)

                except Exception as e:
                    logger.exception(
                        f"Skipped page {page.page_number} due to preprocessing error. {e}"
                    )

                page_information.append(page_text)

        # append a new page break
        return "\n".join(page_information)

    def merge_sentences(self, sentences: list[str], threads: int = -1) -> list[str]:
        """
        Merge short sentences into longer sentences based on a minimum token count.

        Args:
            sentences (List[str]): A list of short sentences.
            threads (int): The number of threads to use for parallel processing. Defaults to -1.
                        If -1 is provided, the function will use the maximum available logical cores.

        Returns:
            List[str]: A list of merged sentences.
        """
        min_tokens = self.min_tokens
        merged_sentences = []
        current_sentence_tokens = []
        token_count = 0

        def process_sentence(sentence):
            nonlocal token_count
            token_count += token_counter(sentence)
            return sentence

        # Determine the number of threads to use
        if threads == -1:
            threads = os.cpu_count()

        else:
            threads = min(threads, os.cpu_count())

        # Use a ThreadPoolExecutor to process sentences in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # Process sentences in parallel
            for sentence in executor.map(process_sentence, sentences):
                current_sentence_tokens.append(sentence)
                if token_count >= min_tokens:
                    merged_sentences.append(" ".join(current_sentence_tokens))
                    current_sentence_tokens = []
                    token_count = 0

        # Add the last merged sentence if any
        if current_sentence_tokens:
            merged_sentences.append(" ".join(current_sentence_tokens))

        return merged_sentences


def split_paragraph(filename: str, text: str, pattern: str) -> list:
    """ """
    sentences = re.split(pattern, text)
    logger.debug(f"The original text is split to {len(sentences)} sentences.")

    chunks = []
    for idx, sentence in enumerate(sentences):

        chunks.append(
            {
                "id": str(uuid.uuid4()),
                "filename": filename,
                "text": sentence.strip(),
                "chunk_number": idx,
            }
        )

    return chunks


def tokenize(text: Union[str, list[str]]) -> list[float]:
    """ """
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    return input_ids


def token_counter(sentence: str) -> int:
    """
    Placeholder function to count tokens in a sentence.

    Args:
        sentence (str): The input sentence.

    Returns:
        int: The number of tokens in the sentence.
    """
    return len(sentence.split())


# def merge_sentences(sentences: list[str], min_tokens: Optional[int] = 256) -> list[str]:

#     sentences_deque = deque(sentences)
#     merged_sentences = []
#     temp_sentence = []

#     while sentences_deque:
#         popped = sentences_deque.popleft()

#         if len(tokenize(" ".join(temp_sentence))) < min_tokens:
#             temp_sentence.append(popped)

#         else:
#             if temp_sentence:
#                 merged_sentences.append(" ".join(temp_sentence))
#                 temp_sentence = []

#             else:
#                 merged_sentences.append(popped)

#     # capture the last chunk
#     if temp_sentence:
#         merged_sentences.append(" ".join(temp_sentence))

#     return merged_sentences


import json


def gen():
    """_summary_

    Args:
        filename (str): _description_
        text (str): _description_

    Yields:
        Iterator: _description_
    """
    sentences = sentences = ["a", "b", "c"]

    for idx, sentence in enumerate(sentences):
        with open("myjson.jsonl", "a") as f:
            test = {"filename": "test", "text": sentence, "chunk": idx}
            f.write(json.dumps(test) + "\n")


if __name__ == "__main__":
    # Example usage
    short_sentences = [
        "This is a short sentence.",
        "This is another short sentence.",
        "This is a third short sentence.",
        "This is a short sentence, again.",
    ]

    merged = merge_sentences(short_sentences, threads=-1)
    print(merged)
    # config = config_from_file(os.environ["DATASET_CONFIG_PATH"])
    # print(config["patterns"]["remove"])
    # gcstore = CloudStorage()
    # procces = PDFChunker(
    #     config_path=os.environ["PROCESS_CONFIG_PATH"],
    #     remove_regex=config["patterns"]["remove"],
    #     skip_book_cover=False,
    # )
    # files = gcstore.list_files_from_bucket(folder="onshots/")

    # for file in files[:1]:
    #     # logger.debug(f"################### {file} ")
    #     byte_content = gcstore.read_from_bucket(file)
    #     text = procces.convert_bytes_to_text(byte_content)

    #     # sentences = re.split(config["patterns"]["split"], text)
    #     chunks = split_paragraph(file, text, config["patterns"]["split"])
    #     logger.info(f"################### {len(chunks)} ")
    #     for chunk in chunks:
    #         with open("myjson.jsonl", "a") as f:
    #             f.write(json.dumps(chunk) + "\n")

    # dataset = load_dataset("json", data_files=["myjson.jsonl"])
    # print(dataset)
