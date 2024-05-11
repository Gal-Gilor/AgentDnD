import concurrent.futures
import logging
import os
import re
import uuid
from typing import Optional

from dotenv import load_dotenv
from transformers import AutoTokenizer

from utils.configurations import config_from_file

from .preprocessors import Preprocessor

load_dotenv()

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


class SentenceTextSplitter(Preprocessor):
    """
    A class for converting PDFs to chunks, with optional preprocessing of the extracted text.

    This class extends `Preprocessor` to include the functionality to read PDF files, preprocess their textual content,
    and then split this content into manageable chunks based on the configuration provided in a YAML file.

    Attributes:
        preprocess (bool): Determines whether the text extracted from PDFs should be preprocessed.
        skip_book_ver (bool): Determines whether to skip the first page.
        config_path (str): Path to the configuration file that specifies tokenizer settings such as model ID, chunk size, and chunk overlap.
        config (dict): Configuration loaded from the YAML file specified by `config_path`.
        _tokenizer (transformers.AutoTokenizer, optional): Tokenizer loaded based on the 'tokenizer_model' specified in the config. Lazy-loaded upon first access.

    Methods:
        tokenizer (property): Returns a tokenizer instance. Initializes the tokenizer if it is not already loaded.

    Args:
        preprocess (bool, optional): If set to True, the text extracted from PDF files will be preprocessed using
                                     methods inherited from `Preprocessor`. Defaults to True.
        config_path (str, optional): The path to the YAML configuration file that specifies the tokenizer and text splitter
                                     settings. Defaults to "configs/process.yaml".
        min_tokens (Optional[int]): The minimum number of tokens for a merged sentence. Defaults to 256.
        **kwargs: Additional keyword arguments are passed to the `Preprocessor` initializer.

    Raises:
        ValueError: If the configuration file does not contain required keys for tokenizer model or text splitting parameters.
    """

    def __init__(
        self,
        config_path: str = os.environ["DATASET_CONFIG_PATH"],
        preprocess: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config_from_file(config_path)
        self.preprocess = preprocess
        self._tokenizer = None
        self._embedder = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            config = self.config.get("tokenizer", {})
            model_id = config.get("model_id")
            if model_id is None:
                raise ValueError(
                    "Config must include 'tokenizer.model_id' with a HF model ID."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.min_tokens = config.get("min_tokens", 256)
            self.max_tokens = config.get("max_tokens", 512)

        return self._tokenizer

    def _split_paragraphs_to_sentences(self, text: str, pattern: str) -> list:
        """ """
        sentences = re.split(pattern, text)
        logger.debug(f"The original text is split to {len(sentences)} sentences.")

        return sentences

    def _merge_sentences_multi_thread(self, sentences: list[str]) -> list[str]:
        """
        Merge short sentences into longer sentences based on a minimum token count, ensuring not to exceed a maximum token limit.

        Args:
            sentences (list[str]): A list of short sentences.

        Returns:
            list[str]: A list of merged sentences.
        """
        merged_sentences = []
        current_sentence_tokens = []
        token_count = 0

        def process_sentence(sentence):
            inputs = self.tokenizer(sentence, return_tensors="pt")
            input_ids = inputs["input_ids"]
            return sentence, input_ids.shape[1]  # Returning sentence and token count

        # Determine the number of threads to use
        max_threads = os.cpu_count()
        num_threads = min(max_threads, len(sentences) // 2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(process_sentence, sentences)
            for sentence, sentence_tokens in results:

                # check if adding this sentence would exceed the max token limit
                if token_count + sentence_tokens > self.max_tokens:
                    if current_sentence_tokens:
                        merged_sentences.append(" ".join(current_sentence_tokens))
                    current_sentence_tokens = [sentence]
                    token_count = sentence_tokens

                else:
                    # append the current sentence to the current merged sentence
                    current_sentence_tokens.append(sentence)
                    token_count += sentence_tokens

                    # if the current token count is enough, append to merged sentences and reset
                    if token_count >= self.min_tokens:
                        merged_sentences.append(" ".join(current_sentence_tokens))
                        current_sentence_tokens = []
                        token_count = 0

        # add the last merged sentence if it meets the minimum token requirement
        if current_sentence_tokens and token_count >= self.min_tokens:
            merged_sentences.append(" ".join(current_sentence_tokens))

        logger.debug(
            f"The original senteces are merged back to {len(merged_sentences)} sentences."
        )

        return merged_sentences

    def chunk_text(self, filename: str, text: str) -> list[dict]:
        """ """

        if self.preprocess:
            text = self._preprocess(text)

        patterns = self.config.get("patterns", {})
        pattern = patterns.get("split", "\n")
        sentences = self._split_paragraphs_to_sentences(text, pattern)
        merged_sentences = self._merge_sentences_multi_thread(sentences)

        chunks = []
        for idx, sentence in enumerate(merged_sentences):

            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "filename": filename,
                    "text": sentence.strip(),
                    "chunk_number": idx,
                }
            )
        return chunks
