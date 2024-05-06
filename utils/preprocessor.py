import concurrent.futures
import logging
import os
import re
import uuid
from io import BytesIO
from typing import Optional

from dotenv import load_dotenv
from langchain_text_splitters import TokenTextSplitter
from pypdf import PdfReader
from transformers import AutoTokenizer

from utils import CloudStorage
from utils.configurations import config_from_file

load_dotenv()

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


def convert_bytes_to_text(
    byte_content: bytes, skip_pages: Optional[list[int]] = []
) -> str:
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
        logger.debug("PdfReader successfully converted byte stream to text.")

        if skip_pages:
            logger.info(f"Skipped pages: {skip_pages}")

    except Exception as e:
        logger.exception(
            f"Failed to open the PDF. Returning an empty string. Error: {e}"
        )
        return ""

    page_text = []
    for page_num, page in enumerate(reader.pages, 1):

        if page_num in skip_pages:
            continue

        text = page.extract_text() or ""
        page_text.append(text)

    return " ".join(page_text)


class Preprocessor:

    def __init__(
        self,
        remove_empty_lines: bool = True,
        remove_extra_whitespaces: bool = True,
        remove_non_utf8_characters: bool = True,
        remove_emojis: bool = True,
        replace_special_characters: bool = True,
        remove_regex: Optional[list[str]] = [],
    ):
        """
        Initializes a TextCleaner object with specified text processing options.

        Args:
            remove_empty_lines (bool, optional): Whether to remove empty lines. Defaults to True.
            remove_extra_whitespaces (bool, optional): Whether to remove excess whitespace. Defaults to True.
            remove_non_utf8_characters (bool, optional): Whether to remove non-UTF-8 characters. Defaults to True.
            remove_emojis (bool, optional): Whether to remove emojis. Defaults to True.
            replace_special_characters (bool, optional): Whether to remove special characters. Defaults to True.
            remove_regex: Regex to match and replace substrings by "". Defaults to [].
        """
        self.remove_empty_lines = remove_empty_lines
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.remove_non_utf8_characters = remove_non_utf8_characters
        self.remove_emojis = remove_emojis
        self.replace_special_characters = replace_special_characters
        self.remove_regex = remove_regex

    def _preprocess(self, text: str) -> str:
        """
        Prepares text for chunking

        Args:
            text (str): The input text string to be cleaned.

        Returns:
            str: The cleaned text string.
        """
        if self.remove_empty_lines:
            text = self._remove_empty_lines(text)

        if self.remove_extra_whitespaces:
            text = self._remove_excess_whitespace(text)

        if self.replace_special_characters:
            text = self._replace_special_characters(text)

        if self.remove_regex:
            text = self._remove_regex(text, self.remove_regex)

        if self.remove_emojis:
            text = self._remove_emojis(text)

        if self.remove_non_utf8_characters:
            text = self._remove_non_utf8_characters(text)

        return text

    def _remove_empty_lines(self, text: str) -> str:
        """
        Remove empty lines and lines that contain nothing but whitespaces from text.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The text with empty lines removed.
        """
        lines = text.split("\n")
        non_empty_lines = filter(lambda line: line.strip() != "", lines)

        return "\n".join(non_empty_lines)

    def _remove_excess_whitespace(self, text: str) -> str:
        """
        Removes excess whitespace from a text string.

        Args:
            text (str): The input text string to remove excess whitespace from.

        Returns:
            str: The text string with excess whitespace removed.
        """
        text = re.sub(r"\n+", " ", text)  ## remove multiple newlines
        text = re.sub(r"\s+", " ", text)  ## remove general whitespace

        return text.strip()

    def _remove_non_utf8_characters(self, text: str) -> str:
        """
        Cleans a string from non-UTF-8 characters.

        Args:
            text (str): The input string to be cleaned.

        Returns:
            str: The cleaned string containing only UTF-8 characters.
        """
        utf8_bytes = text.encode("utf-8", errors="ignore")
        text = utf8_bytes.decode("utf-8")

        return text

    def _remove_emojis(self, text: str) -> str:
        """
        Removes emojis from a string.

        Args:
            text (str): The input string containing emojis and emoticons.

        Returns:
            str: The input string with emojis and emoticons removed.
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r"", text)

        return text

    def _replace_special_characters(self, text: str) -> str:
        """
        Replaces special characters and non breaking whitespaces from a string.

        Args:
            text (str): The input string containing emojis and emoticons.

        Returns:
            str: The input string without unicode characters and non breaking whitespaces.
        """
        config = config_from_file(os.environ["PROCESS_CONFIG_PATH"])
        codes_to_replace = config["patterns"]["replacements"]
        for code, replacement in codes_to_replace.items():
            text = re.sub(code, replacement, text)

        return text

    def _remove_regex(self, text: str, regex: list[str]) -> str:
        """
        Remove substrings that match the specified regex from the text.

         Args:
            text (str): The input string to replace the expression.

        Returns:
            str: The input string without the substrings that match the regex.
        """

        for pattern in regex:
            text = re.sub(pattern, "", text).strip()

        return text


class TextSplitter(Preprocessor):
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
        convert_bytes_to_text(byte_content): Converts byte content of a PDF file to a text string, extracts text from each page,
                                    and optionally preprocesses it.

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
        preprocess: bool = True,
        config_path: str = os.environ["DATASET_CONFIG_PATH"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config_from_file(config_path)
        self.preprocess = preprocess
        self.min_tokens = self.config["tokenizer"]["min_tokens"]
        self.max_tokens = self.config["tokenizer"]["max_tokens"]
        self._tokenizer = None
        self._embedder = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            config = self.config.get("tokenizer", {})
            model_id = config.get("tokenizer_model")
            if model_id is None:
                raise ValueError(
                    "Config must include 'tokenizer_model' with a HF model ID."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        return self._tokenizer

    def _split_paragraphs_to_sentences(self, text: str, pattern: str) -> list:
        """ """
        sentences = re.split(pattern, text)
        logger.debug(f"The original text is split to {len(sentences)} sentences.")

        return sentences

    def _merge_sentences_multi_thread(
        self, sentences: list[str], num_threads: Optional[int] = -1
    ) -> list[str]:
        """
        Merge short sentences into longer sentences based on a minimum token count, ensuring not to exceed a maximum token limit.

        Args:
            sentences (list[str]): A list of short sentences.
            threads (int): The number of threads to use for parallel processing. Defaults to -1.
                        If -1 is provided, the function will use the maximum available logical cores.

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
        num_threads = (
            min(max_threads, len(sentences) // 2) if num_threads >= 1 else max_threads
        )
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

    def chunk_text(self, filename: str, text: str, pattern: str) -> list:
        """ """
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


class PDFChunker(Preprocessor):
    """
    A class for chunking the text content of PDF documents, with optional preprocessing of the extracted text.

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
        _text_splitter (langchain_text_splitters.TokenTextSplitter, optional): Text splitter configured with the tokenizer
                                                                              and settings from the config. Lazy-loaded upon first access.

    Methods:
        tokenizer (property): Returns a tokenizer instance. Initializes the tokenizer if it is not already loaded.
        text_splitter (property): Returns a text splitter instance. Initializes the text splitter if it is not already loaded.
        convert_bytes_to_text(byte_content): Converts byte content of a PDF file to a text string, extracts text from each page,
                                    and optionally preprocesses it.

    Args:
        preprocess (bool, optional): If set to True, the text extracted from PDF files will be preprocessed using
                                     methods inherited from `Preprocessor`. Defaults to True.
        config_path (str, optional): The path to the YAML configuration file that specifies the tokenizer and text splitter
                                     settings. Defaults to "configs/process.yaml".
        **kwargs: Additional keyword arguments are passed to the `Preprocessor` initializer.

    Raises:
        ValueError: If the configuration file does not contain required keys for tokenizer model or text splitting parameters.

    Example:
        chunker = PDFChunker(preprocess=True, config_path="path/to/config.yaml")
        with open("example.pdf", "rb") as file:
            byte_content = file.read()
        text = chunker.convert_bytes_to_text(byte_content)
        chunks = chunker.text_splitter.split(text)
    """

    def __init__(
        self,
        preprocess: bool = True,
        skip_book_cover: bool = True,
        config_path: str = "configs/process.yaml",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config_from_file(config_path)
        self.preprocess = preprocess
        self.skip_book_cover = skip_book_cover
        self._tokenizer = None
        self._text_splitter = None

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

    @property
    def text_splitter(self):
        if self._text_splitter is None:
            tokenizer_config = self.config.get("tokenizer", {})
            chunk_size = tokenizer_config.get("chunk_size")
            chunk_overlap = tokenizer_config.get("chunk_overlap")

            if chunk_size is None or chunk_overlap is None:
                raise ValueError("Config must include 'chunk_size' and 'overlap'.")

            self._text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                self.tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        return self._text_splitter
