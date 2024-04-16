import logging
import os
import re
from io import BytesIO

from dotenv import load_dotenv
from langchain_text_splitters import TokenTextSplitter
from pypdf import PdfReader
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import CloudStorage
from utils.configurations import config_from_file

load_dotenv()

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


class Preprocessor:

    def __init__(
        self,
        remove_empty_lines: bool = False,
        remove_extra_whitespaces: bool = True,
        remove_non_utf8_characters: bool = True,
        remove_emojis: bool = True,
    ):
        """
        Initializes a TextCleaner object with specified text processing options.

        Args:
            remove_empty_lines (bool, optional): Whether to remove empty lines. Defaults to False.
            remove_extra_whitespaces (bool, optional): Whether to remove excess whitespace. Defaults to True.
            remove_non_utf8_characters (bool, optional): Whether to remove non-UTF-8 characters. Defaults to True.
            remove_emojis (bool, optional): Whether to remove emojis. Defaults to True.
        """
        self.remove_empty_lines = remove_empty_lines
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.remove_non_utf8_characters = remove_non_utf8_characters
        self.remove_emojis = remove_emojis

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

        if self.remove_emojis:
            text = self._remove_emojis(text)

        if self.remove_non_utf8_characters:
            text = self._remove_non_utf8_characters(text)

        if self.remove_extra_whitespaces:
            text = self._remove_excess_whitespace(text)

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
        text = re.sub(r"\n\n+", "\n", text)  ## remove multiple newlines
        text = re.sub(r"\s\s+", " ", text)  ## remove general whitespace

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


class PDFChunker(Preprocessor):
    """
    A class for chunking the text content of PDF documents, with optional preprocessing of the extracted text.

    This class extends `Preprocessor` to include the functionality to read PDF files, preprocess their textual content,
    and then split this content into manageable chunks based on the configuration provided in a YAML file.

    Attributes:
        preprocess (bool): Determines whether the text extracted from PDFs should be preprocessed.
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
        byte_to_text(byte_content): Converts byte content of a PDF file to a text string, extracts text from each page,
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
        text = chunker.byte_to_text(byte_content)
        chunks = chunker.text_splitter.split(text)
    """

    def __init__(
        self,
        preprocess: bool = True,
        config_path: str = "configs/process.yaml",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config_from_file(config_path)
        self.preprocess = preprocess
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
            chunk_overlap = tokenizer_config.get("overlap")

            if chunk_size is None or chunk_overlap is None:
                raise ValueError("Config must include 'chunk_size' and 'overlap'.")

            self._text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                self.tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        return self._text_splitter

    def byte_to_text(self, byte_content: bytes) -> str:
        """
        Extracts text content from byte data of a PDF file, preprocesses it if specified,
        and returns the processed text as a string.

        Args:
            byte_content (bytes): Byte content of the PDF file.

        Returns:
            str: Processed text extracted from the PDF file.
        """
        byte_stream = BytesIO(byte_content)
        reader = PdfReader(byte_stream)

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        if self.preprocess:
            text = self._preprocess(text)

        return text


if __name__ == "__main__":
    gcstore = CloudStorage()
    chunker = PDFChunker(config_path=os.environ["PROCESS_CONFIG_PATH"])
    files = gcstore.list_files_from_bucket()
    logger.debug(f"Files identified: {files}")

    for filepath in tqdm(files):
        logger.debug(f"File: {filepath}")
        byte_content = gcstore.read_from_bucket(filepath)

        if not byte_content:
            logger.info(f"Download failed. Skipping {filepath}.")
            continue

        print(len(chunker.byte_to_text(byte_content)))
