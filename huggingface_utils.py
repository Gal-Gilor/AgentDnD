import concurrent.futures
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from langchain_text_splitters import TokenTextSplitter
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from utils.configurations import config_from_file

load_dotenv()

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


class HFEmbedder:
    """A class for embedding text using Hugging Face transformers.

    Args:
        path_to_config (str, optional): Path to the configuration file.
            Defaults to "config.yaml".

    Properties:
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer instance lazily loaded from the configuration.
        model (transformers.PreTrainedModel): A model instance lazily loaded from the configuration.

    Methods:
        _generate_embedding(model, tokenizer, texts) -> torch.Tensor:
            Generate embeddings for a list of texts using a given model and tokenizer.
        generate_embeddings_multi_thread(texts) -> list[str]:
            Generate embeddings for multiple texts using multiple threads.


    """

    def __init__(
        self,
        path_to_config: Optional[str] = "config.yaml",
    ):
        self.config = config_from_file(path_to_config)
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        """Return a tokenizer instance lazily loaded from the configuration.

        Returns:
            transformers.PreTrainedTokenizer: A tokenizer instance.
        """
        if not self._tokenizer:
            config = self.config.get("embedding", {})
            model_id = config.get("model_id")
            if not model_id:
                raise ValueError(
                    "Config must include 'embedding.model_id' with a valid HF model ID."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        return self._tokenizer

    @property
    def model(self):
        """Return a model instance lazily loaded from the configuration.

        Returns:
            transformers.PreTrainedModel: A model instance.
        """
        if not self._model:
            config = self.config.get("embedding", {})
            model_id = config.get("model_id")
            if not model_id:
                raise ValueError(
                    "Config must include 'embedding.model_id' with a valid HF model ID."
                )
            self._model = AutoModel.from_pretrained(model_id)
        return self._model

    def _generate_embedding(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        texts: list[str],
    ) -> torch.Tensor:
        """Generate embeddings for a list of texts using a given model and tokenizer.

        Args:
            model (transformers.PreTrainedModel): The pre-trained model for generating embeddings.
            tokenizer (transformers.PreTrainedTokenizerFast): The tokenizer for tokenizing the input texts.
            texts (list[str]): A list of texts for which embeddings are to be generated.

        Returns:
            torch.Tensor: A tensor containing the embeddings.
        """

        encoded_input = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        model.eval()
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]

        return F.normalize(sentence_embeddings, p=2, dim=1)

    def generate_embeddings_multi_thread(self, texts: list[str]) -> list[str]:
        """Generate embeddings for multiple texts using multiple threads.

        Args:
            texts (List[str]): A list of texts for which embeddings are to be generated.

        Returns:
            list[str]: A list of embeddings corresponding to the input texts.
        """

        tokenizer = self.tokenizer
        model = self.model

        # define the number of threads to use
        max_threads = os.cpu_count()
        num_threads = min(max_threads, len(texts))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Define a function to generate embeddings for a single text
            def generate_embedding(text):
                return self._generate_embedding(model, tokenizer, [text])

            # Submit tasks to the executor for each text
            embeddings = list(executor.map(generate_embedding, texts))

        return embeddings


if __name__ == "__main__":
    embedder = HFEmbedder(path_to_config="configs/config.yaml")

    # Generate embeddings for multiple texts using multiple threads
    texts = ["This is a sample text.", "Another example text.", "Yet another text."]

    embeddings = embedder.generate_embeddings_multi_thread(texts)
    logger.info(str(len(embeddings)))
