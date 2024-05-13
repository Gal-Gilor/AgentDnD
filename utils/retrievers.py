import abc
import asyncio
import concurrent.futures
import logging
import os
from abc import ABCMeta
from typing import Optional

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
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


class BaseRetriever(metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "embed") and callable(subclass.embed) or NotImplemented

    @abc.abstractmethod
    def embed(self, text: str):
        """Generates text embedding"""
        raise NotImplementedError


class BGERetriever(BaseRetriever):
    """A class for embedding text using Hugging Face transformers.

    Args:
        path_to_config (str, optional): Path to the configuration file.
            Defaults to "config.yaml".

    Properties:
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer instance lazily loaded from the configuration.
        model (transformers.PreTrainedModel): A model instance lazily loaded from the configuration.

    Methods:
        embed(model, tokenizer, texts) -> torch.Tensor:
            Generate embeddings for a list of texts using a given model and tokenizer.
        generate_embeddings(texts) -> list[str]:
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
            config = self.config.get("embedder", {})
            model_id = config.get("model_id")
            if not model_id:
                raise ValueError(
                    "Config must include 'embedder.model_id' with a valid HF model ID."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.max_length = config.get("max_length", 256)

        return self._tokenizer

    @property
    def model(self):
        """Return a model instance lazily loaded from the configuration.

        Returns:
            transformers.PreTrainedModel: A model instance.
        """
        if not self._model:
            config = self.config.get("embedder", {})
            model_id = config.get("model_id")
            if not model_id:
                raise ValueError(
                    "Config must include 'embedder.model_id' with a valid HF model ID."
                )
            self._model = AutoModel.from_pretrained(model_id)

        return self._model

    async def embed(
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
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        model.eval()
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]

        return F.normalize(sentence_embeddings, p=2, dim=1)[0]

    async def generate_embeddings(self, texts: list[str]) -> list[str]:
        """Generate embeddings for multiple texts using multiple threads.

        Args:
            texts (List[str]): A list of texts for which embeddings are to be generated.

        Returns:
            list[str]: A list of embeddings corresponding to the input texts.
        """

        tokenizer = self.tokenizer
        model = self.model
        embeddings = [None] * len(texts)  # Placeholder for embeddings

        async def _embed(text):
            return await self.embed(model, tokenizer, [text])

        async def _process_text(text, index):
            embeddings[index] = await _embed(text)

        tasks = []
        for index, text in enumerate(texts):
            tasks.append(asyncio.create_task(_process_text(text, index)))

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        return embeddings

    # def generate_embeddings(self, texts: list[str]) -> list[str]:
    #     """Generate embeddings for multiple texts using multiple threads.

    #     Args:
    #         texts (List[str]): A list of texts for which embeddings are to be generated.

    #     Returns:
    #         list[str]: A list of embeddings corresponding to the input texts.
    #     """

    #     tokenizer = self.tokenizer
    #     model = self.model

    #     # define the number of threads to use
    #     max_threads = os.cpu_count()
    #     num_threads = min(max_threads, len(texts))

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    #         # Define a function to generate embeddings for a single text
    #         def _embed(text):
    #             return self.embed(model, tokenizer, [text])

    #         # Submit tasks to the executor for each text
    #         embeddings = list(executor.map(_embed, texts))

    #     return embeddings


# TODO
# class GTEEmbedder:
# model_path = 'Alibaba-NLP/gte-large-en-v1.5'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# # Tokenize the input texts
# batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

# outputs = model(**batch_dict)
# embeddings = outputs.last_hidden_state[:, 0]

# # (Optionally) normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:1] @ embeddings[1:].T) * 100
