import abc
import concurrent.futures
import logging
import os
from abc import ABCMeta
from typing import Optional

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from utils.configurations import config_from_file

load_dotenv()

logging_dict = config_from_file(os.environ["LOGGING_CONFIG_PATH"])
logging.config.dictConfig(logging_dict)
logger = logging.getLogger(__name__)


class BaseReranker(metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "score") and callable(subclass.score) or NotImplemented

    @abc.abstractmethod
    def score(self, pair: list[str, str]):
        """Calculates the similiarty score between two texts"""
        raise NotImplementedError


class BGEReranker(BaseReranker):
    """A class for reranking chunks using Hugging Face transformers.

    Args:
        path_to_config (str, optional): Path to the configuration file.
            Defaults to "config.yaml".

    Properties:
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer instance lazily loaded from the configuration.
        model (transformers.AutoModelForSequenceClassification): A model instance lazily loaded from the configuration.

    Methods:
        score(model, tokenizer, pair) -> torch.Tensor:
            Calcuate the similarity the similarity score between a question and passage using a given model and tokenizer.
        rerank(texts) -> list[str]:
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
            config = self.config.get("reranker", {})
            model_id = config.get("model_id")
            if not model_id:
                raise ValueError(
                    "Config must include 'reranker.model_id' with a valid HF model ID."
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
            config = self.config.get("reranker", {})
            model_id = config.get("model_id")
            if not model_id:
                raise ValueError(
                    "Config must include 'reranker.model_id' with a valid HF model ID."
                )
            self._model = AutoModelForSequenceClassification.from_pretrained(model_id)

        return self._model

    def score(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        pair: list[str, str],
    ) -> torch.Tensor:
        """Calcuate the similarity  score between a question and passage using a given model and tokenizer.

        Args:
            model (PreTrainedModel): A pretrained HuggingFace model.
            tokenizer (PreTrainedTokenizerFast): A pretrained HuggingFace tokenizer.
            pair (list[str, str]): A two elements list containing two strings to calculate the similiarity

        Returns:
            Tensor: The similarity score between the query and the passage.
                The reranker is optimized based cross-entropy loss, so the relevance score is not bounded to a specific range.

        """

        model.eval()

        with torch.no_grad():
            inputs = tokenizer(
                [pair],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            scores = (
                model(**inputs, return_dict=True).logits.view(-1,).float()  # fmt: skip
            )

        return scores

    def rerank(self, pairs: list[str, str]) -> list[str]:
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
        num_threads = min(max_threads, len(pairs))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # calculate the similarity between the query and the retrieved content
            def _score(pair):
                return self.score(model, tokenizer, pair)

            embeddings = list(executor.map(_score, pairs))

        return embeddings
