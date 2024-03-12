from typing import Tuple
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset


def load_data(dataset_name: Tuple[str, str], cache_dir) -> dict:
    """
    Load a dataset.
    :param dataset_name: The name of the dataset.
    :param cache_dir: The cache directory.
    :return: The dataset.
    """
    dataset = load_dataset(
        *dataset_name,
        cache_dir=cache_dir,
    )
    return dataset


def _tokenize_for_seq_classification(tokenizer: AutoTokenizer, batch: dict) -> dict:
    """
    Tokenize a batch for sequence classification.
    :param tokenizer: The tokenizer.
    :param batch: The batch.
    :return: The tokenized batch.
    """
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
    )


def get_tokens_for_seq_classification(tokenizer: AutoTokenizer, dataset: dict) -> dict:
    """
    Get tokens for sequence classification.
    :param tokenizer: The tokenizer.
    :param dataset: The dataset.
    :return: The tokenized dataset.
    """

    partial_tokenize_for_seq_classification = partial(
        _tokenize_for_seq_classification, tokenizer
    )

    tokenized = dataset.map(partial_tokenize_for_seq_classification, batched=True)
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized = tokenized.rename_columns({"label": "labels"})
    tokenized.set_format("torch")
    return tokenized
