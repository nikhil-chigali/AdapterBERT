"""
    This module contains utility functions for working with BERT models and adapters.
"""

from torch import nn
from transformers.models.bert.modeling_bert import BertModel
from transformers import AutoModelForSequenceClassification
from .adapter import AdapterConfig, BertAdaptedOutput, BertAdaptedSelfOutput


def load_sequence_classification_model(model_name: str, num_labels: int) -> BertModel:
    """
    Load a BERT model for sequence classification.

    :param model_name: The name of the BERT model.
    :param num_labels: The number of labels for the classification task.
    :return: The BERT model for sequence classification.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, cache_dir="cache\\model\\"
    )


def freeze_all_bert_layers(model: nn.Module) -> nn.Module:
    """
    Freeze all layers of a BERT model.

    :param model: The BERT model.
    :return: The BERT model with frozen layers.
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_bert_adapters(model: BertModel) -> BertModel:
    """
    Unfreeze all adapter layers in a BERT model.

    :param model: The BERT model.
    :return: The BERT model with unfrozen adapter layers.
    """
    for i, layer in enumerate(model.encoder.layer):
        assert hasattr(layer.attention.output, "adapter") and hasattr(
            layer.output, "adapter"
        ), f"Adapter not found in layer-{i}"

        for param in layer.attention.output.adapter.parameters():
            param.requires_grad = True

        for param in layer.output.adapter.parameters():
            param.requires_grad = True
    return model


def add_adapters_to_bert_layers(
    model: BertModel, adapter_cfg: AdapterConfig
) -> BertModel:
    """
    Add adapters to all layers of a BERT model.

    :param model: The BERT model.
    :param adapter_cfg: The adapter configuration.
    :return: The BERT model with adapters.
    """
    for layer in model.encoder.layer:
        layer.attention.output = BertAdaptedSelfOutput(
            layer.attention.output, adapter_cfg
        )
        layer.output = BertAdaptedOutput(layer.output, adapter_cfg)
    return model
