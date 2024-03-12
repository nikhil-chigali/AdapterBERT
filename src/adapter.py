"""
    This module contains classes for adapter layers used in BERT model. 

    AdapterConfig:: NamedTuple that holds the configuration for an adapter layer.

    AdapterBlock:: PyTorch module that implements an adapter layer. It takes an AdapterConfig instance as input and applies a down projection, a non-linear activation, and an up projection to the input.

    BertAdaptedSelfOutput:: PyTorch module that applies an adapter layer to the output of a BERT self-attention layer. It takes a BertSelfOutput instance and an AdapterConfig instance as input.

    BertAdaptedOutput:: PyTorch module that applies an adapter layer to the output of a BERT layer. It takes a BertOutput instance and an AdapterConfig instance as input.
"""

from typing import NamedTuple, Union, Callable
import torch
from torch import nn
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfOutput, BertOutput


class AdapterConfig(NamedTuple):
    """
    Configuration for an adapter layer.

    :param hidden_dim: The size of the hidden layer.
    :param adapter_dim: The size of the adapter layer.
    :param adapter_act: The activation function to use.
    :param adapter_initializer_range: The initial range (std_dev) of the adapter layer weights.
    """

    hidden_dim: int
    adapter_dim: int
    adapter_act: Union[str, Callable]
    adapter_initializer_range: float


class AdapterBlock(nn.Module):
    """
    Adapter layer for BERT.

    :param adapter_cfg: Instance of AdapterConfig
    """

    def __init__(self, adapter_cfg: AdapterConfig):
        super(AdapterBlock, self).__init__()
        self.cfg = adapter_cfg

        self.down_project = nn.Linear(self.cfg.hidden_dim, self.cfg.adapter_dim)
        nn.init.normal_(
            self.down_project.weight, std=self.cfg.adapter_initializer_range
        )
        nn.init.zeros_(self.down_project.bias)

        if isinstance(self.cfg.adapter_act, str):
            self.nonlinear = ACT2FN[self.cfg.adapter_act]
        else:
            self.nonlinear = self.cfg.adapter_act

        self.up_project = nn.Linear(self.cfg.adapter_dim, self.cfg.hidden_dim)
        nn.init.normal_(self.up_project.weight, std=self.cfg.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states) -> torch.Tensor:
        """
        Forward pass of the adapter layer.

        :param hidden_states: The hidden states of the model.
        :return: The hidden states after applying the adapter layer.
        """
        down_project = self.down_project(hidden_states)
        nonlinear = self.nonlinear(down_project)
        up_project = self.up_project(nonlinear)
        return hidden_states + up_project


class BertAdaptedSelfOutput(nn.Module):
    """
    Adapter layer for BERT self-attention output.

    :param bert_self_output: Instance of BertSelfOutput
    :param adapter_cfg: Instance of AdapterConfig
    """

    def __init__(self, bert_self_output: BertSelfOutput, adapter_cfg: AdapterConfig):
        super(BertAdaptedSelfOutput, self).__init__()
        self.bert_self_output = bert_self_output
        self.adapter = AdapterBlock(adapter_cfg)

    def forward(
        self, hidden_states: torch.Tensor, attention_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the adapter layer.

        :param hidden_states: The hidden states of the model.
        :param attention_out: The input tensor.
        :return: The hidden states after applying the adapter layer.
        """
        hidden_states = self.bert_self_output.dense(hidden_states)
        hidden_states = self.bert_self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.bert_self_output.LayerNorm(hidden_states + attention_out)
        return hidden_states


class BertAdaptedOutput(nn.Module):
    """
    Adapter layer for BERT output.
    :param bert_output: Instance of BertOutput
    :param adapter_cfg: Instance of AdapterConfig
    """

    def __init__(self, bert_output: BertOutput, adapter_cfg: AdapterConfig):
        super(BertAdaptedOutput, self).__init__()
        self.bert_output = bert_output
        self.adapter = AdapterBlock(adapter_cfg)

    def forward(
        self, intermediate_state: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the adapter layer.
        :param intermediate_state: The hidden states of the model.
        :param input_tensor: The input tensor.
        :return: The hidden states after applying the adapter layer.
        """
        intermediate_state = self.bert_output.dense(intermediate_state)
        intermediate_state = self.bert_output.dropout(intermediate_state)
        intermediate_state = self.adapter(intermediate_state)
        out_state = self.bert_output.LayerNorm(intermediate_state + input_tensor)
        return out_state
