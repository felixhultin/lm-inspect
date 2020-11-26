import copy
import torch

from inspect import getfullargspec
from typing import Dict, List, Union

class Config():

    def __init__(self, attentions, X, Y, predictions, tokenized_inputs, tokenizer, device = 'cuda'):
        super().__init__()
        self.attentions = attentions
        self.X = X
        self.Y = Y
        self.indices = list(range(len(self.Y)))
        self.predictions = predictions
        self.tokenized_inputs = tokenized_inputs
        self.tokenizer = tokenizer
        self.device = device
        self.config = {}

    def __str__(self):
        if not self.config:
            return "no parameters set."
        return str(self.config)

    def __repr__(self):
        if not self.config:
            content = "no parameters set."
        else:
            content = str(self.config)
        return "<" + type(self).__name__ + " " + content + ">"

    def __call__(self, **kwargs):
        filter_args = {k:v for k,v in kwargs.items() if k in getfullargspec(self.filter).args}
        scope_args = {k:v for k,v in kwargs.items() if k in getfullargspec(self.scope).args}
        context_args = {k:v for k,v in kwargs.items() if k in getfullargspec(self.context).args}
        return self.filter(**filter_args).scope(**scope_args).context(**context_args)

    def filter(self, label: Union[list, str] = None, errors_only: bool = False, correct_only: bool = False):
        """Top word, position or word+positions by one or many label(s).

        Parameters
        ----------
        label: Union[list, str]
            Unique label(s) of the evaluation data Y.

        Raises
        ------
        ValueError
            If label is not in the evaluation data Y.
        """
        self.config['filter'] = { k:v for k,v in locals().items() if k != 'self' }
        indices = list(range(len(self.Y)))

        if errors_only:
            indices = [i for i in indices if self.Y[i] != self.predictions[i]]

        if correct_only:
            indices = [i for i in indices if self.Y[i] == self.predictions[i]]

        if label:
            labels = label if type(label) == list else [label]
            indices = [i for i in indices if self.Y[i] in labels]

        if not indices:
            msg = "All data was filtered out. Nothing to compare."
            raise ValueError(msg)

        self.X = [self.X[i] for i in indices]
        self.Y = [self.Y[i] for i in indices]
        self.tokenized_inputs = [self.tokenized_inputs[i] for i in indices]
        self.predictions = [self.predictions[i] for i in indices]
        self.attentions = self.attentions[indices, :, :, :]
        self.indices = indices
        return self

    def scope(self, layer: Union[list, int] = None, head : Union[list, int] = None,
              token_pos : Union[list, int] = None):
        """Returns specified layer(s), head(s) and token position(s) to inspect.

        Parameters
        ----------
        layer: Union[list, str]
            Layer(s) to inspect.
        head: Union[list, str]
            Head(s) to inspect.
        token_pos: Union[list, str]
            Token position(s) to inspect.
        """
        self.config['scope'] = { k:v for k,v in locals().items() if k != 'self' }
        attentions = self.attentions
        if layer is not None:
            layers = [layer] if type(layer) == int else layer
            attentions = attentions[:, layers, :, :]
        if head is not None:
            heads = [head] if type(head) == int else head
            attentions = attentions[:, :, heads, :]
        if token_pos is not None:
            token_positions = [token_pos] if type(token_pos) is int else token_pos
            attentions = attentions[:, :, :, token_positions]
        self.attentions = attentions
        return self

    def context(self, input_id: Union[List[int], int] = None):
        self.config['context'] = { k:v for k,v in locals().items() if k != 'self' }
        attentions = self.attentions
        if input_id is not None:
            input_ids = [input_id] if type(input_id) == int else [input_id[idx] for idx in self.indices]
            n_samples = attentions.shape[0]
            attentions = attentions[list(range(n_samples)), :, :, input_ids, :]
            attentions = attentions.unsqueeze(3)
        self.attentions = attentions
        return self
