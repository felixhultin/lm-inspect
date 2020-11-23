import copy
import json
import logging

import transformers
import torch

from inspect import getfullargspec
from typing import Dict, List, Union, Tuple

from torch.utils.data import Dataset, DataLoader

from .helpers import camel_to_snake
from .configuration import Config
from .methods.topk import TopkMostAttendedTo


class DocumentDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

class DocumentBatcher:

  def __call__(self, XY):
        X = [x for x, _ in XY]
        Y = [y for _, y in XY]
        return X, Y


class LanguageModelInspector:

    def __init__(self, nn, X, Y, tokenizer, label_encoder = None, device = 'cuda'):
        self.nn = nn
        self.lm = self._find_pretrained(nn)
        self.config = self.lm.config
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.device = device
        # inspection methods
        method_classes = [TopkMostAttendedTo]
        for c in method_classes:
            method_name = camel_to_snake(c.__name__)
            self._bind_method(c.inspect, method_name)
        # Values to set in .evaluate()
        self.X = X
        self.Y = Y
        self.tokenized_inputs = None
        self.attentions = []
        self.predictions = None
        self.config  = self.evaluate(X, Y)

    def _bind_method(self, method, name):
        def wrapped(*args, **kwargs):
            return method(self.config, *args, **kwargs)
        wrapped.__doc__ = method.__doc__
        setattr(self, name, wrapped)

    def configure(self, **kwargs):
        self.config = Config(
            self.attentions,
            self.X,
            self.Y,
            self.predictions,
            self.tokenized_inputs,
            self.tokenizer,
            self.device
        )(**kwargs)
        return self

    def _update_attentions_hook(self, model, input, output):
        if self.tokenized_inputs is None:
            self.tokenized_inputs = input[0]
        else:
            self.tokenized_inputs = torch.cat( (self.tokenized_inputs, input[0], ))
        a = torch.stack(output[-1])
        self.attentions.append(a)

    def _find_pretrained(self, layer):
        for l in layer.children():
            if isinstance(l, transformers.PreTrainedModel):
                l.register_forward_hook(self._update_attentions_hook)
                return l
            return self._find_pretrained(l)
        raise ValueError("Model must contain a transformers pretrained language model.")

    def evaluate(self, X, Y):
        attentions = []
        with torch.no_grad():
            batcher = DocumentBatcher()
            dataset = DocumentDataset(X, Y)
            loader = DataLoader(dataset, batch_size=10, collate_fn=batcher)
            n = 0
            predictions = []
            print("Evaluating data")
            for batch in loader:
                x, _ = batch
                scores = self.nn(x)
                guesses = scores.argmax(dim=1)
                if self.label_encoder:
                    guesses = self.label_encoder.inverse_transform(guesses.cpu()).tolist()
                predictions += guesses
                n += loader.batch_size
                # TODO: fix bug in progress message
                logging.info(n, " / ", len(Y))
            print("\nDone.")
            self.attentions = torch.cat(self.attentions, dim=1).permute(1, 0, 2, 3, 4)
            self.predictions = predictions
            return Config(self.attentions, X, Y, predictions, self.tokenized_inputs, self.tokenizer, self.device)
