import json
import logging

import transformers
import torch

from typing import Dict, List, Union, Tuple

from torch.utils.data import Dataset, DataLoader

from .methods.topk import TopKMixin

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


class LanguageModelInspector(TopKMixin):

    def __init__(self, nn, X, Y, tokenizer, label_encoder = None):
        self.nn = nn
        self.lm = self._find_pretrained(nn)
        self.config = self.lm.config
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        # Values to reset in .evaluate()
        self.X = X
        self.Y = Y
        self.tokenized_inputs = None
        self.attentions = []
        self.predictions  = self.evaluate(self.X, self.Y)
        self.config = {}

    def __str__(self):

        pass


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
        self.X, self.Y = X, Y
        self.tokenized_inputs, self.attentions = None, []
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
            return predictions


    def _apply_config(self, **kwargs):
        filter_args = {k:v for k,v in kwargs.items() if k in ['label', 'words']}
        scope_args = {k:v for k,v in kwargs.items() if k in ['layer', 'head', 'token_pos']}
        context_args = {k:v for k,v in kwargs.items() if k in ['input_id']}

        attentions, params = self._apply_filter(self.attentions, **filter_args)
        if params.get('indices') and type(context_args.get('input_id')) == list:
            context_args['input_id'] = [context_args['input_id'][i] for i in params.get('indices')]

        attentions, _ = self._apply_scope(attentions, **scope_args)
        attentions, _ = self._apply_context(attentions, **context_args)

        return attentions, params


    def _apply_scope(self, attentions, layer: Union[list, int] = None, head : Union[list, int] = None,
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
        if layer is not None:
            layers = [layer] if type(layer) == int else layer
            attentions = attentions[:, layers, :, :]
        if head is not None:
            heads = [head] if type(head) == int else head
            attentions = attentions[:, :, heads, :]
        if token_pos is not None:
            token_positions = [token_pos] if type(token_pos) is int else token_pos
            attentions = attentions[:, :, :, token_positions]
        return attentions, {}

    def _apply_filter(self, attentions, label: Union[list, str] = None):
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
        if not label:
            return attentions
        labels = label if type(label) == list else [label]
        indices = [idx for idx, y in enumerate(self.Y) if y in labels]
        if not indices:
            msg = "No such label(s): " + str(label)
            raise ValueError(msg)
        attentions = attentions[indices, :, :, :]
        return attentions, {'indices': indices}

    def _apply_context(self, attentions, input_id: Union[List[int], int] = None):
        if input_id is not None:
            input_ids = [input_id] if type(input_id) == int else input_id
            n_samples = attentions.shape[0]
            attentions = attentions[list(range(n_samples)), :, :, input_ids, :]
            attentions = attentions.unsqueeze(3)
        return attentions, {}

    def todict(self, top):
        _, _, _, n_positions, _ = top.shape
        topk = (top.sum(dim=0).sum(dim=2) / n_positions).topk(5)
        indices, probs = topk.indices, topk.values
        output = {'indices': indices.tolist(), 'values': probs.tolist()}
        return output
