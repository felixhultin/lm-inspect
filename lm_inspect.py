import transformers
import torch

from typing import Union

from torch.utils.data import Dataset, DataLoader

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


class Inspector:

    def __init__(self, nn, X, Y, tokenizer):
        self.nn = nn
        self.lm = self._find_pretrained(nn)
        self.config = self.lm.config
        self.tokenizer = tokenizer
        self.X = X
        self.Y = Y
        # Values to be initialized in self.evaluate()
        self.tokenized_inputs = None
        self.attentions = []
        self.predictions  = self.evaluate()

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

    def evaluate(self):
        with torch.no_grad():
            batcher = DocumentBatcher()
            dataset = DocumentDataset(self.X, self.Y)
            loader = DataLoader(dataset, batch_size=10, collate_fn=batcher)
            n = 0
            predictions = []
            attentions = []
            for batch in loader:
                x, _ = batch
                predictions += self.nn(x)
                n += loader.batch_size
                print('\r' + str(n) + " / " + str(len(loader)), end='')
            self.attentions = torch.cat(self.attentions, dim=1).permute(1, 0, 2, 3, 4)
            return predictions

    def scope(self, attentions, layer: int = None, head : int = None,
              token_pos : Union[list, int] = None):
        if layer is not None:
            attentions = attentions[:, layer, :, :]
        if head is not None:
            attentions = attentions[:, :, head, :]
        if token_pos is not None:
            token_pos = [token_pos] if type(token_pos) is int else token_pos
            attentions = attentions[:, :, :, token_pos]
        return attentions

    def agg(self, top, display: str = "words", decode: bool = True, k: int = 1000):
        if display == "words":
            topk = top.sum(dim=0).topk(k)
        elif display == "positions":
            topk = top.sum(dim=1).topk(k)
        elif display == "words+positions":
            topk = top.topk(k)
        else:
            msg = ("display value " + display + " "
                   "is not a valid display value"
                   "Valid display values are: words, positions, words+positions")
            raise ValueError(msg)
        return self.tokenizer.decode(topk.indices)

    def agg_by_label(self, label, **kwargs):
        indices = [idx for idx, y in enumerate(self.Y) if y == label]
        attentions = self.scope(self.attentions[indices], **kwargs)
        top = torch.zeros(self.tokenized_inputs.shape[1], self.tokenizer.vocab_size)
        for idx, att in zip(indices, attentions):
            input_ids = self.tokenized_inputs[idx]
            n_layers, n_heads, max_size, _ = att.shape
            att_sum = att.sum(dim=0).sum(dim=0).sum(dim=0) / n_layers / n_heads / max_size
            top[list(range(max_size)), input_ids] += att_sum
        return self.agg(top, **kwargs)

    def agg_by_word(self, word, layer: int = None, head : int = None):
        indices = [idx for idx, y in self.Y if y == label]
        pass

    def agg_by_ngram(self, words, **kwargs):
        pass

    def tokenize_attentions():
        pass

    def agg_by_word(self, layer : int = None, head : int = None, top: int = 10):
        raise NotImplementedError

    # x = layer, head, position, token
