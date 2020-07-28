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

    def scope(self, attentions, layer: Union[list, int] = None, head : Union[list, int] = None,
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
        return attentions

    def agg(self, top, display: str = "words", decode: bool = True, k: int = 1000):
        if display == "words":
            topk = top.sum(dim=0).topk(k)
        elif display == "positions":
            topk = top.sum(dim=1).topk(k)
        elif display == "words+positions":
            topk = top.topk(k)
        else:
            msg = (str(display) +
                   "is not a valid display value"
                   "Valid display values are: words, positions, words+positions")
            raise ValueError(msg)
        return self.tokenizer.convert_ids_to_tokens(topk.indices)

    def by_label(self, label: Union[list, str], **kwargs):
        """Top word, position or word+positions by one or many label(s).

        Parameters
        ----------
        label: Union[list, str]
            Unique label of the evaluation data Y.

        Raises
        ------
        ValueError
            If label is not in the evaluation data Y.
        """
        labels = label if type(label) == list else [label]
        indices = [idx for idx, y in enumerate(self.Y) if y in labels]
        if not indices:
            msg = "No such label(s): " + str(label)
            raise ValueError(msg)
        layer, head, token_pos = kwargs.get('layer'), kwargs.get('head'), kwargs.get('token_pos')
        attentions = self.scope(self.attentions[indices], layer, head, token_pos)
        top = torch.zeros(self.tokenized_inputs.shape[1], self.tokenizer.vocab_size)
        for idx, att in zip(indices, attentions):
            input_ids = self.tokenized_inputs[idx]
            n_layers, n_heads, max_size, _ = att.shape
            att_sum = att.sum(dim=0).sum(dim=0).sum(dim=0) / n_layers / n_heads / max_size
            top[list(range(max_size)), input_ids] += att_sum
        display, decode, k = kwargs.get('display', 'words'), kwargs.get('decode'), kwargs.get('k')
        return self.agg(top, display, decode, k)

    def by_word(self, word, layer: int = None, head : int = None):
        indices = [idx for idx, y in self.Y if y == label]
        pass

    def by_ngram(self, words, **kwargs):
        pass
