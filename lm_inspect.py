import transformers
import torch

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

    def __init__(self, nn, lm, X, Y, tokenizer):
        self.nn = nn
        #self.lm = lm
        self.lm = self._find_pretrained(nn)
        self.config = lm.config
        self.X = X
        self.Y = Y
        self.attentions = []
        self.predictions  = self.evaluate()

    def _add_attentions_hook(self, model, input, output):
        a = torch.stack(output[-1])
        self.attentions.append(a)

    def _find_pretrained(self, layer):
        for l in layer.children():
            if isinstance(l, transformers.PreTrainedModel):
                l.register_forward_hook(self._add_attentions_hook)
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
                #a = self.lm(x)
                #attentions.append(torch.stack(a))
                n += loader.batch_size
                print('\r' + str(n) + " / " + str(len(loader)), end='')
            return predictions

    def layer_head(self, layer: int = None, head : int = None):
        attentions = self.attentions
        if layer is not None:
            attentions = attentions[:, layer, :, :]
        if head is not None:
            attentions = attentions[:, :, head, :]
        return attentions

    def agg_by_label(self, layer: int = None, head : int = None, top : int = 10):
        raise NotImplementedError

    def agg_by_word(self, layer : int = None, head : int = None, top: int = 10):
        raise NotImplementedError
