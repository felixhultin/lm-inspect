import torch

from itertools import chain

from lm_inspect.visualize_view import visualize

class TopKMixin:

    def topk_most_attended(self, k: int = 5, **kwargs):
        """Top word, position or word+positions.

        Parameters
        ----------
        label: Union[list, str]
            Unique label(s) of the evaluation data Y.

        Raises
        ------
        ValueError
            If label is not in the evaluation data Y.
        """
        attentions, params = self._apply_config(**kwargs)
        n_samples, n_layers, n_heads, max_size, _ = attentions.shape
        # Store attentions to attentions.
        top = torch.zeros( n_samples, n_layers, n_heads, max_size, self.tokenizer.vocab_size ).to('cuda')
        indices = params.get('indices', list(range(len(self.tokenized_inputs))))
        for idx, i in enumerate(indices):
            token_ids = self.tokenized_inputs[i]
            top[idx].index_add_(3, token_ids, attentions[idx])
        if kwargs.get('visualize'): # refactor this part.
            d = self.todict(top, k = k)
            id_to_token = {i:self.tokenizer.decode(i).replace(" ", "") for i in list(chain(*chain(*d['indices'])))}
            visualize(d, id_to_token)
        topk_kwargs = { k:v for k,v in kwargs.items() if k in {'display', 'decode', 'k', 'return_json'} }
        return top
