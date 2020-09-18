import torch

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
        attentions = self.apply_config(**kwargs)
        n_samples, n_layers, n_heads, max_size, _ = attentions.shape
        import pdb
        pdb.set_trace()
        # Store attentions to attentions.
        top = torch.zeros( n_samples, n_layers, n_heads, max_size, self.tokenizer.vocab_size )
        indices = kwargs.get('indices', list(range(len(self.tokenized_inputs))))
        for n in range(len(self.tokenized_inputs[indices])):
            token_ids = self.tokenized_inputs[indices][n]
            top.index_add_(4, token_ids, attentions)
        topk_kwargs = { k:v for k,v in kwargs.items() if k in {'display', 'decode', 'k', 'return_json'} }
        return top

    def _visualize():
        pass
