from .config import Configuration

def topk_most_attended(config: Configuration, k: int = 5):
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
    top = torch.zeros( n_samples, n_layers, n_heads, max_size, self.tokenizer.vocab_size ).to(self.device)
    indices = params.get('indices', list(range(len(self.tokenized_inputs))))
    for idx, i in enumerate(indices):
        token_ids = self.tokenized_inputs[i]
        top[idx].index_add_(3, token_ids, attentions[idx])
    if kwargs.get('visualize'): # refactor this part.
        d = self.todict(top, k = k)
        id_to_token = self._flatten_and_decode(d)
        visualize(d, id_to_token)
    topk_kwargs = { k:v for k,v in kwargs.items() if k in {'display', 'decode', 'k', 'return_json'} }
    return top

"""

inspector = Inspector(Xval, Yval)


"""
