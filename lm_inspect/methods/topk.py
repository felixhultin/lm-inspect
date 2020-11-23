import json

import torch

from itertools import chain

from lm_inspect.visualize_view import visualize as run_visualize
from lm_inspect.configuration import Config


class TopkMostAttendedTo:

    def inspect(config : Config, k: int = 5, return_type: str = 'all', visualize : bool = False):
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
        n_samples, n_layers, n_heads, max_size, _ = config.attentions.shape
        results = torch.zeros( n_samples, n_layers, n_heads, max_size, config.tokenizer.vocab_size ).to(config.device)
        indices = list(range(len(config.tokenized_inputs)))
        for idx, i in enumerate(indices):
            token_ids = config.tokenized_inputs[i]
            results[idx].index_add_(3, token_ids, config.attentions[idx])

        n_samples, n_layers, n_heads, n_positions, _ = results.shape
        if return_type == 'scope':
            topk_scope = (results.sum(dim=0).sum(dim=2) / n_samples / n_positions).topk(k)
            indices_scope, probs_scope = topk_scope.indices, topk_scope.values
            output = {'indices': indices_scope.tolist(), 'values': probs_scope.tolist()}
            output['config'] = config.config
        else:
            topk_agg = results.sum(dim=0).sum(dim=0).sum(dim=0).sum(dim=0).topk(k)
            indices_agg, probs_agg = topk_agg.indices, topk_agg.values
            probs_agg = probs_agg / n_samples/ n_layers / n_heads / n_positions
            output = {'indices': indices_agg.tolist(), 'values': probs_agg.tolist()}
            if not visualize:
                output['indices'] = [config.tokenizer.decode(i).replace(" ", "") for i in output['indices']]
                output = [ (i, p) for i, p in zip(output['indices'], output['values'])]

        if visualize:
            TopkMostAttendedTo.visualize(output, config.tokenizer, return_type)

        return output

    def visualize(output, tokenizer, return_type : str):
        response = {}
        if return_type == 'all':
            response['agg'] = output
            id_to_token = {i:tokenizer.decode(i).replace(" ", "") for i in output['indices']}
        else:
            response['all'] = output
            id_to_token = {i:tokenizer.decode(i).replace(" ", "") for i in list(chain(*chain(*output['indices'])))}

        run_visualize( response, id_to_token )
