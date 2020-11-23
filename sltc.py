import torch

from transformers import AutoConfig, AutoModel, AutoTokenizer

from lm_inspect import LanguageModelInspector
from word_sense_disambiguation import bert_encoder, label_encoder, Xval as Xtest, Yval as Ytest

# Load trained classifier from binary file.
seq = torch.nn.Sequential(
            bert_encoder,
            torch.nn.Dropout(0.2),
            torch.nn.Linear(bert_encoder.output_size, out_features=358)
        ).to('cpu')
state_dict = torch.load('models/KB-bert-swedish-cased-wsd.pt', map_location=torch.device('cpu'))
seq.load_state_dict(state_dict)

# Load KB Bert base tokenizer
config = AutoConfig.from_pretrained('KB/bert-base-swedish-cased',
                                    output_hidden_states=True,
                                    output_attentions=True
                                    )
tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased', config=config)

# Get the positions of the ambigious words
input_ids = [x['pos'] for x in Xtest]

inspector = LanguageModelInspector(seq, Xtest, Ytest, tokenizer, label_encoder, device = 'cpu')
inspector.configure(label='följa_1_3_a', layer=[0,3, 6, 11], head=1, input_id=input_ids)
results = inspector.topk_most_attended_to(k=5, return_type="all", visualize=True)
