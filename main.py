import sys

import pandas
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch import nn

from transformers import AutoConfig, AutoTokenizer, AutoModel

from lm_inspect import LanguageModelInspector

class BERTEncoder(nn.Module):

    def __init__(self, model_name, device, sample_size):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name,
                                            output_hidden_states=True,
                                            output_attentions=True
                                            )
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.output_size = self.bert.config.hidden_size
        self.device = device

        # For logging purposes
        self.sample_idx = 0
        self.sample_size = sample_size

    def forward(self, documents):
        # Convert the encoded sequences to a tensor.
        docs = torch.tensor([d['doc'] for d in documents]).to(self.device)
        # Get the word representations of all sequences.
        bert_outputs = self.bert(docs)
        # Use the top layer of BERT
        top_layer_output = bert_outputs[0]
        # word_outputs will store the word representation of the ambigious word.
        word_outputs = torch.empty( len(documents), self.bert.config.hidden_size).to(self.device)
        #word_outputs = top_layer_output[:,0,:]
        for idx, d in enumerate(documents):
          pos = d['pos']
          word_outputs[idx] = top_layer_output[idx, pos]

        # For logging purposes, since this can take a long time.
        print('\r' + str(self.sample_idx) + " / " + str(self.sample_size), end='')
        sys.stdout.flush()
        self.sample_idx += len(documents)
        if self.sample_idx >= self.sample_size:
          self.sample_idx = 0

        return word_outputs

encoder = BERTEncoder('KB/bert-base-swedish-cased', 'cuda', 10000)

def context(doc, pos, window_size):

  new_pos = pos
  if pos < window_size:
    left = 0
  else:
    left = pos - window_size
    new_pos = window_size

  window_doc = doc[left: pos + window_size]

  # Sanity check that new_pos still refers to the ambigious word.
  assert doc[pos] == window_doc[new_pos]

  return {'doc': window_doc, 'pos': new_pos}

def read_data(filename, window_size):
  column_names = ['sense_key', 'lemma', 'pos', 'text']
  df = pandas.read_csv(filename, sep='\t', names=column_names)
  X = []
  Y = []
  for idx, row in df.iterrows():
    pos, text, sense_key = row['pos'], row['text'].split(), row['sense_key']
    c = context(text, pos, window_size)
    X.append(c)
    Y.append(sense_key)
  return X, Y

def bert_tokenize_and_encode(tokenizer, X, max_len):
  for x in X:
    x['doc'] = tokenizer.encode(x['doc'], max_length=max_len, pad_to_max_length=True, add_special_tokens=False, truncation=True)
  return X

Xval, Yval = read_data('../swedish_wsd/swedish_lexical_sample_GOLD_corpus.csv', 32)

seq = torch.nn.Sequential(
            encoder,
            torch.nn.Dropout(0.2),
            torch.nn.Linear(encoder.output_size, out_features=358)
        ).to('cuda')
state_dict = torch.load('models/KB-bert-swedish-cased-wsd.pt')
seq.load_state_dict(state_dict)

#_, Xval, _, Yval = train_test_split(X, Y, test_size=0.2, random_state=0)
#Xval, Yval = zip(*[ (x, y) for x, y in zip(Xval,Yval) if y == 'case%1:26:00::'])

config = AutoConfig.from_pretrained('KB/bert-base-swedish-cased',
                                    output_hidden_states=True,
                                    output_attentions=True
                                    )

tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased', config=config)
Xval = bert_tokenize_and_encode(tokenizer, Xval, 64)

# BOF HACK to get label encoder
_, Ytrain = read_data('../swedish_wsd/swedish_lexical_sample_TRAIN_corpus.csv', 32)
label_encoder = LabelEncoder()
label_encoder.fit(Ytrain)
# EOF HACK
inspector = LanguageModelInspector(seq, Xval, Yval, tokenizer, label_encoder)
input_ids = [x['pos'] for x in Xval]
results = inspector.topk_most_attended(k=5, label='betydelse_1_2', layer=[0,3, 6, 11], head=1, 
                                       input_id=input_ids)

def report(Y, predictions):
    report = {}
    for y, p in zip(Y, predictions):
        if y not in report:
            report[y] = {'correct': 0, 'false': 0}
        if y == p:
            report[y]['correct'] += 1
        else:
            report[y]['false'] += 1
    return report

"""
inspector.filter().scope().context().most_attended_to(k=3)


"""
