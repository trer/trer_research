import json
import os

import numpy as np
import pandas as pd
import torch

from gpt.model import gptModel
from gpt.my_utils import load_checkpoint, get_batch, GptDataset

block_size = 256  # This is around 2000 in GPT
batch_size = 1


WEBTEXT = True
TINYSHAKESPEARE = False

training = True
test = False

train_filepath = 'data/webtext_training.mapped.txt'
query_filepath = 'data/webtext_small_queries.mapped.txt'
consequent_filepath = 'data/webtext_small_consequents.mapped.txt'
# train_filepath = 'data/tiny_shakespeare_training.mapped.txt'
# query_filepath = 'data/tiny_shakespeare_queries.mapped.txt'
# consequent_filepath = 'data/tiny_shakespeare_consequents.mapped.txt'

if WEBTEXT:

    txt_filepath = os.path.join(os.getcwd(), 'external/gpt-2-output-dataset/data/webtext.train.jsonl')
    data = []
    with open(txt_filepath, 'r') as f:
        for i, line in enumerate(f):
            data.append(json.loads(line)['text'])

    txt_filepath_test = os.path.join(os.getcwd(), 'external/gpt-2-output-dataset/data/webtext.test.jsonl')
    data_test = []
    with open(txt_filepath_test, 'r') as f:
        for i, line in enumerate(f):
            data_test.append(json.loads(line)['text'])

    d = set()
    for line in data:
        for c in line:
            d.add(c)

    for line in data_test:
        for c in line:
            d.add(c)
    d = sorted(list(d))
else:

    dataset = pd.read_csv('data/tiny_shakespeare.csv')
    data = dataset['train'][0]
    data_test = dataset['test'][0]
    data_test_size = len(data_test)
    d = set()
    for c in data:
        d.add(c)

    for c in data_test:
        d.add(c)
    d = sorted(list(d))
ss = lambda i, c: (i, c) if i<=126 else (127, '¿')

chtoi = {chr:ss(i, chr)[0] for i, chr in enumerate(d)}
itoch = {i: ss(i, chr)[1] for i, chr in enumerate(d)}


encode = lambda s: [chtoi[x] for x in s]
decode = lambda i: "".join(itoch[x] for x in i)

d = sorted(list(set(decode(encode(d)))))

vocab_size = len(d)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
large = True


def to_str(X):
    tmp = [str(i.item()) for i in X]
    tmp = " ".join(tmp)
    return tmp + '\n'


def to_int(query):
    query = query.split(" ")
    return [int(c) for c in query[:]]

if WEBTEXT:
    gpt_dataset_test = GptDataset(txt_filepath_test, block_size, batch_size, encode, decode, device)
else:
    data = torch.tensor(encode(data), dtype=torch.long)
    data_test = torch.tensor(encode(data_test), dtype=torch.long)

if training:
    max_int = 0
    for i in range(len(data)):
        max_int = max(max_int, max(encode(data[i])))

    with open(train_filepath, 'w') as f0:
        f0.seek(0)
        txt = str(max_int) + '\n'
        f0.write(txt)
        j = 0
        max_size = 2600000000
        for i in range(len(data)):
            if j >= max_size:
                print("breaking on max_size")
                break
            encoded = encode(data[i])
            j += 4*len(encoded)
            txt = [str(i) for i in encoded]
            txt = " ".join(txt) + " "
            f0.write(txt)
if test:
    with open(query_filepath, 'w') as f1:
        with open(consequent_filepath, 'w') as f2:
            for i in range(100):
                if WEBTEXT:
                    X, Y = next(gpt_dataset_test)
                else:
                    X, Y = get_batch(data_test, device, data_test_size, block_size, batch_size)
            
                X = to_str(X[0])
                Y = to_str(Y[0][-2:])

                f1.write(X)
                f2.write(Y)



