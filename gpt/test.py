import os

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from model import gptModel
from my_utils import get_batch, estimate_loss, load_checkpoint

dataset = pd.read_csv('../data/tiny_shakespeare.csv')

data = dataset['train'][0]

d = sorted(list(set(data)))
chtoi = {chr: i for i, chr in enumerate(d)}
itoch = {i: chr for i, chr in enumerate(d)}

encode = lambda s: [chtoi[x] for x in s]
decode = lambda i: "".join(itoch[x] for x in i)


vocab_size = len(d)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
large = True

if large:
    block_size = 256  # This is around 2000 in GPT
    batch_size = 64
    embedding_size = 384
    n_heads = 6
    n_multiheads = 6
    lr = 3e-4
    eval_iters = 200
    epochs = 10000
    dropout = 0.2
    filename = 'models/model_large.pt'
else:
    block_size = 8  # This is around 2000 in GPT
    batch_size = 1
    embedding_size = 16
    n_heads = 4
    n_multiheads = 1
    lr = 3e-4
    eval_iters = 200
    epochs = 10000
    dropout = 0.1
    filename = 'models/model.pt'
# ----------------


model = gptModel(vocab_size, batch_size, block_size, embedding_size, n_heads, n_multiheads, dropout, device)
if device == 'cuda':
    model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
prev_loss = 999
filepath = os.getcwd()
filepath = os.path.join(filepath, filename)

model, optimizer, epoch, prev_loss = load_checkpoint(model, optimizer, prev_loss, filepath, device)
if device == 'cuda':
    model.cuda()


model.eval()
folds = [i for i in range(14)]
for fold in folds:
    predictions = []
    ground_truth = []
    queries = open(f'../external/SUBSEQ/IPredict/outputs/TINYSHAKESPEARE.fold.{fold}.queries.mapped.txt', 'r')
    queries_lines = queries.readlines()
    consequents = open(f'../external/SUBSEQ/IPredict/outputs/TINYSHAKESPEARE.fold.{fold}.consequent.mapped.txt', 'r')
    consequent_lines = consequents.readlines()

    dataset = pd.read_csv('../data/tiny_shakespeare.csv')
    data = dataset['test'][0]
    data = torch.tensor(encode(data), device=device)

    for query in queries_lines:
        query = query.split(" ")
        query = [int(c) for c in query[:-1]]
        tmp = np.zeros(block_size)
        if len(query) >= len(tmp):
            tmp = query[-len(tmp):]
        else:
            tmp[-len(query):] = query
        tmp = torch.tensor(tmp, dtype=int, device=device).view((1, len(tmp)))
        out = model(tmp)

        out = torch.argmax(out[:, -1, :], -1)
        predictions.append(out)


    for consequent in consequent_lines:
        consequent = consequent.split(" ")
        consequent = [int(c) for c in consequent[:-1]]
        ground_truth.append(consequent)

    if len(ground_truth) != len(predictions):
        print("sumthing wong")
        print(len(ground_truth), len(predictions))

    correct = 0
    wrong = 0
    total = 0
    missed = 0
    for i in range(len(ground_truth)):
        for j in range(len(ground_truth[i])):
            if predictions[i] == -1:
                missed = missed + 1
                break
            elif ground_truth[i][j] == predictions[i]:
                correct = correct + 1
            total = total + 1
    print(f"correct: {correct}")
    print(f"acc, {correct / total}")
