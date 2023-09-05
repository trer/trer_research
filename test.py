import os
import time as t

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from gpt.model import gptModel
from gpt.my_utils import get_batch, estimate_loss, load_checkpoint, GptDataset

TINYSHAKESPEARE = False
WEBTEXT = True


if TINYSHAKESPEARE:
    model_name = 'tinyShakespeare'
    query_file = 'data/tiny_shakespeare_queries.mapped.txt'
    consequents_file = 'data/tiny_shakespeare_consequents.mapped.txt'
    dataset = pd.read_csv('data/tiny_shakespeare.csv')
    answers_file = None
    data = dataset['train'][0]
    d = sorted(list(set(data)))
else:
    query_file = 'data/webtext_queries.mapped.txt'
    consequents_file = 'data/webtext_consequents.mapped.txt'
    answers_file = 'data/WEBTEXTBIG.fold.0.answers.sbp.mapped.txt'
    model_name = 'webtext'
    d = pd.read_csv('data/alphabet.csv')['d']

ss = lambda i, c: (i, c) if i<=126 else (127, '¿')

chtoi = {chr:ss(i, chr)[0] for i, chr in enumerate(d)}
itoch = {i: ss(i, chr)[1] for i, chr in enumerate(d)}

encode = lambda s: [chtoi[x] for x in s]
decode = lambda i: "".join(itoch[x] for x in i)

d = sorted(list(set(decode(encode(d)))))

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
    filename = f'models/{model_name}model_large.pt'
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
    filename = 'gpt/models/model_random.pt'
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


def to_str(X):
    tmp = [str(i.item()) for i in X]
    tmp = " ".join(tmp)
    return tmp + '\n'


def to_int(query):
    query = query.rsplit()
    return [int(c) for c in query[:]]


folds = [i for i in range(1)]
for fold in folds:
    predictions = []
    ground_truth = []
    times = []
    queries = open(query_file, 'r')
    queries_lines = queries.readlines()
    consequents = open(consequents_file, 'r')
    consequent_lines = consequents.readlines()
    if answers_file is not None:
        answers = open(answers_file, 'r')
        answers_lines = answers.readlines()
        answers_lines = iter(answers_lines)

    correct = 0
    wrong = 0
    total = 0
    t_0_correct = 0
    for query, consequent in zip(queries_lines, consequent_lines):

        query = to_int(query)
        consequent = to_int(consequent)
        if answers_file is not None:
            try:
                answer = next(answers_lines)
                out = to_int(answer)[0]
            except ValueError:
                continue
            except StopIteration:
                break
        else:
            tmp = np.zeros(block_size)
            if len(query) >= len(tmp):
                tmp = query[-len(tmp):]
    
            else:
                tmp[-len(query):] = query
            tmp = torch.tensor(tmp, dtype=int, device=device).view((1, len(tmp)))
            t1 = t.time()
            out = model(tmp)
            t2 = t.time()

            out = torch.argmax(out[:, -1, :], -1)
            
            times.append(t2-t1)
        
        print(out, consequent)
        if out == -1:
            missed = missed + 1
            break
        elif consequent[-1] == out:
            correct = correct + 1
        elif consequent[-2] == out:
            t_0_correct = t_0_correct + 1
        total = total + 1
    if not times:
        times = [0]
    print(f"Inference time: max: {max(times)}, min: {min(times)}, avg: {np.mean(times)}")
    print(f"correct: {correct}")
    print(f'T_0 correct: {t_0_correct}')
    print(f'total: {total}')
    print(f"acc, {correct / total}")

