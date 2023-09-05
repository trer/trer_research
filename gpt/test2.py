import os
import time as t

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from model import gptModel
from my_utils import get_batch, estimate_loss, load_checkpoint, GptDataset

TINYSHAKESPEARE = False
WEBTEXT = True


if TINYSHAKESPEARE:
    model_name = 'tinyShakespeare'
    query_file = '../data/tiny_shakespeare_queries.mapped.txt'
    consequents_file = '../data/tiny_shakespeare_consequents.mapped.txt'
    dataset = pd.read_csv('../data/tiny_shakespeare.csv')
    answers_file = None
    data = dataset['train'][0]
    d = sorted(list(set(data)))
else:
    query_file = '../data/webtext_small_queries.mapped.txt'
    consequents_file = '../data/webtext_small_consequents.mapped.txt'
    answers_file = '../data/WEBTEXTBIG.fold.0.answers.sbp.mapped.txt'
    answers_file = None
    model_name = 'webtextSubset'
    d = pd.read_csv('../data/alphabet.csv')['d']

ss = lambda i, c: (i, c) if i<=126 else (127, '¿')

chtoi = {chr:ss(i, chr)[0] for i, chr in enumerate(d)}
itoch = {i: ss(i, chr)[1] for i, chr in enumerate(d)}

encode = lambda s: [chtoi[x] for x in s]
decode = lambda i: "".join(itoch[x] for x in i)

d = sorted(list(set(decode(encode(d)))))

vocab_size = len(d)
print(vocab_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import pipeline, set_seed
model = pipeline('text-generation', model='gpt2-xl')
set_seed(42)

#if device == 'cuda':
#    model.cuda()



def to_str(X):
    tmp = [str(i) for i in X]
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
    #queries = open(f'../data/tiny_shakespeare_queries.mapped.txt', 'r')
    queries_lines = queries.readlines()
    consequents = open(consequents_file, 'r')
    #consequents = open(f'../data/tiny_shakespeare_consequents.mapped.txt', 'r')
    consequent_lines = consequents.readlines()
    if answers_file is not None:
        answers = open(answers_file, 'r')
        answers_lines = answers.readlines()
        answers_lines = iter(answers_lines)
    
    wrong = 0
    total = 0
    last = 0
    for query, consequent in zip(queries_lines, consequent_lines):
        if answers_file is not None:
            try:
                answer = next(answers_lines)
                out = to_int(answer)[0]
            except ValueError:
                continue
            except StopIteration:
                break
        else:
            
            
            query = decode(to_int(query))
            #consequent = consequent.split(" ")
            consequent = decode(to_int(consequent))
            t1 = t.time()
            out = model(query, max_new_tokens=100, num_return_sequences=1)
            t2 = t.time()
            
            out = out[0]['generated_text']
            
            out = out[len(query)]
            
            times.append(t2-t1)
        
        if out == -1:
            missed = missed + 1
            break
        elif consequent[1] == out:
            last += 1
        total = total + 1
    if not times:
        times = [0]
    print(f"Inference time: max: {max(times)}, min: {min(times)}, avg: {np.mean(times)}")
    print(f"correct: {last}")
    print(f'total: {total}')
    print(f"acc, {last / total}")

