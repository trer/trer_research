import json
import os

import numpy as np
import pandas as pd
import torch

from model import gptModel
from my_utils import load_checkpoint, get_batch, GptDataset

block_size = 256  # This is around 2000 in GPT
batch_size = 1

print("reading data\n")


training = True
if training:
    max_int = 0
    with open('../data/webtext_training.mapped.txt', 'r') as f0:
        print("trying to print line")
        txt = f0.read(1000)
        print(txt)



txt_filepath = os.path.join(os.getcwd(), '../external/gpt-2-output-dataset/data/webtext.train.jsonl')
data = []
with open(txt_filepath, 'r') as f:
    for i, line in enumerate(f):
        data.append(json.loads(line)['text'])

txt_filepath_test = os.path.join(os.getcwd(), '../external/gpt-2-output-dataset/data/webtext.test.jsonl')
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

gpt_dataset_test = GptDataset(txt_filepath_test, block_size, batch_size, encode, decode, device)

training = False
if training:
    max_int = 0
    with open('../data/webtext_training.mapped.txt', 'w') as f0:
        
        for i in range(len(data)):
            encoded = encode(data[i])
            max_int = max(max_int, max(encoded))
            encoded = [str(i) for i in encoded]
            txt = " ".join(encoded) + " "
            f0.write(txt)
    
    with open('../data/webtext_training.mapped.txt', 'a') as f0:
        f0.seek(0)
        txt = str(max_int) + '\n'
        f0.write(txt)
        

else:
    with open('../data/webtext_queries.mapped.txt', 'w') as f1:
        with open('../data/webtext_consequents.mapped.txt', 'w') as f2:
            for i in range(10000):
                X, Y = next(gpt_dataset_test)
            
                X = to_str(X[0])
                Y = to_str(Y[0][-2:])

                f1.write(X)
                f2.write(Y)

"""

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
    filename = 'models/model_random.pt'
# ----------------


model = gptModel(vocab_size, batch_size, block_size, embedding_size, n_heads, n_multiheads, dropout, device)
if device == 'cuda':
    model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
prev_loss = 999
filepath = os.getcwd()
filepath = os.path.join(filepath, filename)

model, optimizer, epoch, prev_loss = load_checkpoint(model, optimizer, prev_loss, filepath, device)
model.eval()







b_correct = 0
b_total = 0
a_correct = 0
a_total = 0


            out = model(X)


            out_index = torch.argmax(out[:, -1, :], -1)
            out2 = torch.argmax(out, -1)

            if out_index.item() == Y[0][-1].item():
                b_correct += 1

            b_total += 1

            X = to_str(X[0])
            Y = to_str(Y[0][-2:])

            query = to_int(X)
            Y_tmp = to_int(Y)

            tmp = np.zeros(block_size)
            if len(query) >= len(tmp):
                tmp = query[-len(tmp):]

            else:
                tmp[-len(query):] = query
            tmp = torch.tensor(tmp, dtype=int, device=device).view((1, len(tmp)))

            out2 = model(tmp)
            out2 = torch.argmax(out2[:, -1, :], -1)

            if out2.item() == Y_tmp[-1]:
                a_correct += 1
            a_total += 1

            print("b_correct", b_correct, b_correct/b_total)
            print("a_correct", a_correct, a_correct/a_total)
            
            
            
      
print(f"final before: \n b_correct: {b_correct}, b_total: {b_total}, acc: {b_correct / b_total}")
print(f"final after: \n a_correct: {a_correct}, a_total: {a_total}, acc: {a_correct / a_total}")      
"C:/Users/tor-d/git/trer_research/external/sdsl-lite"

data_1 = data['train'][0]
data_2 = data['test'][0]
d = sorted(list(set(data_1)))
chtoi = {chr: i for i, chr in enumerate(d)}
itoch = {i: chr for i, chr in enumerate(d)}

encode = lambda s: [chtoi[x] for x in s]
decode = lambda i: "".join(itoch[x] for x in i)
|
data_t_int = encode(data_1)
data_test_int = encode(data['test'][0])

print(type(data_1), len(data_1))
print(data_1.find('\\n'))
print('\\n')
print(data_1[:1000])
data_1 = data_1.replace('\\n', '\\r\\n')
data_2 = data_2.replace('\\n', '\\r\\n')

print(data_1.find('\\n'))





with open('../external/SUBSEQ/IPredict/datasets/tiny_shakespeare.txt', 'a') as f:
    f.write(data_2)

"""
