import pandas as pd
from datasets import load_dataset

dataset = load_dataset("tiny_shakespeare")
data = dataset['train'][0]['text']
print(type(data))
with open('C:/Users/tor-d/git/trer_research/data/tiny_shakespeare_train.txt', 'w') as f:
    f.write(data)

data = dataset['test'][0]['text']
print(type(data))
with open('C:/Users/tor-d/git/trer_research/data/tiny_shakespeare_test.txt', 'w') as f:
    f.write(data)

"C:/Users/tor-d/git/trer_research/external/sdsl-lite"
"""
data_1 = data['train'][0]
data_2 = data['test'][0]
d = sorted(list(set(data_1)))
chtoi = {chr: i for i, chr in enumerate(d)}
itoch = {i: chr for i, chr in enumerate(d)}

encode = lambda s: [chtoi[x] for x in s]
decode = lambda i: "".join(itoch[x] for x in i)

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
