import pandas as pd

dataset = pd.read_csv('../../../data/tiny_shakespeare.csv')

data = dataset['train'][0]

d = sorted(list(set(data)))
print(d)
print(len(d))
chtoi = {chr: i for i, chr in enumerate(d)}
itoch = {i: chr for i, chr in enumerate(d)}

encode = lambda s: [chtoi[x] for x in s]
decode = lambda i: "".join(itoch[x] for x in i)

for i in range(len(d)):
    print(i, decode([i]))
"""
folds = [i for i in range(14)]
for fold in folds:
    predictions = []
    ground_truth = []
    queries = open(f'./TINYSHAKESPEARE.fold.{fold}.queries.mapped.txt', 'r')
    queries_lines = queries.readlines()
    consequents = open(f'./TINYSHAKESPEARE.fold.{fold}.consequent.mapped.txt', 'r')
    consequent_lines = consequents.readlines()

    for consequent in consequent_lines:
        consequent = consequent.split(" ")
        consequent = [int(c) for c in consequent[:-1]]
        ground_truth.append(consequent)

    for query in queries_lines:
        query = query.split(" ")
        query = [int(c) for c in query[:-1]]
        predictions.append(query)

    for l, c in zip(predictions, ground_truth):
        # l = [int(i) for i in l]
        # c = [int(i) for i in c]
        print("new line")
        print(decode(l))
        print(decode(c))
"""