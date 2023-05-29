import os

import torch
from datasets import load_dataset
from model import gptModel
from my_utils import get_batch, estimate_loss, load_checkpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

dataset = load_dataset("tiny_shakespeare")

data = dataset['train']['text'][0]

d = sorted(list(set(data)))
chtoi = {chr: i for i, chr in enumerate(d)}
itoch = {i: chr for i, chr in enumerate(d)}

encode = lambda s: [chtoi[x] for x in s]
decode = lambda i: "".join(itoch[x] for x in i)

vocab_size = len(d)  # This is about 50 000 in GPT
large = False
if large:
    block_size = 32  # This is around 2000 in GPT
    batch_size = 4
    embedding_size = 128
    n_heads = 4
    data = torch.tensor(encode(data), device=device)
    dataset_size = len(data)
    lr = 3e-4
    eval_iters = 100
    epochs = 10000
    dropout = 0.1
    filename = 'model_large.pt'
else:
    block_size = 8  # This is around 2000 in GPT
    batch_size = 32
    embedding_size = 16
    n_heads = 4
    data = torch.tensor(encode(data), device=device)
    dataset_size = len(data)
    lr = 3e-4
    eval_iters = 200
    epochs = 10000
    dropout = 0.1
    filename = 'model.pt'
# ----------------


torch.manual_seed(123)

model = gptModel(vocab_size, batch_size, block_size, embedding_size, n_heads, dropout, device)
if device == 'gpu':
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
prev_loss = 999
filepath = os.getcwd()
filepath = os.path.join(filepath, filename)

model, optimizer, epoch, prev_loss = load_checkpoint(model, optimizer, prev_loss, filepath, device)

loss_est = estimate_loss(model, data, dataset_size)
print(loss_est)

for epoch in range(epochs):
    x, y = get_batch(data, device, dataset_size, block_size, batch_size)
    log, loss = model(x, y)

    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print("epoch", epoch)
        loss_est = estimate_loss(model, data, dataset_size)

        if loss_est['val'].item() < prev_loss:
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'loss_est': loss_est, }
            torch.save(state, filepath)
            prev_loss = loss_est['val'].item()
            print('new best loss found saving model.', loss_est)
        else:
            print('loss is not better than previous best', loss_est)

test = torch.tensor(encode(dataset['test']['text'][0][:block_size]))
test = test.reshape(1, test.shape[0])
test = test.to(device)
print(decode(model.generate(test, 2000)[0].tolist()))
