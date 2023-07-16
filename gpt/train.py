import os

import pandas as pd
import torch
from torch.nn import functional as F

from model import gptModel
from my_utils import get_batch, estimate_loss, load_checkpoint, estimate_accuracy
from my_utils import GptDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# dataset = load_dataset("tiny_shakespeare")
dataset = pd.read_csv('../data/tiny_shakespeare.csv')

data = dataset['train'][0]

d = sorted(list(set(data)))

chtoi = {chr: i for i, chr in enumerate(d)}
itoch = {i: chr for i, chr in enumerate(d)}

encode = lambda s: [chtoi[x] for x in s]
decode = lambda i: "".join(itoch[x] for x in i)



vocab_size = len(d)  # This is about 50 000 in GPT
large = True
if large:
    block_size = 256  # This is around 2000 in GPT
    batch_size = 1
    embedding_size = 384
    n_heads = 6
    n_multiheads = 6
    data = torch.tensor(encode(data), device=device)
    dataset_size = len(data)
    lr = 3e-4
    eval_iters = 200
    epochs = 100000
    dropout = 0.2
    filename = 'models/model_large.pt'
else:
    block_size = 8  # This is around 2000 in GPT
    batch_size = 2
    embedding_size = 16
    n_heads = 4
    n_multiheads = 1
    data = torch.tensor(encode(data), device=device)
    dataset_size = len(data)
    lr = 3e-4
    eval_iters = 200
    epochs = 50000
    dropout = 0.1
    filename = 'models/model_random.pt'
# ----------------


torch.manual_seed(123)

model = gptModel(vocab_size, batch_size, block_size, embedding_size, n_heads, n_multiheads, dropout, device)
if device == 'cuda':
    model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
prev_loss = 999
filepath = os.getcwd()
filepath = os.path.join(filepath, filename)
txt_filepath = os.path.join(os.getcwd(), '../data/tiny_shakespeare_train.txt')
gpt_dataset = GptDataset(txt_filepath, block_size, batch_size, d, device)
txt_filepath_test = os.path.join(os.getcwd(), '../data/tiny_shakespeare_test.txt')
gpt_dataset_test = GptDataset(txt_filepath_test, block_size, batch_size, d, device)


# model, optimizer, epoch, prev_loss = load_checkpoint(model, optimizer, prev_loss, filepath, device)

print(model)



loss_est = estimate_loss(model, data, dataset_size, gpt_dataset_test)
print(loss_est)


for epoch in range(0):
    x, y = get_batch(data, device, dataset_size, block_size, batch_size)

    logits = model(x)
    B, T, C = logits.shape
    logits = logits.view(B * T, C)

    y = y.view(B * T)
    
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print("epoch", epoch)
        loss_est = estimate_loss(model, data, dataset_size, gpt_dataset_test)
        if loss_est['random'].item() < prev_loss:
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'loss_est': loss_est, }
            torch.save(state, filepath)
            prev_loss = loss_est['random'].item()
            del state
            print('new best loss found saving model.', loss_est)
        else:
            print('loss is not better than previous best', loss_est)
        del loss_est
        print(torch.cuda.max_memory_allocated())
    del x, y, logits, loss, B, T, C
#test = torch.tensor(encode(dataset['test'][0][:block_size]))
#test = test.reshape(1, test.shape[0])
#test = test.to(device)
data = dataset['test'][0]
data = torch.tensor(encode(data), device=device)
print(data)
loss_est = estimate_accuracy(model, data, len(data), gpt_dataset_test)
print(loss_est)
test = torch.zeros((1, 1), dtype=torch.long, device=device)
with open('../data/output_text.txt', 'w') as file:
    file.write(decode(model.generate(test, 2000)[0].tolist()))
