import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class GptDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dataset_path, block_size, batch_size, d, device):
        'Initialization'
        self.block_size = block_size
        self.batch_size = batch_size
        self.id = 0
        self.device = device
        self.j = 0
        self.mask = torch.zeros((batch_size, block_size))
        chtoi = {chr: i for i, chr in enumerate(d)}
        itoch = {i: chr for i, chr in enumerate(d)}

        self.encode = lambda s: [chtoi[x] for x in s]
        self.decode = lambda i: "".join(itoch[x] for x in i)

        data = open(dataset_path, 'r')
        data = data.readlines()
        data_2 = [0 for _ in range(len(data))]
        for i in range(len(data_2)):
            data_2[i] = self.encode(data[i])
        new_data = []
        for line in data_2:
            line.append(self.encode('\n'))
            for i in range(2, len(line)):
                tmp = np.zeros(block_size + 1)
                if len(line[:i]) >= len(tmp):
                    tmp = line[:i][-len(tmp):]

                else:
                    tmp[-len(line[:i]):] = line[:i]
                new_data.append(tmp)
        self.dataset = new_data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.dataset[index][:-1]
        y = self.dataset[index][1:]
        return torch.tensor(X, dtype=int).to(self.device), torch.tensor(y, dtype=int).to(self.device)

    def __next__(self):
        X, Y = zip(*[self.__getitem__(self.id + i) for i in range(self.batch_size)])
        X, Y = torch.stack(X), torch.stack(Y)
        self.id = self.id + self.batch_size
        if self.id >= self.__len__():
            self.id - self.__len__()
        return X, Y





def load_checkpoint(model, optimizer, losslogger, filename, device):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model.device = device
        print('model device is set to', model.device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['loss_est']['random'].item()
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger


def get_batch(data, device, dataset_size, block_size, batch_size):
    idx = torch.randint(dataset_size - block_size, (batch_size,))
    x = torch.stack([data[i: (i + block_size)] for i in idx])
    y = torch.stack([data[i + 1: (i + block_size + 1)] for i in idx])
    idx = torch.randint(block_size * 2, (batch_size,))
    zero = torch.zeros(x.shape)
    for i in range(batch_size):
        x[i, :-idx[i]] = zero[i, :-idx[i]]
    x, y = x.to(device), y.to(device)
    del idx
    return x, y


def get_train(data, device, dataset_size, block_size, batch_size):
    idx = torch.randint(dataset_size - block_size, (batch_size,))
    x = torch.stack([data[i: (i + block_size)] for i in idx])
    y = torch.stack([data[i + 1: (i + block_size + 1)] for i in idx])
    idx = torch.randint(1, block_size * 2, (batch_size,))
    zero = torch.zeros(x.shape)
    for i in range(batch_size):
        x[i, :-idx[i]] = zero[i, :-idx[i]]
        y[i, : -idx[i]] = zero[i, :-idx[i]]
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, dataset_size, dataset_2, eval_iters=100):
    out = {}
    model.eval()
    for split in ['random', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'random':
                X, Y = get_batch(data, model.device, dataset_size, model.block_size, model.batch_size)
            else:
                X, Y = next(dataset_2)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            Y = Y.view(B * T)
            loss = F.cross_entropy(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    del losses, X, Y, logits, loss, T, B, C
    return out


@torch.no_grad()
def estimate_accuracy(model, data, dataset_size, dataset_2, eval_iters=100):
    out = {}

    model.eval()
    for split in ['random', 'test']:
        out[split] = {}
        out[split] = {}
        out[split]['correct'] = 0
        out[split]['wrong'] = 0
        out[split]['total'] = 0
        out[split]['last'] = 0
        out[split]['last_total'] = 0
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'random':
                X, Y = get_batch(data, model.device, dataset_size, model.block_size, 1)
            else:
                X, Y = next(dataset_2)
            # print(X.shape)
            logits = model(X)
            B, T, C = logits.shape

            bigidx_X = torch.argmax(logits, -1)
            bigidx_X = bigidx_X.view(B * T)
            # print(logits)
            # print("big indexsss", bigidx_X)
            Y = Y.view(B * T)

            out[split]['correct'] = out[split]['correct'] + torch.sum((bigidx_X == Y))
            out[split]['wrong'] = out[split]['wrong'] + torch.sum((bigidx_X != Y))
            out[split]['total'] = out[split]['total'] + len(Y)
            out[split]['last'] = out[split]['last'] + torch.sum((bigidx_X[-1] == Y[-1]))
            out[split]['last_total'] += 1


    model.train()
    for split in ['random', 'test']:
        print('test')
        print(out[split]['correct'])
        print(out[split]['wrong'])
        print(out[split]['total'])
        print(f'accuracy = {out[split]["correct"] / out[split]["total"]}')
        print(f'accuracy = {out[split]["last"] / out[split]["last_total"]}')
    del losses, X, Y, logits, T, B, C
    return out
