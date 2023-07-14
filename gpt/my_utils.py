import os

import numpy as np
import torch
from torch.nn import functional as F


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
        losslogger = checkpoint['loss_est']['val'].item()
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger


def get_batch(data, device, dataset_size, block_size, batch_size):
    idx = torch.randint(dataset_size - block_size, (batch_size,))
    x = torch.stack([data[i: (i + block_size)] for i in idx])
    y = torch.stack([data[i + 1: (i + block_size + 1)] for i in idx])
    idx = torch.randint(block_size*2, (batch_size,))
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
def estimate_loss(model, data, dataset_size, eval_iters=100):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, model.device, dataset_size, model.block_size, model.batch_size)
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
def estimate_accuracy(model, data, dataset_size, eval_iters=100):
    out = {}
    out['correct'] = 0
    out['wrong'] = 0
    out['total'] = 0


    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, model.device, dataset_size, model.block_size, 1)

            logits = model(X)
            B, T, C = logits.shape

            bigidx_X = torch.argmax(logits, -1)
            logits = logits.view(B * T, C)
            bigidx_X = bigidx_X.view(B * T)
            Y = Y.view(B * T)

            out['correct'] = out['correct'] + torch.sum((bigidx_X==Y))
            out['wrong'] = out['wrong'] + torch.sum((bigidx_X!=Y))
            out['total'] = out['total'] + len(Y)


    model.train()
    print(out['correct'])
    print(out['wrong'])
    print(out['total'])
    print(f'accuracy = {out["correct"] / out["total"]}')
    del losses, X, Y, logits, T, B, C
    return out