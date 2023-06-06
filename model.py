
import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):

    def __init__(self, vocab_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, 4 * vocab_size),
            nn.ReLU(),
            nn.Linear(vocab_size * 4, vocab_size),
            nn.Dropout(dropout),
        )

    def forward(self, X):
        return self.net(X)


class Head(nn.Module):

    def __init__(self, head_size, embedding_size, block_size, device):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False).to(device)
        self.query = nn.Linear(embedding_size, head_size, bias=False).to(device)
        self.value = nn.Linear(embedding_size, head_size, bias=False).to(device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, X):
        B, T, C = X.shape
        k = self.key(X)  # B, T, H
        q = self.query(X)  # B, T, H
        weights = q @ k.transpose(-2, -1) * C ** -0.5  # B, T, T
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # B, T, T
        weights = F.softmax(weights, dim=1)  # B, T, T

        v = self.value(X)  # B, T, C
        return weights @ v  # B, T, C


class MultiHead(nn.Module):

    def __init__(self, embedding_size, n_heads, block_size, dropout, device) -> None:
        super().__init__()
        head_size = int(embedding_size / n_heads)
        self.heads = nn.ModuleList([Head(head_size, embedding_size, block_size, device).to(device) for _ in range(n_heads)])
        self.ffwds = FeedForward(embedding_size, dropout)
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, X):
        X_heads = X
        X_heads = torch.concat([head(X) for head in self.heads], dim=-1)
        X = self.norm(X_heads)
        X += self.ffwds(X)
        X = self.norm(X_heads)
        return X


class gptModel(nn.Module):

    def __init__(self, vocab_size, batch_size, block_size, embedding_size, n_heads, dropout, device):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.embedding_size = embedding_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding_table = nn.Embedding(block_size, embedding_size)
        self.heads = nn.ModuleList([MultiHead(embedding_size, n_heads, block_size, dropout, device).to(device) for _ in range(4)])
        self.head_lm = nn.Linear(embedding_size, vocab_size)

    def forward(self, X, Y=None, device='cpu'):
        B, T = X.shape
        tok = self.token_embedding_table(X)  # B, T, C
        pos = self.positional_embedding_table(torch.arange(T, device=device))  # B, T, C
        X = tok + pos  # B, T, C
        for multi_head in self.heads:
            X = multi_head(X)  # B, T, C
        X = self.head_lm(X)  # B, T, vocab_size

        logits = X

        if Y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            Y = Y.view(B * T)
            loss = F.cross_entropy(logits, Y)

        return logits, loss

    def generate(self, X, max_len=100):
        # block = X if len(X) < block_size else X[-block_size:]
        # B = 1
        for _ in range(max_len):
            X_forward = X[:, -self.block_size:]
            logits, loss = self.forward(X_forward)
            logits = logits[:, -1, :]  # B, (last_char), probabilities
            probs = F.softmax(logits, dim=-1) # B, probabilities (1, probabilities)
            next = torch.multinomial(probs, num_samples=1)
            X = torch.cat((X, next), dim=1)
        return X
