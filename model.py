
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
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
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
        self.heads = nn.ModuleList([Head(head_size, embedding_size, block_size, device) for _ in range(n_heads)])
        self.ffwds = FeedForward(embedding_size, dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear = nn.Linear(embedding_size, embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout =nn.Dropout(dropout)


    def forward(self, X):
        X = self.norm1(X)
        X_heads = X
        X_heads = X_heads + torch.concat([head(X) for head in self.heads], dim=-1)
        X = self.linear(X_heads)
        X = self.dropout(X)
        X = self.norm2(X)
        X = X + self.ffwds(X)
        return X


class gptModel(nn.Module):

    def __init__(self, vocab_size, batch_size, block_size, embedding_size, n_heads, n_multiheads, dropout, device):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.embedding_size = embedding_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding_table = nn.Embedding(block_size, embedding_size)
        self.heads = nn.Sequential(*[MultiHead(embedding_size, n_heads, block_size, dropout, device) for _ in range(n_multiheads)])
        self.head_lm = nn.Linear(embedding_size, vocab_size)

    def forward(self, X, Y=None):
        B, T = X.shape
        tok = self.token_embedding_table(X)  # B, T, C
        pos = self.positional_embedding_table(torch.arange(T, device=self.device))  # B, T, C
        X = tok + pos  # B, T, C
        X =  self.heads(X)
            
        X = self.head_lm(X)  # B, T, vocab_size

        return X

    def generate(self, X, max_len=100):
        # block = X if len(X) < block_size else X[-block_size:]
        # B = 1
        for _ in range(max_len):
            X_forward = X[:, -self.block_size:]
            logits = self.forward(X_forward)
            logits = logits[:, -1, :]  # B, (last_char), probabilities
            probs = F.softmax(logits, dim=-1) # B, probabilities (1, probabilities)
            next = torch.multinomial(probs, num_samples=1)
            X = torch.cat((X, next), dim=1)
        return X
