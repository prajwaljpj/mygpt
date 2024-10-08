#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import functional as F, init

# initial parameters
torch.manual_seed(1337)
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Get data
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)


# create the tokenizer
def tokenizer(tokenizer_type):
    if tokenizer_type == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        encode = enc.encode
        decode = enc.decode
    else:
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    return encode, decode  # returns the encode and decode function


encode, decode = tokenizer("basic")

# prepare dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # registering buffer does not consider this function as a model parameter
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, C) @ (B, C, T) --> (B, T, T) --> scaled attention
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T) # tril doesnt communicate with the past as for a decoder module
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


# Muti headed attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # concatenate over the channel dimension
        out = self.dropout(self.proj(out))
        return out


# Position-wise feedforward network
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                n_embed, 4 * n_embed
            ),  # inner layer is multiplied by 4 in the position-wise feedforward network
            nn.ReLU(),
            nn.Linear(
                4 * n_embed, n_embed
            ),  # inner layer is multiplied by 4 in the position-wise feedforward network
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# blocks of decoders
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        # n_embed = embedding dimension; n_head = number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(
            n_embed
        )  # slight deviation from the paper, ln happens before the sa and ffwd
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  #  "+" represents residual connections
        x = x + self.ffwd(self.ln2(x))  # "+" represents residual connections
        return x


# Our simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (Batch, Time, Embed Channel)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C) because broadcasting happens to the pos_emb
        x = self.blocks(x)  # apply all the decoder blocks
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (Batch, Time, Channel)

        if targets is None:
            loss = None
        else:
            # cross entropy requires B, C, T for multidimensional input
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # generate the next values of (B, T) for all batches i.e. (B, T+1), (B, T+2),..., (B, T+max_new_tokens)
        for _ in range(max_new_tokens):
            # crop idx to block_size context
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            logits = logits[
                :, -1, :
            ]  # (B, C), pluck out the last element that is our prediction
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# initialize model and optimizer
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), learning_rate)

# train and validate model
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# test model by generating data
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))
