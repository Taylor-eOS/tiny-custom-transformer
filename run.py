import torch
import torch.nn as nn
from torch.nn import functional as F

data_path = 'training.txt'
vocab_path = 'vocab.txt'
batch_size = 8
block_size = 64
max_steps = 5000
eval_interval = 500
learning_rate = 4e-4
device = 'cpu'
n_embd = 256
n_head = 8
n_layer = 4
dropout = 0.1
torch.manual_seed(1337)

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = [line.strip() for line in f if line.strip()]

stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for tok, i in stoi.items()}
vocab_size = len(vocab)
pad_id = stoi['<PAD>']
unk_id = stoi['<UNK>']
eos_id = stoi['<EOS>']

def encode_words(text):
    return [stoi.get(w, unk_id) for w in text.split()]

def decode_words(ids):
    out = []
    for i in ids:
        if i == eos_id:
            out.append('.')
        elif i != pad_id:
            out.append(itos[i])
    return ' '.join(out)

with open(data_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

tokens = encode_words(raw_text)
data = torch.tensor(tokens, dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data_split) - 1, (batch_size,))
    x = torch.full((batch_size, block_size), pad_id, dtype=torch.long)
    y = torch.full((batch_size, block_size), pad_id, dtype=torch.long)
    for i, start in enumerate(ix):
        seq = data_split[start:start + block_size + 1]
        seq_len = min(len(seq) - 1, block_size)
        x[i, :seq_len] = seq[:seq_len]
        y[i, :seq_len] = seq[1:seq_len + 1]
    return x.to(device), y.to(device)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x / rms * self.scale

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.scaling = head_size ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.scaling
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 8 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.c_proj(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SmallTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, padding_idx=pad_id)
        self.position_embedding_table = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table[:, :T, :]
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, vocab_size),
                targets.view(B * T),
                ignore_index=pad_id
            )
        return logits, loss

@torch.no_grad()
def generate(model, max_new_tokens):
    model.eval()
    idx = torch.full((1, 1), pad_id, dtype=torch.long, device=device)
    idx[0, 0] = torch.randint(vocab_size, (1,))
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return idx

model = SmallTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

for step in range(max_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0 or step == max_steps - 1:
        print(f"step {step}: train loss {loss.item():.4f}")
        generated = generate(model, max_new_tokens=50)
        print(decode_words(generated[0].tolist()))
        print('\n---\n')

