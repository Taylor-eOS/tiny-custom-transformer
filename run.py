import torch
from model import SmallTransformer, generate
import settings

data_path = settings.TRAINING_FILE
vocab_path = settings.VOCAB_FILE
batch_size = 8
block_size = 64
max_steps = 5000
eval_interval = 400
learning_rate = 2e-4
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
        if i == pad_id:
            continue
        tok = itos[i]
        if tok == '<EOS>':
            if out:
                out[-1] = out[-1] + '.'
        else:
            out.append(tok)
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

model = SmallTransformer(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, pad_id=pad_id, dropout=dropout).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

for step in range(max_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb, pad_id=pad_id)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0 or step == max_steps - 1:
        print(f"step {step}: train loss {loss.item():.4f}")
        start = torch.randint(vocab_size, (1, 1), device=device)
        generated = generate(model, start, 40, block_size, device)
        print(decode_words(generated[0].tolist()))
        print()

