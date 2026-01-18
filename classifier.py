import torch
import random
from classifier_model import BinaryClassifier, predict
import settings

data_path = settings.CLASSIFIER_TRAINING_FILE
vocab_path = settings.VOCAB_FILE

batch_size = 16
block_size = 32
max_steps = 4000
eval_interval = 100
learning_rate = 1e-3
device = 'cpu'
n_embd = 128
n_head = 4
n_layer = 2
dropout = 0.1
torch.manual_seed(1337)
random.seed(1337)

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = [line.strip() for line in f if line.strip()]
stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for tok, i in stoi.items()}
vocab_size = len(vocab)
pad_id = stoi['<PAD>']
unk_id = stoi['<UNK>']

def encode_words(text):
    return [stoi.get(w, unk_id) for w in text.split()]

with open(data_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]
texts = []
labels = []
for line in lines:
    parts = line.split('\t')
    if len(parts) == 2:
        texts.append(parts[0])
        labels.append(int(parts[1]))

combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)
texts = list(texts)
labels = list(labels)

n = int(0.9 * len(texts))
train_texts = texts[:n]
train_labels = labels[:n]
val_texts = texts[n:]
val_labels = labels[n:]

def get_batch(split):
    data_texts = train_texts if split == 'train' else val_texts
    data_labels = train_labels if split == 'train' else val_labels
    ix = torch.randint(0, len(data_texts), (batch_size,))
    x = torch.full((batch_size, block_size), pad_id, dtype=torch.long)
    y = torch.zeros(batch_size, dtype=torch.long)
    for i, idx in enumerate(ix):
        tokens = encode_words(data_texts[idx])
        seq_len = min(len(tokens), block_size)
        x[i, :seq_len] = torch.tensor(tokens[:seq_len], dtype=torch.long)
        y[i] = data_labels[idx]
    return x.to(device), y.to(device)

model = BinaryClassifier(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    pad_id=pad_id,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
for step in range(max_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0 or step == max_steps - 1:
        model.eval()
        total_loss = 0.0
        correct = 0
        num_val = len(val_texts)
        with torch.no_grad():
            for start in range(0, num_val, batch_size):
                end = min(start + batch_size, num_val)
                batch_len = end - start
                x = torch.full((batch_len, block_size), pad_id, dtype=torch.long)
                y = torch.tensor(val_labels[start:end], dtype=torch.long)
                for b in range(batch_len):
                    tokens = encode_words(val_texts[start + b])
                    seq_len = min(len(tokens), block_size)
                    x[b, :seq_len] = torch.tensor(tokens[:seq_len], dtype=torch.long)
                xb = x.to(device)
                yb = y.to(device)
                val_logits, val_loss = model(xb, yb)
                total_loss += val_loss.item() * batch_len
                val_preds = (torch.sigmoid(val_logits).squeeze() > 0.5).long()
                correct += (val_preds == yb).sum().item()
        val_loss = total_loss / num_val if num_val > 0 else 0.0
        val_acc = correct / num_val if num_val > 0 else 0.0
        model.train()
        print(f"step {step}: train loss {loss.item():.4f}, val loss {val_loss:.4f}, val acc {val_acc:.4f}")
        if val_texts:
            val_idx = torch.randint(len(val_texts), (1,)).item()
            test_text = val_texts[val_idx]
            sample_label = val_labels[val_idx]
            prob = predict(model, test_text, vocab, stoi, pad_id, unk_id, block_size, device)
            pred_class = 1 if prob > 0.5 else 0
            print(f"'{test_text}' (true label: {sample_label}) -> prob positive={prob:.4f}")
            print()
