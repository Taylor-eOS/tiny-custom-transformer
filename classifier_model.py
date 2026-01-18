import torch
import torch.nn as nn
from torch.nn import functional as F
from model import RMSNorm, Head, MultiheadAttention, FeedForward, Block

class BinaryClassifier(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, pad_id, dropout):
        super().__init__()
        self.block_size = block_size
        self.pad_id = pad_id
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, padding_idx=pad_id)
        self.position_embedding_table = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.classifier_head = nn.Linear(n_embd, 1)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table[:, :T, :]
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        mask = (idx != self.pad_id).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier_head(pooled).squeeze(-1)
        loss = None
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        return logits, loss

@torch.no_grad()
def predict(model, text, vocab, stoi, pad_id, unk_id, block_size, device):
    model.eval()
    tokens = [stoi.get(w, unk_id) for w in text.split()]
    idx = torch.full((1, block_size), pad_id, dtype=torch.long, device=device)
    seq_len = min(len(tokens), block_size)
    idx[0, :seq_len] = torch.tensor(tokens[:seq_len], dtype=torch.long)
    logits, _ = model(idx)
    prob = torch.sigmoid(logits).item()
    model.train()
    return prob
