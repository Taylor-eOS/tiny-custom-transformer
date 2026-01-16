import re

input_file = "input.txt"
vocab_file = "vocab.txt"
output_file = "training.txt"
max_tokens_per_line = 80

with open(vocab_file, 'r', encoding='utf-8') as vf:
    vocab = [line.strip() for line in vf if line.strip()]
stoi = set(vocab)
unk_token = '<UNK>'
eos_token = '<EOS>'
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read().lower()
raw_tokens = re.findall(r'\b\w+\b|[.!?]', text)
processed = []
for tok in raw_tokens:
    if tok in '.!?':
        processed.append(eos_token)
    elif tok in stoi:
        processed.append(tok)
    else:
        processed.append(unk_token)
with open(output_file, 'w', encoding='utf-8') as out:
    line = []
    for tok in processed:
        line.append(tok)
        if len(line) >= max_tokens_per_line:
            out.write(' '.join(line) + '\n')
            line = []
    if line:
        out.write(' '.join(line) + '\n')
print(f"Processed training data written to '{output_file}'")
print(f"Total tokens: {len(processed)}")
print(f"Unique tokens used: {len(set(processed))}")

