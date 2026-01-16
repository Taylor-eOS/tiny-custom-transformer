import re

input_file = "input.txt"
vocab_file = "vocab.txt"
output_file = "training.txt"

with open(vocab_file, 'r', encoding='utf-8') as vf:
    vocab = [line.strip() for line in vf if line.strip()]

stoi = {tok: i for i, tok in enumerate(vocab)}
unk_token = '<UNK>'
eos_token = '<EOS>'

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read().lower()

tokens = re.findall(r'\b\w+\b|[.!?]', text)

processed = []
for tok in tokens:
    if tok in '.!?':
        processed.append(eos_token)
    elif tok in stoi:
        processed.append(tok)
    else:
        processed.append(unk_token)

with open(output_file, 'w', encoding='utf-8') as out:
    out.write(' '.join(processed))

print(f"Processed training data written to '{output_file}'")
print(f"Total tokens: {len(processed)}")
print(f"Unique tokens used: {len(set(processed))}")

