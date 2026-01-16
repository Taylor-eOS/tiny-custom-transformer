import re

input_file = "input.txt"
vocab_file = "vocab.txt"
output_file = "training.txt"

with open(vocab_file, 'r', encoding='utf-8') as vf:
    vocab_lines = [line.strip() for line in vf if line.strip()]
specials = ['<UNK>', '<PAD>']
allowed_words = set(vocab_lines) - set(specials)
unk_token = '<UNK>'
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read().lower()
words = re.findall(r'\b\w+\b', text)
processed_tokens = []
for word in words:
    if word in allowed_words:
        processed_tokens.append(word)
    else:
        processed_tokens.append(unk_token)
with open(output_file, 'w', encoding='utf-8') as out:
    line_length = 0
    max_line_tokens = 80
    for i, token in enumerate(processed_tokens):
        if i > 0:
            out.write(' ')
            line_length += 1
        out.write(token)
        line_length += 1
        if line_length >= max_line_tokens:
            out.write('\n')
            line_length = 0
    out.write('\n')
print(f"Processed training data written to '{output_file}'")
print(f"Total tokens: {len(processed_tokens)}")
print(f"Unique tokens used: {len(set(processed_tokens))}")
print("First 10 tokens preview:")
print(' '.join(processed_tokens[:10]))
