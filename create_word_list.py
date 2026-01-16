import re
from collections import Counter

file_to_analyze = "input.txt"
vocab_limit = 1500
vocab_file = "vocab.txt"

def count_words(filename):
    word_count = Counter()
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    words = re.findall(r'\b\w+\b', text)
    word_count.update(words)
    sorted_words = sorted(word_count.items(), key=lambda x: (-x[1], x[0]))[:vocab_limit]
    specials = ['<UNK>', '<PAD>', '<EOS>']
    vocab = specials + [word for word, _ in sorted_words]
    with open(vocab_file, 'w', encoding='utf-8') as vf:
        for token in vocab:
            vf.write(token + '\n')
    print(f"Saved vocabulary with {len(vocab)} tokens to '{vocab_file}'")

if __name__ == "__main__":
    count_words(file_to_analyze)

