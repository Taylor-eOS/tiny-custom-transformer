import re
from collections import Counter
import settings

file_to_analyze = settings.INPUT_FILE
vocab_file = settings.VOCAB_FILE
vocab_limit = settings.VOCABULARY

def count_words(filename):
    word_count = Counter()
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    words = re.findall(r'\b\w+\b', text)
    total_tokens = len(words)
    word_count.update(words)
    total_unique = len(word_count)
    sorted_words = sorted(word_count.items(), key=lambda x: (-x[1], x[0]))
    kept_words = sorted_words[:vocab_limit]
    dropped_words = sorted_words[vocab_limit:]
    kept_unique = len(kept_words)
    dropped_unique = len(dropped_words)
    kept_token_count = sum(count for _, count in kept_words)
    dropped_token_count = total_tokens - kept_token_count
    specials = ['<UNK>', '<PAD>', '<EOS>']
    vocab = specials + [word for word, _ in kept_words]
    with open(vocab_file, 'w', encoding='utf-8') as vf:
        for token in vocab:
            vf.write(token + '\n')
    print(f"Total tokens in corpus: {total_tokens}")
    print(f"Total unique words: {total_unique}")
    print(f"Vocabulary limit: {vocab_limit}")
    print(f"Dropped unique words: {dropped_unique}")
    print(f"Kept token coverage: {kept_token_count} ({kept_token_count / total_tokens:.2%})")
    print(f"Saved vocabulary with {len(vocab)} tokens to '{vocab_file}'")

if __name__ == "__main__":
    count_words(file_to_analyze)

