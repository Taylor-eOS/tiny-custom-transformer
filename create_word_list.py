import re
from collections import Counter

file_to_analyze = "input.txt"
vocab_limit = 1000
vocab_file = "vocab.txt"

def count_words(filename):
    word_count = Counter()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        words = re.findall(r'\b\w+\b', text)
        if not words:
            print("No words found in the file.")
            return
        word_count.update(words)
        sorted_words = sorted(word_count.items(), key=lambda x: (-x[1], x[0]))[:vocab_limit]
        if not sorted_words:
            print("No words found.")
            return
        print(f"\nTotal unique words: {len(word_count)}")
        print(f"Total words counted: {sum(word_count.values())}")
        print(f"Words shown: {len(sorted_words)}\n")
        print("Top words:")
        print("-----------------------------")
        for i, (word, count) in enumerate(sorted_words, 1):
            print(f"{i:3d}. {word:18} : {count}")

        specials = ['<UNK>', '<PAD>']
        vocab = specials + [word for word, _ in sorted_words]

        with open(vocab_file, 'w', encoding='utf-8') as vf:
            for token in vocab:
                vf.write(token + '\n')

        print(f"\nSaved vocabulary with {len(vocab)} tokens to '{vocab_file}'")
        print("First few lines of the file:")
        for i, token in enumerate(vocab[:10], 1):
            print(f"{i:3d}. {token}")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error while processing file: {e}")

if __name__ == "__main__":
    count_words(file_to_analyze)
