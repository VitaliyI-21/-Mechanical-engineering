import re
import numpy as np
import pandas as pd
from collections import Counter

# 1. Обробка тексту та побудова словника частотності

def process_data(file_name):
    with open(file_name, "r") as f:
        text = f.read().lower()
    words = re.findall(r'\w+', text)
    return words

def get_count(word_l):
    word_count_dict = {}
    for w in word_l:
        if w in word_count_dict:
            word_count_dict[w] += 1
        else:
            word_count_dict[w] = 1
    return word_count_dict

def get_probs(word_count_dict):
    probs = {}
    total_count = sum(word_count_dict.values())
    for word, count in word_count_dict.items():
        probs[word] = count / total_count
    return probs

# 2. Операції редагування

def delete_letter(word):
    return [L + R[1:] for L, R in [(word[:i], word[i:]) for i in range(len(word))] if R]

def switch_letter(word):
    return [L + R[1] + R[0] + R[2:] for L, R in [(word[:i], word[i:]) for i in range(len(word))] if len(R) > 1]

def replace_letter(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [L + c + R[1:] for L, R in [(word[:i], word[i:]) for i in range(len(word))] if R for c in letters if L + c + R[1:] != word]

def insert_letter(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [L + c + R for L, R in [(word[:i], word[i:]) for i in range(len(word)+1)] for c in letters]

# 3. Кандидати на виправлення

def edit_one_letter(word, allow_switches=True):
    edits = set(delete_letter(word) + replace_letter(word) + insert_letter(word))
    if allow_switches:
        edits.update(switch_letter(word))
    return edits

def edit_two_letters(word, allow_switches=True):
    edits = set()
    for w in edit_one_letter(word, allow_switches):
        edits.update(edit_one_letter(w, allow_switches))
    return edits

# 4. Автокорекція

def get_corrections(word, probs, vocab, n=2):
    if word in vocab:
        return [(word, probs.get(word, 0))]

    edit_one_set = edit_one_letter(word) & vocab
    if edit_one_set:
        suggestions = list(edit_one_set)
    else:
        edit_two_set = edit_two_letters(word) & vocab
        suggestions = list(edit_two_set) if edit_two_set else [word]

    best_words = {w: probs.get(w, 0) for w in suggestions}
    return sorted(best_words.items(), key=lambda x: x[1], reverse=True)[:n]

# 5. Мінімальна відстань редагування

def min_edit_distance(source, target, ins_cost=1, del_cost=1, rep_cost=2):
    m, n = len(source), len(target)
    D = np.zeros((m+1, n+1), dtype=int)

    for row in range(m+1):
        D[row, 0] = row * del_cost
    for col in range(n+1):
        D[0, col] = col * ins_cost

    for row in range(1, m+1):
        for col in range(1, n+1):
            r_cost = 0 if source[row-1] == target[col-1] else rep_cost
            D[row, col] = min(
                D[row-1, col] + del_cost,
                D[row, col-1] + ins_cost,
                D[row-1, col-1] + r_cost
            )
    return D, D[m, n]

# 6. Тестування (приклад використання)
if __name__ == "__main__":
    word_l = process_data('shakespeare.txt')
    vocab = set(word_l)
    word_count_dict = get_count(word_l)
    probs = get_probs(word_count_dict)

    test_words = ['hllo', 'mony', 'comming', 'happpy', 'xylophonne']
    for word in test_words:
        corrections = get_corrections(word, probs, vocab, 3)
        print(f"Корекції для '{word}':")
        for i, (correction, prob) in enumerate(corrections):
            print(f"  {i+1}. {correction} (ймовірність: {prob:.6f})")
        print()

    test_pairs = [('play', 'stay'), ('sunday', 'saturday'), ('intention', 'execution')]
    for source, target in test_pairs:
        D, med = min_edit_distance(source, target)
        print(f"Мінімальна відстань між '{source}' та '{target}': {med}")
        print(pd.DataFrame(D, index=['#'] + list(source), columns=['#'] + list(target)))
        print()
