# POS Tagging with Hidden Markov Models (HMM) and Viterbi Algorithm (using Floresta Corpus)

import numpy as np
import pandas as pd
import string
import math
from collections import defaultdict
import nltk
from nltk.corpus import floresta
from nltk.tag import UnigramTagger, BigramTagger
nltk.download('floresta')

# 1. Завантаження корпусу Floresta Treebank
sentences = floresta.tagged_sents()

# 2. Розділення на навчальний і тестовий корпуси
split_idx = int(len(sentences) * 0.8)
train_sentences = sentences[:split_idx]
test_sentences = sentences[split_idx:]

# 3. Побудова частотних словників
emission_counts = defaultdict(int)
transition_counts = defaultdict(int)
tag_counts = defaultdict(int)

prev_tag = '--s--'
for sent in train_sentences:
    for word, tag in sent:
        emission_counts[(tag, word.lower())] += 1
        transition_counts[(prev_tag, tag)] += 1
        tag_counts[tag] += 1
        prev_tag = tag
    transition_counts[(prev_tag, '--s--')] += 1  # Кінець речення
    prev_tag = '--s--'

# 4. Унікальні теги і словник
tags = sorted(tag_counts.keys())
vocab = sorted(set([word.lower() for sent in train_sentences for word, _ in sent]))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}

# 5. Матриці переходів та емісій

def create_transition_matrix(alpha=0.001):
    A = np.zeros((len(tags), len(tags)))
    for i, tag_i in enumerate(tags):
        for j, tag_j in enumerate(tags):
            count = transition_counts.get((tag_i, tag_j), 0)
            A[i, j] = (count + alpha) / (tag_counts[tag_i] + alpha * len(tags))
    return A

def create_emission_matrix(alpha=0.001):
    B = np.zeros((len(tags), len(vocab)))
    for i, tag in enumerate(tags):
        for j, word in enumerate(vocab):
            count = emission_counts.get((tag, word), 0)
            B[i, j] = (count + alpha) / (tag_counts[tag] + alpha * len(vocab))
    return B

A = create_transition_matrix()
B = create_emission_matrix()

# 6. Вітербі алгоритм

def initialize(A, B, test_words):
    T = len(test_words)
    N = len(tags)
    best_probs = np.full((N, T), -np.inf)
    best_paths = np.zeros((N, T), dtype=int)
    first_word = test_words[0].lower()
    word_idx = word_to_idx.get(first_word, word_to_idx.get('<unk>', 0))
    s_idx = tag_to_idx['--s--'] if '--s--' in tag_to_idx else 0
    for i in range(N):
        best_probs[i, 0] = np.log(A[s_idx, i]) + np.log(B[i, word_idx])
    return best_probs, best_paths

def viterbi_forward(A, B, test_words, best_probs, best_paths):
    T = len(test_words)
    N = len(tags)
    for t in range(1, T):
        word = test_words[t].lower()
        word_idx = word_to_idx.get(word, word_to_idx.get('<unk>', 0))
        for j in range(N):
            max_prob = -np.inf
            best_k = 0
            for i in range(N):
                prob = best_probs[i, t-1] + np.log(A[i, j]) + np.log(B[j, word_idx])
                if prob > max_prob:
                    max_prob = prob
                    best_k = i
            best_probs[j, t] = max_prob
            best_paths[j, t] = best_k
    return best_probs, best_paths

def viterbi_backward(best_probs, best_paths, test_words):
    T = len(test_words)
    z = np.zeros(T, dtype=int)
    pred = [''] * T
    z[T-1] = np.argmax(best_probs[:, T-1])
    pred[T-1] = tags[z[T-1]]
    for t in range(T-2, -1, -1):
        z[t] = best_paths[z[t+1], t+1]
        pred[t] = tags[z[t]]
    return pred

# 7. Оцінка точності

def compute_accuracy(pred_tags, gold_sentences):
    correct, total = 0, 0
    idx = 0
    for sent in gold_sentences:
        for _, tag in sent:
            if idx >= len(pred_tags):
                break
            if tag == pred_tags[idx]:
                correct += 1
            total += 1
            idx += 1
    return correct / total

# 8. Запуск моделі на тестовому реченні (можна розширити на весь корпус)
test_sent = [word for word, _ in test_sentences[0]]
best_probs, best_paths = initialize(A, B, test_sent)
best_probs, best_paths = viterbi_forward(A, B, test_sent, best_probs, best_paths)
pred_tags = viterbi_backward(best_probs, best_paths, test_sent)

# 9. Точність моделі (на одному реченні, для прикладу)
accuracy = compute_accuracy(pred_tags, [test_sentences[0]])
print("Прогнозовані теги:", pred_tags)
print("Точність (на одному реченні):", accuracy)

# 10. Порівняння з NLTK Tagger
unigram_tagger = UnigramTagger(train_sentences)
bigram_tagger = BigramTagger(train_sentences, backoff=unigram_tagger)
nltk_accuracy = bigram_tagger.evaluate(test_sentences)
print("Точність NLTK BigramTagger:", nltk_accuracy)
