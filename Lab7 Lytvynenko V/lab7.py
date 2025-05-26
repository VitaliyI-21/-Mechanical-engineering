# N-gram Autocomplete System

import nltk
import re
import numpy as np
from collections import Counter

nltk.download('punkt')

# Load data
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Split to sentences
def split_to_sentences(data):
    sentences = data.split('\n')
    return [s.strip() for s in sentences if len(s.strip()) > 0]

# Tokenize

def tokenize_sentences(sentences):
    return [nltk.word_tokenize(s.lower()) for s in sentences]

# Count word frequencies
def count_words(tokenized_sentences):
    word_counts = {}
    for sent in tokenized_sentences:
        for token in sent:
            word_counts[token] = word_counts.get(token, 0) + 1
    return word_counts

# Replace OOV words
def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    vocab_set = set(vocabulary)
    replaced = []
    for sent in tokenized_sentences:
        replaced.append([token if token in vocab_set else unknown_token for token in sent])
    return replaced

# Get frequent words only
def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    word_counts = count_words(tokenized_sentences)
    return [w for w, c in word_counts.items() if c >= count_threshold]

# Preprocess

def preprocess_data(train_data, test_data, count_threshold):
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train = replace_oov_words_by_unk(train_data, vocabulary)
    test = replace_oov_words_by_unk(test_data, vocabulary)
    return train, test, vocabulary

# Count n-grams
def count_n_grams(data, n, start_token='<s>', end_token='</s>'):
    n_grams = {}
    for sent in data:
        sent = [start_token] * (n - 1) + sent + [end_token]
        for i in range(len(sent) - n + 1):
            ngram = tuple(sent[i:i+n])
            n_grams[ngram] = n_grams.get(ngram, 0) + 1
    return n_grams

# Estimate probability
def estimate_probability(word, prev_ngram, ngram_counts, next_ngram_counts, vocab_size, k=1.0):
    prev_ngram = tuple(prev_ngram)
    denom = ngram_counts.get(prev_ngram, 0) + k * vocab_size
    next_ngram = prev_ngram + (word,)
    numer = next_ngram_counts.get(next_ngram, 0) + k
    return numer / denom

# Estimate full distribution
def estimate_probabilities(prev_ngram, ngram_counts, next_ngram_counts, vocab, k=1.0):
    vocab = vocab + ['<unk>', '</s>']
    probs = {}
    for w in vocab:
        probs[w] = estimate_probability(w, prev_ngram, ngram_counts, next_ngram_counts, len(vocab), k)
    return probs

# Calculate perplexity
def calculate_perplexity(sentence, ngram_counts, next_ngram_counts, vocab_size, k=1.0):
    n = len(list(ngram_counts.keys())[0])
    sent = ['<s>'] * n + sentence + ['</s>']
    N = len(sent)
    pp = 1.0
    for t in range(n, N):
        prev_ngram = sent[t - n:t]
        word = sent[t]
        p = estimate_probability(word, prev_ngram, ngram_counts, next_ngram_counts, vocab_size, k)
        pp *= 1 / p
    return pp ** (1 / N)

# Suggest next word
def suggest_a_word(prev_tokens, ngram_counts, next_ngram_counts, vocab, k=1.0, start_with=None):
    n = len(list(ngram_counts.keys())[0])
    prev_tokens = ['<s>'] * max(0, n - len(prev_tokens)) + prev_tokens[-n:]
    probs = estimate_probabilities(prev_tokens, ngram_counts, next_ngram_counts, vocab, k)
    if start_with:
        probs = {w: p for w, p in probs.items() if w.startswith(start_with)}
    best = max(probs.items(), key=lambda x: x[1]) if probs else (None, 0)
    return best

# Launch the autocomplete system
def run_autocomplete_system(file_path):
    print("Loading and tokenizing...")
    data = load_data(file_path)
    tokenized = tokenize_sentences(split_to_sentences(data))

    import random
    random.seed(42)
    random.shuffle(tokenized)
    train = tokenized[:int(len(tokenized)*0.8)]
    test = tokenized[int(len(tokenized)*0.8):]

    train_proc, test_proc, vocab = preprocess_data(train, test, count_threshold=2)
    print(f"Vocabulary size: {len(vocab)}")

    # Example: Build bigram model
    bigrams = count_n_grams(train_proc, 2)
    trigrams = count_n_grams(train_proc, 3)
    print(suggest_a_word(["я", "люблю"], bigrams, trigrams, vocab))

# Example usage:
# run_autocomplete_system("les_poderviansky.txt")
