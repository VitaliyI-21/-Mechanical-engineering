# POS Tagging with Hidden Markov Models (HMM) and Viterbi Algorithm

import numpy as np
import pandas as pd
import string
import math
from collections import defaultdict

# Punctuation set
punct = set(string.punctuation)

# English suffix rules
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]

def assign_unk(tok):
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"
    elif any(char in punct for char in tok):
        return "--unk_punct--"
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    return "--unk--"

def preprocess(vocab, data_fp):
    orig, prep = [], []
    if "--n--" not in vocab:
        vocab["--n--"] = len(vocab)
    with open(data_fp, "r") as data_file:
        for word in data_file:
            if not word.split():
                orig.append(word.strip())
                prep.append("--n--")
            elif word.strip() not in vocab:
                orig.append(word.strip())
                prep.append(assign_unk(word.strip()))
            else:
                orig.append(word.strip())
                prep.append(word.strip())
    return orig, prep

def get_word_tag(line, vocab):
    if not line.split():
        return "--n--", "--s--"
    parts = line.split()
    if len(parts) >= 2:
        word, tag = parts[0], parts[1]
        if word not in vocab:
            word = assign_unk(word)
        return word, tag
    return "--n--", "--s--"

def create_dictionaries(training_corpus, vocab):
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    prev_tag = '--s--'
    for word_tag in training_corpus:
        word, tag = get_word_tag(word_tag, vocab)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1
        prev_tag = tag
    return emission_counts, transition_counts, tag_counts

def create_transition_matrix(alpha, tag_counts, transition_counts):
    tags = sorted(tag_counts.keys())
    A = np.zeros((len(tags), len(tags)))
    for i, ti in enumerate(tags):
        for j, tj in enumerate(tags):
            A[i, j] = (transition_counts[(ti, tj)] + alpha) / (tag_counts[ti] + alpha * len(tags))
    return A

def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    tags = sorted(tag_counts.keys())
    B = np.zeros((len(tags), len(vocab)))
    for i, tag in enumerate(tags):
        for j, word in enumerate(vocab):
            B[i, j] = (emission_counts[(tag, word)] + alpha) / (tag_counts[tag] + alpha * len(vocab))
    return B

def initialize(states, tag_counts, A, B, corpus, vocab):
    n_tags = len(tag_counts)
    best_probs = np.zeros((n_tags, len(corpus)))
    best_paths = np.zeros((n_tags, len(corpus)), dtype=int)
    s_idx = states.index("--s--")
    word = corpus[0]
    word_idx = vocab.get(word, vocab.get("--unk--", 0))
    for i in range(n_tags):
        best_probs[i, 0] = math.log(A[s_idx, i]) + math.log(B[i, word_idx])
    return best_probs, best_paths

def viterbi_forward(A, B, corpus, best_probs, best_paths, vocab):
    for i in range(1, len(corpus)):
        word = corpus[i]
        word_idx = vocab.get(word, vocab.get("--unk--", 0))
        for j in range(best_probs.shape[0]):
            best_prob_i = float("-inf")
            best_path_i = None
            for k in range(best_probs.shape[0]):
                prob = best_probs[k, i-1] + math.log(A[k, j]) + math.log(B[j, word_idx])
                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
            best_probs[j, i] = best_prob_i
            best_paths[j, i] = best_path_i
    return best_probs, best_paths

def viterbi_backward(best_probs, best_paths, corpus, states):
    m = best_paths.shape[1]
    z = [0] * m
    pred = ["" for _ in range(m)]
    max_prob = float('-inf')
    for k in range(best_probs.shape[0]):
        if best_probs[k, m-1] > max_prob:
            max_prob = best_probs[k, m-1]
            z[m-1] = k
    pred[m-1] = states[z[m-1]]
    for i in range(m-2, -1, -1):
        z[i] = best_paths[z[i+1], i+1]
        pred[i] = states[z[i]]
    return pred

def compute_accuracy(pred, y):
    correct = 0
    total = 0
    for prediction, actual in zip(pred, y):
        parts = actual.strip().split('\t')
        if len(parts) != 2:
            continue
        word, tag = parts
        if tag == prediction:
            correct += 1
        total += 1
    return correct / total

# Example usage
if __name__ == "__main__":
    with open("./data/WSJ_02-21.pos", 'r') as f:
        training_corpus = f.readlines()

    with open("./data/hmm_vocab.txt", 'r') as f:
        voc_l = f.read().split('\n')

    vocab = {word: i for i, word in enumerate(sorted(voc_l))}

    with open("./data/WSJ_24.pos", 'r') as f:
        y = f.readlines()

    _, prep = preprocess(vocab, "./data/test.words")

    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
    states = sorted(tag_counts.keys())

    alpha = 0.001
    word_to_index = {word: i for i, word in enumerate(sorted(vocab.keys()))}

    for token in ["--n--", "--unk--", "--unk_digit--", "--unk_punct--", "--unk_upper--", "--unk_noun--", "--unk_verb--", "--unk_adj--", "--unk_adv--"]:
        if token not in word_to_index:
            word_to_index[token] = len(word_to_index)

    A = create_transition_matrix(alpha, tag_counts, transition_counts)
    B = create_emission_matrix(alpha, tag_counts, emission_counts, list(word_to_index.keys()))

    best_probs, best_paths = initialize(states, tag_counts, A, B, prep, word_to_index)
    best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, word_to_index)
    pred = viterbi_backward(best_probs, best_paths, prep, states)

    accuracy = compute_accuracy(pred, y)
    print(f"Точність моделі: {accuracy:.4f}")
