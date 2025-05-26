import numpy as np
import nltk
import matplotlib.pyplot as plt
import pickle
import re
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# 1. Токенізація тексту

def tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data = nltk.word_tokenize(data)
    return [ch.lower() for ch in data if ch.isalpha() or ch == '.']

# 2. Словники

def get_dict(data):
    words = sorted(set(data))
    word2Ind = {word: i for i, word in enumerate(words)}
    Ind2word = {i: word for word, i in word2Ind.items()}
    return word2Ind, Ind2word

# 3. Вікна контексту

def get_windows(words, C):
    for i in range(C, len(words) - C):
        yield words[i - C:i] + words[i+1:i+C+1], words[i]

# 4. One-hot вектори

def word_to_one_hot_vector(word, word2Ind, V):
    vec = np.zeros(V)
    vec[word2Ind[word]] = 1
    return vec

def context_words_to_vector(context_words, word2Ind, V):
    vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    return np.mean(vectors, axis=0)

# 5. Навчальні приклади

def get_training_example(words, C, word2Ind, V):
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)

# 6. Ініціалізація

def initialize_model(N, V, seed=1):
    np.random.seed(seed)
    return (np.random.rand(N, V), np.random.rand(V, N), np.random.rand(N, 1), np.random.rand(V, 1))

# 7. Softmax

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / np.sum(e_z, axis=0)

# 8. Forward

def forward_prop(x, W1, W2, b1, b2):
    h = np.maximum(0, np.dot(W1, x) + b1)
    z = np.dot(W2, h) + b2
    return z, h

# 9. Cost

def compute_cost(y, yhat, m):
    return - np.sum(y * np.log(yhat)) / m

# 10. Backprop

def back_prop(x, yhat, y, h, W1, W2, b1, b2, m):
    z1 = np.dot(W1, x) + b1
    l1 = np.dot(W2.T, yhat - y)
    l1[z1 < 0] = 0
    grad_W1 = np.dot(l1, x.T) / m
    grad_W2 = np.dot(yhat - y, h.T) / m
    grad_b1 = np.sum(l1, axis=1, keepdims=True) / m
    grad_b2 = np.sum(yhat - y, axis=1, keepdims=True) / m
    return grad_W1, grad_W2, grad_b1, grad_b2

# 11. Навчання

def gradient_descent(data, word2Ind, N, V, num_iters=100, alpha=0.03, batch_size=128, C=2):
    W1, W2, b1, b2 = initialize_model(N, V)
    examples = list(get_training_example(data, C, word2Ind, V))
    for it in range(num_iters):
        np.random.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            x = np.array([ex[0] for ex in batch]).T
            y = np.array([ex[1] for ex in batch]).T
            z, h = forward_prop(x, W1, W2, b1, b2)
            yhat = softmax(z)
            cost = compute_cost(y, yhat, batch_size)
            grads = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)
            W1 -= alpha * grads[0]
            W2 -= alpha * grads[1]
            b1 -= alpha * grads[2]
            b2 -= alpha * grads[3]
        if (it+1) % 10 == 0:
            print(f"Iteration {it+1}, cost: {cost:.4f}")
    return W1, W2

# 12. PCA

def compute_pca(X, n_components=2):
    X -= np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return np.dot(X, eigvecs[:, -n_components:])

# 13. Візуалізація

def visualize_embeddings(embeddings, word2Ind, words):
    idx = [word2Ind[w] for w in words if w in word2Ind]
    X = embeddings[idx, :]
    X_pca = compute_pca(X, 2)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    for i, word in enumerate(words):
        if word in word2Ind:
            plt.annotate(word, (X_pca[i, 0], X_pca[i, 1]))
    plt.grid(True)
    plt.title("PCA visualization of word embeddings")
    plt.show()

# 14. Збереження

def save_embeddings(W1, W2, word2Ind, filename):
    embeddings = (W1.T + W2) / 2
    with open(filename, 'wb') as f:
        pickle.dump({"embeddings": embeddings, "word2Ind": word2Ind}, f)

# 15. Приклад запуску
if __name__ == "__main__":
    with open("les_poderviansky.txt", encoding="utf-8") as f:
        corpus = f.read()
    words = tokenize(corpus)
    word2Ind, Ind2word = get_dict(words)
    V = len(word2Ind)
    W1, W2 = gradient_descent(words, word2Ind, N=50, V=V, num_iters=100)
    save_embeddings(W1, W2, word2Ind, "word_embeddings.pkl")
    visualize_embeddings((W1.T + W2) / 2, word2Ind, ["король", "жінка", "чоловік", "королева"])
