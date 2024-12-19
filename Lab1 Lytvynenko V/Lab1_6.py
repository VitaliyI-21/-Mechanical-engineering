import re
from collections import Counter
import numpy as np

def preprocess_text(text):
    """
    Функція для попередньої обробки тексту.
    Виконує такі дії:
    - Приводить текст до нижнього регістру.
    - Видаляє спеціальні символи та цифри.
    - Розбиває текст на слова.
    """
    text = text.lower()  # Привести текст до нижнього регістру
    text = re.sub(r'[^a-zA-Zа-яА-ЯїЇєЄіІґҐ\s]', '', text)  # Видалити спеціальні символи та цифри
    words = text.split()  # Розбити текст на слова
    return words

def build_word_frequency(text):
    """
    Функція для побудови словника частотності слів.
    """
    words = preprocess_text(text)
    word_counts = Counter(words)
    return word_counts

def sigmoid(z):
    """Функція активації сигмоїд."""
    h = 1 / (1 + np.exp(-z))
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    """
    Реалізація алгоритму градієнтного спуску для логістичної регресії.
    """
    m = x.shape[0]
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
        theta = theta - (alpha/m) * np.dot(x.T, (h-y))
    J = float(J)
    return J, theta

def extract_features(tweet, freqs):
    """
    Формує вектор ознак для одного твіту.
    Перша ознака - bias (завжди дорівнює 1),
    друга - кількість позитивних слів у твіті,
    третя - кількість негативних слів у твіті.
    """
    word_l = preprocess_text(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1  # Bias
    for word in word_l:
        if (word, 1.0) in freqs:
            x[0, 1] += freqs[(word, 1.0)]
        if (word, 0.0) in freqs:
            x[0, 2] += freqs[(word, 0.0)]
    return x

def predict_tweet(tweet, freqs, theta):
    """
    Передбачає тональність твіту за допомогою натренованої моделі.
    Повертає ймовірність приналежності до позитивного класу.
    """
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

# Тестовий текст
text = "The original PropBank project, funded by ACE, created a corpus of text annotated with information \nabout basic semantic propositions. Predicate-argument relations were added to the syntactic trees \nof the Penn Treebank. This resource is now available via LDC."

# Обробка тексту та побудова словника частотності
word_frequency = build_word_frequency(text)

# Приклад словника частотностей для позитивних і негативних слів
freqs = {
    ("original", 1.0): 2,
    ("project", 1.0): 1,
    ("predicate", 0.0): 1,
    ("treebank", 0.0): 1
}

# Початкові ваги для тесту
theta = np.array([[0], [0.5], [-0.5]])

# Тренувальні дані
train_x = ["The PropBank project was remarkable", "This is a terrible outcome", "Amazing progress in Treebank", "Absolutely disappointing results"]
train_y = np.array([[1], [0], [1], [0]])  # Мітки (1 - позитивний, 0 - негативний)

# Формування матриці ознак для тренувальних даних
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

Y = train_y

# Навчання моделі
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

# Результати
print(f"Помилка після навчання: {J:.8f}.")
print(f"Вектор вагових коефіцієнтів: {[round(t, 8) for t in np.squeeze(theta)]}")

# Тестовий твіт
tweet = "Fantastic development in the Treebank corpus!"
features = extract_features(tweet, freqs)
y_pred = predict_tweet(tweet, freqs, theta)
print("Вектор ознак:", features)
print("Ймовірність позитивного класу:", y_pred)
