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

def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Тестування моделі логістичної регресії на тестових даних.
    Обчислює точність моделі.
    """
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
    accuracy = (np.sum(np.array(y_hat) == np.squeeze(test_y))) / len(test_x)  
    return accuracy

# Тестовий текст
test_text = "The original PropBank project, funded by ACE, created a corpus of text annotated with information \nabout basic semantic propositions. Predicate-argument relations were added to the syntactic trees \nof the Penn Treebank. This resource is now available via LDC."

# Обробка тексту та побудова словника частотності
word_frequency = build_word_frequency(test_text)

# Приклад словника частотностей для позитивних і негативних слів
freqs = {
    ("original", 1.0): 1,
    ("propbank", 1.0): 1,
    ("project", 0.0): 1,
    ("corpus", 0.0): 1
}

# Початкові ваги для тесту
theta = np.array([[0], [0.5], [-0.5]])

# Тренувальні дані
train_x = ["The original project was successful", "This is a bad example", "Great resource", "Horrible mistake"]
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

# Тестові дані
test_x = ["Amazing project", "Terrible result", "Fantastic approach"]
test_y = np.array([[1], [0], [1]])  # Мітки (1 - позитивний, 0 - негативний)

# Тестування моделі
tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Точність логістичної регресії: {tmp_accuracy:.4f}")

# Перевірка моделі на власних прикладах
my_tweet = 'This model performs exceptionally well!'
print("Оброблений текст:", preprocess_text(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(f"Ймовірність позитивного класу: {y_hat}")
if y_hat > 0.5:
    print('Позитивна тональність')
else:
    print('Негативна тональність')
