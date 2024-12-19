import numpy as np

def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
        theta = theta - (alpha/m) * np.dot(x.T, (h-y))
    J = float(J)
    return J, theta
# Дані
x = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])  # Матриця ознак (доданий стовпець для константи)
y = np.array([[0], [0], [1], [1]])  # Мітки
theta = np.zeros((2, 1))  # Початкові ваги
alpha = 0.1  # Швидкість навчання
num_iters = 1000  # Кількість ітерацій

# Виконання градієнтного спуску
J, theta_optimized = gradientDescent(x, y, theta, alpha, num_iters)

# Результати
print(f"Функція втрат: {J}")
print(f"Оптимізовані ваги: {theta_optimized}")
