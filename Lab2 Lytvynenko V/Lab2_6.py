def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Функція для оцінки точності наївного Байєсового класифікатора.
    
    Вхідні дані:
    - test_x: список тестових твітів.
    - test_y: список справжніх міток (1 для позитивних, 0 для негативних).
    - logprior: логарифм апріорної ймовірності класів.
    - loglikelihood: словник логарифмічних правдоподібностей для кожного слова.
    
    Вихідні дані:
    - accuracy: точність класифікатора.
    """
    # Список передбачених міток
    y_hats = []

    for tweet in test_x:
        # Передбачення тональності твіту
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1  # Позитивний клас
        else:
            y_hat_i = 0  # Негативний клас
        y_hats.append(y_hat_i)

    # Обчислення середньої абсолютної похибки
    error = np.mean(np.abs(np.array(y_hats) - np.array(test_y)))

    # Обчислення точності
    accuracy = 1 - error

    return accuracy

# Приклад тестових даних
test_x = ["I love this product!", "This is the worst thing ever.", "Amazing experience!", "Not good, very bad."]
test_y = [1, 0, 1, 0]  # Справжні мітки

# Обчислення точності
accuracy = test_naive_bayes(test_x, test_y, logprior, loglikelihood)

print(f"Точність класифікатора: {accuracy:.4f}")
