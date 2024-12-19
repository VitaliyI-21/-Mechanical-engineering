import numpy as np

# Функція для обробки тексту
def process_tweet(tweet):
    # Простий приклад обробки тексту (можна замінити на більш складний препроцесинг)
    return tweet.lower().split()

# Приклад функції класифікації
def naive_bayes_predict(tweet, logprior, loglikelihood):
    words = process_tweet(tweet)
    prob = logprior
    for word in words:
        prob += loglikelihood.get(word, 0)
    return prob

# Тестовий набір даних (спрощений приклад)
test_x = ["I love this!", "This is terrible!", "Just okay", "Awesome movie!", "Not good at all"]
test_y = [1, 0, 1, 1, 0]  # Справжні мітки (1 - позитивний, 0 - негативний)

# Параметри наївного Байєса (приклад)
logprior = 0.0
loglikelihood = {"love": 0.5, "terrible": -1.0, "okay": 0.2, "awesome": 0.8, "not": -0.4, "good": 0.3, "at": -0.1, "all": -0.2}

# Аналіз помилок
print('Truth\tPredicted\tTweet')

for x, y in zip(test_x, test_y):
    y_hat = naive_bayes_predict(x, logprior, loglikelihood)
    predicted_label = 1 if y_hat > 0 else 0  # Конвертація ймовірності у клас
    if y != predicted_label:
        processed_tweet = ' '.join(process_tweet(x))
        print(f'{y}\t{predicted_label}\t{processed_tweet}')
