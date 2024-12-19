# Твіт для тестування
my_tweet = 'Comes with software, good price, excellent pictures.'

# Використовуємо функцію для передбачення
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
sentiment = 'позитивна' if p > 0 else 'негативна'

# Вивід результату
print(f'Тональність твіту: {sentiment}, передбачене значення: {p}')
import json

# Передбачається, що модель вже навчена
data = {
    'logprior': logprior,
    'loglikelihood': loglikelihood
}

# Збереження у файлі JSON
with open('naive_bayes_model.json', 'w') as outfile:
    json.dump(data, outfile)
