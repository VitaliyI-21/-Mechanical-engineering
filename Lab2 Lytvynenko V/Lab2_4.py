def naive_bayes_predict(tweet, logprior, loglikelihood):
    """
    Функція для передбачення тональності твіту на основі наївного Байєсового класифікатора.
    
    Вхідні дані:
    - tweet: рядок, що містить твіт для класифікації.
    - logprior: логарифм апріорної ймовірності класів.
    - loglikelihood: словник логарифмічних правдоподібностей для кожного слова.
    
    Вихідні дані:
    - p: логарифмічна ймовірність приналежності твіту до позитивного класу.
    """
    # Попередня обробка твіту
    word_l = process_tweet(tweet)

    # Початкова ймовірність
    p = logprior

    # Додавання loglikelihood для кожного слова
    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]

    return p

# Приклад твіту
custom_tweet = "This movie is fantastic! Best I've seen in years."

# Передбачення тональності
predicted_score = naive_bayes_predict(custom_tweet, logprior, loglikelihood)

print(f"Ймовірність позитивної тональності: {predicted_score}")

# Інтерпретація результату
if predicted_score > 0:
    print("Позитивна тональність")
else:
    print("Негативна тональність")
