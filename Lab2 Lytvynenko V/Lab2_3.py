def train_naive_bayes(freqs, train_x, train_y):
    """
    Функція для навчання наївного Байєсового класифікатора.
    Обчислює log prior та log likelihood.
    
    Вхідні дані:
    - freqs: словник частотностей (слово, мітка класу) -> частота.
    - train_x: список тренувальних твідів.
    - train_y: список міток класів (1 для позитивного, 0 для негативного).
    
    Вихідні дані:
    - logprior: логарифм апріорної ймовірності класів.
    - loglikelihood: словник логарифмічних правдоподібностей для кожного слова.
    """
    loglikelihood = {}
    logprior = 0

    # Унікальні слова в словнику частотностей
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # Підрахунок кількості слів у позитивних і негативних класах
    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]

    # Загальна кількість документів та їх розподіл між класами
    D = len(train_y)
    D_pos = sum(train_y)
    D_neg = D - D_pos

    # Обчислення log prior
    logprior = np.log(D_pos) - np.log(D_neg)

    # Обчислення log likelihood для кожного слова
    for word in vocab:
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)
        
        # Лапласівське згладжування
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # Логарифм відношення ймовірностей
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)

print("Log prior:", logprior)
print("Приклади log likelihood:")
for word, ll in list(loglikelihood.items())[:10]:  # Перші 10 слів
    print(f"{word}: {ll}")
