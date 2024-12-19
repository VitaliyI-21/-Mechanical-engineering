def lookup(freqs, word, label):
    """
    Повертає частотність слова для вказаного класу.
    
    Вхідні дані:
    - freqs: словник частотностей слів.
    - word: слово для пошуку.
    - label: мітка класу (1 для позитивного, 0 для негативного).
    
    Вихідні дані:
    - n: частотність слова для заданого класу.
    """
    n = 0  
    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]
    return n

def get_ratio(freqs, word):
    """
    Обчислює відношення частотності слова для позитивного та негативного класів.
    
    Вхідні дані:
    - freqs: словник частотностей слів.
    - word: слово, для якого обчислюється відношення.
    
    Вихідні дані:
    - pos_neg_ratio: словник з частотностями та відношенням.
    """
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    
    # Частотність для позитивного класу
    pos_neg_ratio['positive'] = lookup(freqs, word, 1)

    # Частотність для негативного класу
    pos_neg_ratio['negative'] = lookup(freqs, word, 0)
    
    # Обчислення відношення з лапласівським згладжуванням
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)
    
    return pos_neg_ratio

def get_words_by_threshold(freqs, label, threshold):
    """
    Повертає слова, частотності яких задовольняють заданому порогу.
    
    Вхідні дані:
    - freqs: словник частотностей слів.
    - label: мітка класу (1 для позитивного, 0 для негативного).
    - threshold: поріг для відбору слів.
    
    Вихідні дані:
    - word_list: словник слів і їх частотностей, які відповідають умовам.
    """
    word_list = {}

    for key in freqs.keys():
        word, _ = key
        
        # Обчислення відношення частотностей для слова
        pos_neg_ratio = get_ratio(freqs, word)

        # Відбір слів на основі мітки та порогу
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:
            word_list[word] = pos_neg_ratio
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
            word_list[word] = pos_neg_ratio

    return word_list

# Приклад частотностей
freqs = {
    ('love', 1): 10,
    ('love', 0): 1,
    ('hate', 1): 1,
    ('hate', 0): 10,
    ('happy', 1): 8,
    ('happy', 0): 2,
    ('sad', 1): 2,
    ('sad', 0): 8
}

# Пошук найбільш позитивних слів із порогом 2
positive_words = get_words_by_threshold(freqs, label=1, threshold=2)
print("Позитивні слова:", positive_words)

# Пошук найбільш негативних слів із порогом 0.5
negative_words = get_words_by_threshold(freqs, label=0, threshold=0.5)
print("Негативні слова:", negative_words)
