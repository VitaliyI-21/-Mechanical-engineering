import re
from collections import Counter

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

# Тестовий текст
text = "Привіт, як справи? Привіт! Це тестовий текст для аналізу частотності слів."

# Обробка тексту та побудова словника частотності
word_frequency = build_word_frequency(text)

# Вивести результати
print("Словник частотності слів:")
for word, count in word_frequency.items():
    print(f"{word}: {count}")
