import numpy as np

# Тренувальна вибірка
train_text = """
The original PropBank project, funded by ACE, created a corpus of text annotated with information 
about basic semantic propositions. Predicate-argument relations were added to the syntactic trees 
of the Penn Treebank. This resource is now available via LDC.
"""

# Тестова вибірка
test_text = """
This project was continued under NSF funding and DARPA GALE and BOLT. with the aim of creating Parallel 
PropBanks (the English-Chinese Treebank/PropBank) and also PropBanking other genres, such as Broadcast News, 
Broadcast Conversation, WebText and Discussion Fora, at the University of Colorado.
"""

# Підготовка вибірок
train_x = train_text.strip().split('\n')  # Розділити на рядки (або можна на речення)
test_x = test_text.strip().split('\n')

# Мітки (можуть бути створені вручну, наприклад 1 для тренувальної вибірки і 0 для тестової)
train_y = np.ones(len(train_x))
test_y = np.zeros(len(test_x))

# Виведення результатів
print("Train X:", train_x)
print("Train Y:", train_y)
print("Test X:", test_x)
print("Test Y:", test_y)
