# src/preprocessing.py

import re
import pandas as pd
import numpy as np
import pymorphy3
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split

# Скачиваем стоп-слова
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words('russian'))

# Словари для редактирования
SLANG_DICT = {
    'общага': 'общежитие',
    'общагу': 'общежитие',
    'общаге': 'общежитие',
    'бюдж': 'бюджет',
    'физвоз': 'физическая культура',
    'егэ': 'единый государственный экзамен',
    'ви': 'вступительное испытание',
    'лк': 'личный кабинет',
    'пк': 'приемная комиссия',
}

TYPO_DICT = {
    'билогия': 'биология',
    'билогии': 'биология',
    'уважнием': 'уважением',
    'здраствуйте': 'здравствуйте',
    'педаго': 'педагог',
}


# ---------------------------------------------------------
# 1. Базовая очистка (Clean Text)
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'^(Здравствуйте|Добрый день|Доброго дня|Добрый вечер)[!,.]?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(С\s+уважением|С\s+уважнием|Спасибо|Благодарю).*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'#+', ' <NUM> ', text)
    text = re.sub(r'\b\d+\b', ' <NUM> ', text)
    text = re.sub(r'[^а-яА-Яa-zA-Z\s<>]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# ---------------------------------------------------------
# 2. Лемматизация (для LSTM)
# ---------------------------------------------------------
def lemmatize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    words = text.split()
    lemmas = []
    for word in words:
        if word == '<num>':
            lemmas.append('<num>')
            continue
        if word in russian_stopwords:
            continue
        p = morph.parse(word)[0]
        lemma = p.normal_form
        if len(lemma) == 1:
            continue
        lemmas.append(lemma)
    return ' '.join(lemmas)


# ---------------------------------------------------------
# 3. Стемминг (для Классики / TF-IDF)
# ---------------------------------------------------------
def stem_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    words = text.split()
    stems = []
    for word in words:
        if word == '<num>':
            stems.append('<num>')
            continue
        if word in russian_stopwords:
            continue
        stem = morph.parse(word)[0].normal_form[:5]
        if len(stem) == 1:
            continue
        stems.append(stem)
    return ' '.join(stems)


# ---------------------------------------------------------
# 4. Редактирование (Опечатки и Сленг)
# ---------------------------------------------------------
def fix_typos(text: str) -> str:
    words = text.split()
    return ' '.join([TYPO_DICT.get(w, w) for w in words])


def replace_slang(text: str) -> str:
    words = text.split()
    return ' '.join([SLANG_DICT.get(w, w) for w in words])


# ---------------------------------------------------------
# 5. Аугментация (Размножение редких классов) — только для TRAIN
# ---------------------------------------------------------
def augment_minority_classes(
    df: pd.DataFrame,
    target_classes=(7, 8, 3),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Доводит размер каждого редкого класса до медианного размера класса
    в датасете
    Мутация — перемешивание слов (без маркеров-утечек)
    """
    print("Выполняем аугментацию редких классов (только train)...")
    rng = np.random.default_rng(random_state)

    counts = df['class'].value_counts()
    target_size = int(counts.median())
    print(f"Медианный размер класса: {target_size}")

    augmented_rows = []

    for cls in target_classes:
        class_df = df[df['class'] == cls]
        if class_df.empty:
            continue

        need = max(0, target_size - len(class_df))
        if need == 0:
            print(f"  Класс {cls}: уже {len(class_df)} строк — пропуск")
            continue

        for _ in range(need):
            row = class_df.sample(1, random_state=int(rng.integers(1e9))).iloc[0].copy()

            # Перемешиваем слова в лемматизированном тексте — реальная вариация
            words = row['CONTENT_lemm'].split()
            rng.shuffle(words)
            row['CONTENT_lemm'] = ' '.join(words)

            # Стеммированный текст пересобираем из перемешанных лемм
            row['CONTENT_stem'] = stem_text(row['CONTENT_lemm'])

            augmented_rows.append(row)

        print(f"  Класс {cls}: {len(class_df)} -> {len(class_df) + need}")

    df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    print(f"Размер train до аугментации: {df.shape[0]}")
    print(f"Размер train после аугментации: {df_augmented.shape[0]}")
    return df_augmented


# ---------------------------------------------------------
# 6. Главная функция пайплайна (без аугментации)
# ---------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("1. Применяем базовую очистку...")
    df['CONTENT_clean'] = df['CONTENT'].apply(clean_text)

    print("2. Исправляем опечатки...")
    df['CONTENT_clean'] = df['CONTENT_clean'].apply(fix_typos)

    print("3. Применяем лемматизацию...")
    df['CONTENT_lemm'] = df['CONTENT_clean'].apply(lemmatize_text)

    print("4. Применяем стемминг (для классики)...")
    df['CONTENT_stem'] = df['CONTENT_clean'].apply(stem_text)

    print("5. Применяем замену сленга (до лемматизации, иначе словарь не сработает)...")
    df['CONTENT_clean'] = df['CONTENT_clean'].apply(replace_slang)
    df['CONTENT_lemm'] = df['CONTENT_lemm'].apply(replace_slang)
    df['CONTENT_stem'] = df['CONTENT_stem'].apply(replace_slang)

    return df


# ---------------------------------------------------------
# 7. Сплит + аугментация только train + сохранение
# ---------------------------------------------------------
def split_and_save_data(
    df: pd.DataFrame,
    test_size: float = 0.10,
    random_state: int = 42,
    augment: bool = True,
):
    print(f"\n6. Делим данные на Train/Test ({1 - test_size}/{test_size})...")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['class'],
    )

    # Проверяем, что редкие классы есть в тесте (для честной оценки)
    rare_in_test = set(test_df['class'].unique()) & {3, 7, 8}
    if not rare_in_test:
        print("ВНИМАНИЕ: в тесте нет редких классов (3/7/8) — macro-F1 по ним не оценить.")

    if augment:
        train_df = augment_minority_classes(train_df, target_classes=(7, 8, 3),
                                            random_state=random_state)

    train_df.to_parquet('../data/train.parquet', index=False)
    test_df.to_parquet('../data/test.parquet', index=False)

    print(f"Train сохранен: {train_df.shape[0]} строк")
    print(f"Test сохранен:  {test_df.shape[0]} строк")
    return train_df, test_df