# src/preprocessing.py

import re
import pandas as pd
import numpy as np
import pymorphy3
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
import random

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
        if len(lemma) == 1: # Убираем мусорные 'й'
            continue
        lemmas.append(lemma)
    return ' '.join(lemmas)

# ---------------------------------------------------------
# 3. Стемминг (для Классики / TF-IDF)
# ---------------------------------------------------------
def stem_text(text: str) -> str:
    """Более мягкая обработка, чем лемматизация. Отсекает только окончания."""
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
        # Берем псевдо-основу слова (стемминг)
        stem = morph.parse(word)[0].normal_form[:5] # Грубый стемминг
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
# 5. Аугментация (Размножение редких классов)
# ---------------------------------------------------------
def augment_minority_classes(df: pd.DataFrame, target_classes=[7, 8, 3]) -> pd.DataFrame:
    """
    Берет редкие классы (7-Аспирантура, 8-Регистрация, 3-Достижения)
    и создает их копии с заменой сленга, чтобы увеличить их количество.
    """
    print("Выполняем аугментацию редких классов...")
    augmented_rows = []
    
    for cls in target_classes:
        # Берем все строки этого класса
        class_df = df[df['class'] == cls]
        
        # Создаем 3 копии каждой строки
        for _ in range(3):
            for _, row in class_df.iterrows():
                new_row = row.copy()
                # Меняем текст (применяем замену сленга для создания вариации)
                new_row['CONTENT_edited'] = replace_slang(row['CONTENT_edited'])
                new_row['CONTENT_lemm'] = replace_slang(row['CONTENT_lemm'])
                # Добавляем суффикс к тексту, чтобы модель не выучила его наизусть как дубликат
                new_row['CONTENT_lemm'] = new_row['CONTENT_lemm'] + " вариация"
                augmented_rows.append(new_row)
                
    # Склеиваем старый датафрейм с новыми строками
    df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    print(f"Размер датасета до аугментации: {df.shape[0]}")
    print(f"Размер датасета после аугментации: {df_augmented.shape[0]}")
    return df_augmented

# ---------------------------------------------------------
# 6. Главная функция пайплайна
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
    
    print("5. Применяем замену сленга...")
    df['CONTENT_edited'] = df['CONTENT_lemm'].apply(replace_slang)
    
    # НОВЫЙ ШАГ: Аугментация
    df = augment_minority_classes(df, target_classes=[7, 8, 3])
    
    return df

# ---------------------------------------------------------
# 7. Сплит и сохранение
# ---------------------------------------------------------
def split_and_save_data(df: pd.DataFrame, test_size: float = 0.10, random_state: int = 42):
    print(f"\n6. Делим данные на Train/Test ({1-test_size}/{test_size})...")
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['class'] 
    )
    
    train_df.to_parquet('../data/train.parquet', index=False)
    test_df.to_parquet('../data/test.parquet', index=False)
    
    print(f"Train сохранен: {train_df.shape[0]} строк")
    print(f"Test сохранен:  {test_df.shape[0]} строк")
    return train_df, test_df