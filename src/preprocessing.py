# src/preprocessing.py

import re
import pandas as pd
import numpy as np
import pymorphy3
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split

# Скачиваем стоп-слова (если их нет)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Инициализируем морфологический анализатор
morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words('russian'))

# Словарь синонимов и сленга для "редактирования" (Задача 2)
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

# Словарь частых опечаток (Задача 2)
TYPO_DICT = {
    'билогия': 'биология',
    'билогии': 'биология',
    'уважнием': 'уважением',
    'здраствуйте': 'здравствуйте',
    'педаго': 'педагог',
}

# ---------------------------------------------------------
# 1. Базовая очистка текста (Clean Text)
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Удаляет HTML-теги, приветствия, подписи, маскирует цифры.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Убираем HTML-теги <br>, <br/>, <br /> и т.д.
    text = re.sub(r'<br\s*/?>', ' ', text)
    
    # 2. Убираем приветствия в начале (Здравствуйте, Добрый день и т.д.)
    text = re.sub(r'^(Здравствуйте|Добрый день|Доброго дня|Добрый вечер)[!,.]?\s*', '', text, flags=re.IGNORECASE)
    
    # 3. Убираем подписи в конце (Ищем фразу и всё, что после неё до конца текста)
    # Флаг re.DOTALL позволяет точке . захватывать переносы строк
    text = re.sub(r'(С\s+уважением|С\s+уважнием|Спасибо|Благодарю).*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 4. Маскируем цифры и решетки (заменяем на <NUM>)
    text = re.sub(r'#+', ' <NUM> ', text)
    text = re.sub(r'\b\d+\b', ' <NUM> ', text)
    
    # 5. Оставляем только буквы, пробелы и наш токен <NUM>
    text = re.sub(r'[^а-яА-Яa-zA-Z\s<>]', ' ', text)
    
    # 6. Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Приводим к нижнему регистру
    text = text.lower()
    
    return text


# ---------------------------------------------------------
# 2. Лемматизация (для LSTM и Классики)
# ---------------------------------------------------------
def lemmatize_text(text: str) -> str:
    """
    Приводит слова к начальной форме и удаляет стоп-слова, а также мусорные односимвольные токены.
    """
    if not isinstance(text, str):
        return ""
    
    words = text.split()
    lemmas = []
    
    for word in words:
        # Оставляем токен <num> как есть
        if word == '<num>':
            lemmas.append('<num>')
            continue
            
        # Убираем стоп-слова
        if word in russian_stopwords:
            continue
            
        # Лемматизируем
        p = morph.parse(word)[0]
        lemma = p.normal_form
        
        # Убираем мусорные односимвольные токены (например, 'й' от '#-й')
        # Оставляем только токен '<num>', который обработан выше
        if len(lemma) == 1:
            continue
            
        lemmas.append(lemma)
        
    return ' '.join(lemmas)


# ---------------------------------------------------------
# 3. Методы редактирования: Замена опечаток и сленга
# ---------------------------------------------------------
def fix_typos(text: str) -> str:
    """Исправляет частые опечатки по словарю."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    return ' '.join([TYPO_DICT.get(w, w) for w in words])

def replace_slang(text: str) -> str:
    """Заменяет сленг и сокращения на полные слова."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    return ' '.join([SLANG_DICT.get(w, w) for w in words])


# ---------------------------------------------------------
# 4. Главная функция для запуска (Pipeline)
# ---------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет все этапы очистки к датафрейму.
    """
    print("1. Применяем базовую очистку...")
    df['CONTENT_clean'] = df['CONTENT'].apply(clean_text)
    
    print("2. Исправляем опечатки...")
    df['CONTENT_clean'] = df['CONTENT_clean'].apply(fix_typos)
    
    print("3. Применяем лемматизацию (может занять пару минут)...")
    df['CONTENT_lemm'] = df['CONTENT_clean'].apply(lemmatize_text)
    
    print("4. Применяем замену сленга...")
    df['CONTENT_edited'] = df['CONTENT_lemm'].apply(replace_slang)
    
    return df


# ---------------------------------------------------------
# 5. Функция для сплита и сохранения (Артефакт)
# ---------------------------------------------------------
def split_and_save_data(df: pd.DataFrame, test_size: float = 0.10, random_state: int = 42):
    """
    Делит данные на train/test со стратификацией и сохраняет в parquet.
    """
    print(f"\n5. Делим данные на Train/Test ({1-test_size}/{test_size})...")
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['class'] 
    )
    
    # Сохраняем в папку data (относительно корня проекта)
    train_df.to_parquet('../data/train.parquet', index=False)
    test_df.to_parquet('../data/test.parquet', index=False)
    
    print(f"Train сохранен: {train_df.shape[0]} строк")
    print(f"Test сохранен:  {test_df.shape[0]} строк")
    print("Файлы train.parquet и test.parquet готовы")
    
    return train_df, test_df