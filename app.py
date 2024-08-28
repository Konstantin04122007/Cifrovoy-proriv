from flask import Flask, request, render_template
import pytesseract
import cv2
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk import pos_tag

app = Flask(__name__)

# Путь к Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'"C:\Users\User\Downloads\tesseract-ocr-w64-setup-5.4.0.20240606.exe"'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Сохранение файла
    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)

    # Обработка изображения
    text = process_image(filepath)

    # Анализ текста
    analysis_result = analyze_text(text)

    return render_template('index.html', text=text, analysis=analysis_result)


def process_image(filepath):
    # Чтение изображения
    img = cv2.imread(filepath)
    # Преобразование в серый цвет
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Применение OCR
    text = pytesseract.image_to_string(gray)
    return text


def analyze_text(text):
    # Пример анализа текста
    # Здесь можно использовать NLTK
    age_recommendation = extract_age(text)
    marketing_claims = extract_marketing_claims(text)
    return f"Минимальный целевой возраст: {age_recommendation}, Маркетинговые декларации: {marketing_claims}"

def extract_age(text):
    # Токенизация текста
    tokens = word_tokenize(text)

    # Удаление стоп-слов и знаков препинания
    russian_stopwords = stopwords.words("russian")
    table = str.maketrans('', '', punctuation)
    filtered_tokens = [token.translate(table) for token in tokens if
                       token.lower() not in russian_stopwords and token.strip() not in punctuation]

    # Поиск чисел и возрастных указаний
    def find_age_indication(tokens):
        age_indications = ["лет", "года", "год"]
        for token in tokens:
            if token.isdigit():
                for indication in age_indications:
                    if indication in tokens[tokens.index(token) + 1]:
                        return token
        return None

    age = find_age_indication(filtered_tokens)

    # Валидация возраста
    def validate_age(age):
        if age and age.isdigit() and 0 < int(age) < 100:
            return int(age)
        return None

    min_age = validate_age(age)
    return min_age


def extract_marketing_claims(text):
    # Токенизация текста
    tokens = word_tokenize(text)

    # Удаление стоп-слов и знаков препинания
    russian_stopwords = stopwords.words("russian")
    table = str.maketrans('', '', punctuation)
    filtered_tokens = [token.translate(table) for token in tokens if
                       token.lower() not in russian_stopwords and token.strip() not in punctuation]

    # Частеречная разметка
    tagged_tokens = pos_tag(filtered_tokens)

    # Поиск маркетинговых деклараций
    def find_marketing_claims(tagged_tokens):
        marketing_claims = []
        for token, tag in tagged_tokens:
            if tag in ["ADJF", "ADJS", "VERB", "NOUN"] and token.lower() not in russian_stopwords:
                marketing_claims.append(token)
        return marketing_claims

    claims = find_marketing_claims(tagged_tokens)

    # Объединение ключевых фраз
    def combine_claims(claims, tokens):
        combined_claims = []
        claim = ""
        for token in tokens:
            if token in claims:
                claim += token + " "
            elif claim:
                combined_claims.append(claim.strip())
                claim = ""
        if claim:
            combined_claims.append(claim.strip())
        return combined_claims

    combined_claims = combine_claims(claims, filtered_tokens)
    return combined_claims

if __name__ == '__main__':
    app.run(debug=True)
