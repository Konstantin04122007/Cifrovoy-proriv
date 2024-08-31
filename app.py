import requests
from flask import Flask, request, render_template
import pytesseract
import cv2 as cv
import os
import nltk
import string
from pymorphy3 import MorphAnalyzer
import numpy as np
import easyocr


app = Flask(__name__)


catalog = ''
identifier = ''
apikey = ''


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

    # Анализ текста
    analysis_result = analyze_text(filepath, True)

    return render_template('index.html', analysis=analysis_result)


@app.route('/upload_back', methods=['POST'])
def upload_file_back():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Сохранение файла
    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)

    # Анализ текста
    analysis_result = analyze_text(filepath, False)

    return render_template('index.html', analysis=analysis_result)


def extract_text_from_image(processed_image):
    """Извлекает текст из предварительно обработанного изображения."""
    custom_config = '--psm 4 -c tessedit_char_whitelist= .,абвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789'
    text = pytesseract.image_to_string(processed_image, lang='rus', config=custom_config)
    return text


def classificator_response(text, catalog, identifier, apikey):
    print(text)
    headers = {
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {apikey}",
                "x-folder-id": identifier
            }
    
    text = text_cleaner(text)

    prompt_text = {
            "modelUri": f"cls://{catalog}/yandexgpt/latest",
            "taskDescription": "определить есть ли 1",
            "labels": [
                "1",
                "0",
            ],
            "text": text,
            "samples": [{"text": "детск каш без пальмов масл c b1 омега3 " , "label": "1" },
                {"text": "детск каш рисов каш гипоаллерген безмолочн рост бифидобактер без пальмов масл c b1 омега3 " , "label": "1" },
                {"text": "детск каш кукурузн каш безмолочн рост без пальмов масл c b1 омега3 " , "label": "1" },
                {"text": "детск каш каш безмолочн рост без пальмов масл c b1 омега3 " , "label": "1" },
                {"text": "каш безмолочн сахар " , "label": "1" },
                {"text": "кашк молочн 100 natural сахар " , "label": "1" },
                {"text": "100 natural каш молочн " , "label": "1" },
                {"text": "рисов каш тройн польз 100 залк 12 " , "label": "1" },
                {"text": "гречнев каш тройн польз 100 залк 12 гипоаллерген клиническ доказа " , "label": "1" },
                {"text": "рисвов кашк груш белк энерг омег 3 развит интеллект гипоаллерген клиническ доказа " , "label": "1" },
                {"text": "гречнев каш тройн польз 100 залк 12 гипоаллерген клиническ доказа " , "label": "1" },
                {"text": "кашк молочн 100 natural сахар " , "label": "1" },
                {"text": "каш 100 natural добавл " , "label": "1" },
                {"text": "детск каш гипоаллерген безмолочн рост бифидобактер без пальмов без пальмов масл c b1 омега3 " , "label": "1" },
                {"text": "детск каш рисов каш гипоаллерген безмолочн рост бифидобактер без пальмов масл c b1 омега3 " , "label": "1" },
                {"text": "детск каш кукурузн каш безмолочн рост бифидобактер без пальмов масл c b1 омега3 " , "label": "1" },
                {"text": "детск каш каш безмолочн рост бифидобактер без пальмов масл c b1 омега3 " , "label": "1" },
                {"text": "каш безмолочн добавлен сахар " , "label": "1" },
                {"text": "кашк молочн 100 natural добавлен сахар " , "label": "1" },
                {"text": "100 natural каш молочн " , "label": "1" },
                {"text": "кашк молочн 100 natural добавлен сахар " , "label": "1" },
                {"text": "каш яблок банан безмолочн поддержк иммунитет рост " , "label": "1" },
                {"text": "гречнев каш гипоаллерген безмолочн дополнительн витамин минерал " , "label": "1" },
                {"text": "пшен каш тыкв морков добавлен сахар подход дет аллерг " , "label": "1" },
                {"text": "ячнев каш черник безмолочн бифидобактер укреплен иммунитет " , "label": "1" },
                {"text": "каш четырех злак малин добавлен сахар вкус " , "label": "1" },
                {"text": "каш мультизлаков чернослив глют гипоаллерген дополнительн витамин " , "label": "1" },
                {"text": "каш персик добавлен сахар подход " , "label": "1" },
                {"text": "детск каш кукурузн каш безмолочн " , "label": "0" },
                {"text": "детск каш каш безмолочн " , "label": "0" },
                {"text": "каш безмолочн " , "label": "0" },
                {"text": "детск каш каш " , "label": "0" },
                {"text": "детск каш рисов каш безмолочн " , "label": "0" },
                {"text": "детск каш гречнев каш " , "label": "0" },
                {"text": "детск каш кукурузн каш безмолочн " , "label": "0" },
                {"text": "детск каш гречнев каш " , "label": "0" },
                {"text": "детск каш мультизлаков каш пят злак безмолочн " , "label": "0" },
                {"text": "каш молочн " , "label": "0" },
                {"text": "детск каш рисов каш молочн " , "label": "0" },
                {"text": "детск каш рисов каш яблок молочн " , "label": "0" },
                {"text": "детск каш гречнев каш кураг молочн " , "label": "0" },
                {"text": "детск каш гречнев каш ягод молочн " , "label": "0" },
                {"text": "детск каш гречнев каш банан безмолочн " , "label": "0" },
                {"text": "детск каш рисов каш банан молочн " , "label": "0" },
                {"text": "детск каш пшеничн каш тыкв молочн " , "label": "0" },
                {"text": "детск каш пшеничн каш тыкв безмолочн " , "label": "0" },
                {"text": "детск каш каш яблок молочн " , "label": "0" },
                {"text": "детск каш гречнев каш молочн " , "label": "0" },
                {"text": "детск каш кукурузн каш молочн " , "label": "0" },
                {"text": "детск каш гречнев каш молочн " , "label": "0" },
                {"text": "детск каш гречнев каш груш банан молочн " , "label": "0" },
                {"text": "детск каш гречнев каш яблок банан молочн " , "label": "0" },
                {"text": "детск каш мультизлаков каш пят злак молочн " , "label": "0" },
                {"text": "детск каш мультизлаков каш груш персик молочн " , "label": "0" },
                {"text": "детск каш мультизлаков каш яблок черник малин молочн " , "label": "0" },
                {"text": "яблок " , "label": "0" },]
            }
    response = requests.post(
        'https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification',
        json=prompt_text, headers=headers
    )
    predictions = response.json()['predictions']
    result = max(predictions, key=lambda x: x['confidence'])['label']
    return result


def text_cleaner(text):
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower()
    # Токенизация текста
    tokens = nltk.tokenize.word_tokenize(text)

    # Удаление стоп-слов и знаков препинания
    russian_stopwords = nltk.corpus.stopwords.words("russian")
    additional_punctuation = '—«».“”№‹›…\\-®`\'’‘'
    table = str.maketrans('', '', string.punctuation)
    table = str.maketrans('', '', additional_punctuation)
    filtered_tokens = [token.translate(table) for token in tokens if
                    token.lower() not in russian_stopwords and token.strip() not in string.punctuation]


    morph = MorphAnalyzer()
    def is_meaningful(word):
        # Проверка на существование в русском языке через pymorphy3
        if word.isdigit():
            return True
        parsed_word = morph.parse(word.lower())[0]
        if (parsed_word.tag.POS == 'NOUN' or parsed_word.tag.POS == 'ADJF' or parsed_word.tag.POS == 'ADJS' or parsed_word.tag.POS == 'PREP') and len(word) > 1 and word.isalpha():
            return True
        return False
    
    # Стемминг
    filtered_tokens = [word for word in filtered_tokens if is_meaningful(word)]
    snowball = nltk.stem.SnowballStemmer(language="russian")
    stemmed_tokens = [snowball.stem(token) for token in filtered_tokens]
    def combine_tokens(tokens):
        res = ''
        for token in tokens:
            res += token + ' '
        return res
    return combine_tokens(stemmed_tokens)


def age_classificator(text):
    text = text_cleaner(text)
    key = text.find('месяц') - 2
    age = int(text[key])
    
    if age < 6:
        return 'Обнаружено заявление об отказе от рекомендаций общественного здравоохранения'
    else:
        return 'Недопустимые заявления не обнаружены'


def analyze_text(image_path, front:bool):
    processed_image = process_image(image_path)
    if front:
        reader = easyocr.Reader(['en', 'ru'])
        extracted_text = recognize_text(processed_image, reader)
        res = classificator_response(extracted_text, catalog, identifier, apikey)
        if res == '0':
            return 'Маркетинговые фразы не обнаружены'
        return 'Обнаружены маркетинговые фразы'
    else:
        extracted_text = extract_text_from_image(processed_image)
        return age_classificator(extracted_text)
    


def initialize_reader(languages=['en', 'ru']):
    return easyocr.Reader(languages)

def process_image(image_path):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    retval, binary = cv.threshold(resized, 160, 255, cv.THRESH_BINARY)
    denoised = cv.GaussianBlur(binary, (3, 3), 0)
    return denoised

def recognize_text(image, reader):
    results = reader.readtext(image, detail=0)
    return results


if __name__ == '__main__':
    app.run(debug=True)