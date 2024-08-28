from flask import Flask, request, render_template
import pytesseract
import cv2
import os

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
    # Логика для извлечения минимального целевого возраста
    return 0  # Заменить на реальную логику

def extract_marketing_claims(text):
    # Логика для извлечения маркетинговых деклараций
    return "Декларации"  # Заменить на реальную логику

if __name__ == '__main__':
    app.run(debug=True)
