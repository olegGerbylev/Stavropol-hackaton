from flask import Flask, render_template, request
from question_generation.pipelines import pipeline
from docx import Document
from deep_translator import GoogleTranslator


def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    skip_paragraphs = 64
    for i, paragraph in enumerate(doc.paragraphs):
        if i < skip_paragraphs:
            continue  # Пропустить абзацы в начале документа

        text += paragraph.text + "\n"

    return text


nlp = pipeline("e2e-qg", model="valhalla/t5-base-e2e-qg", use_cuda=False)

result = read_docx('question_generation/23 06 2022 N 250 en.docx')
translator_en = GoogleTranslator(source='en', target='ru')
translator_ru = GoogleTranslator(source='ru', target='en')

app = Flask(__name__)


class SaveResult:
    def __init__(self):
        self.question = None
        self.answer = None


save_res = SaveResult()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_number = request.form.get('number')
        if user_number:
            user_question = nlp(result[int(user_number) * 1500: (int(user_number) + 1) * 1500])
            return_question = '/n'.join(list(map(lambda x: translator_en.translate(x), user_question)))
            save_res.question = return_question
        user_answer = request.form.get('user_answer')
        if user_answer:
            return_answer = translator_ru.translate(user_answer)
            save_res.answer = return_answer

        if save_res.question and save_res.answer:
            return render_template('index.html', user_number=save_res.question, user_answer=save_res.answer)
        elif save_res.question:
            return render_template('index.html', user_number=save_res.question)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
