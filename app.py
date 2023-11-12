from flask import Flask, render_template, request
from question_generation.pipelines import pipeline
from docx import Document
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline as llama_pipe
import transformers
import torch
import replicate


def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    skip_paragraphs = 64
    for i, paragraph in enumerate(doc.paragraphs):
        if i < skip_paragraphs:
            continue  # Пропустить абзацы в начале документа

        text += paragraph.text + "\n"

    return text


replicate = replicate.Client(api_token='r8_7UzACtSxVhcfgf8yLTmz5AV9Xtpgwoz2r5rsk')

nlp = pipeline("e2e-qg", model="valhalla/t5-base-e2e-qg", use_cuda=False)

result = read_docx(
    '/home/vladislav/python/OORL/question/Stavropol-hackaton/question_generation/23 06 2022 N 250 en.docx')
translator_en = GoogleTranslator(source='en', target='ru')
translator_ru = GoogleTranslator(source='ru', target='en')

app = Flask(__name__)


class SaveResult:
    def __init__(self):
        self.question = None
        self.en_question = None
        self.answer = None
        self.ctx = None


save_res = SaveResult()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_number = request.form.get('number')
        if user_number:
            ctx = result[int(user_number) * 1500: (int(user_number) + 1) * 1500]
            user_question = nlp(ctx)
            save_res.en_question = user_question
            return_question = '/n'.join(list(map(lambda x: translator_en.translate(x), user_question)))
            save_res.ctx = ctx
            save_res.question = return_question
        user_answer = request.form.get('user_answer')
        if user_answer:
            res = list()
            for ans, question in zip(user_answer.split('/n'), save_res.en_question):
                ans_en = translator_ru.translate(ans)
                prompt = f'Do you think this is the right answer to the question. Tell me only yes or no' \
                         f' Context: {save_res.ctx}, \n question: {question}, \n answer: {ans_en}'
                print(prompt)
                output = replicate.run(
                    "meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
                    input={"prompt": prompt,
                           "max_new_tokens": 2})
                output = list(output)
                print(output)
                res.append(output[-1])
            save_res.answer = '\n'.join(res)

        if save_res.question and save_res.answer:
            return render_template('index.html', user_number=save_res.question, user_answer=save_res.answer)
        elif save_res.question:
            return render_template('index.html', user_number=save_res.question)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
