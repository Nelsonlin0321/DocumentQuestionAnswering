from flask import Flask, render_template, url_for, redirect, request
from werkzeug.utils import secure_filename
import os
import time

application = Flask(__name__)

qa_dict_list = [

    {'document': "second document",
     'question': 'second question',
     'answer': 'second answer'},

    {'document': "first document",
     'question': 'first question',
     'answer': 'first answer'},
]


UPLOAD_FOLDER = './static/uploaded_documents'
ALLOWED_EXTENSIONS = ['txt']
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['TEMPLATE_AUTO_RELOAD'] = True


def predict_answer(document, question):
    time.sleep(2)
    return "predict_document", "predict_answer"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_uploaded_documents():
    file_list = os.listdir(UPLOAD_FOLDER)
    return [file for file in file_list if file.split(".")[1] in ALLOWED_EXTENSIONS]


def write_document_list():
    file_list = get_uploaded_documents()
    document_names = [file.split(".")[0].replace(
        "_", " ").strip() for file in file_list]
    write_out_path = "./static/document_list.txt"
    with open(write_out_path, mode="w") as f:
        for name in document_names:
            f.write(name)
            f.write("\n")


@application.route('/', methods=['GET', 'POST'])
def home_page():

    global qa_dict_list
    uploaded_document_list = get_uploaded_documents()
    write_document_list()

    if request.method == "GET":

        context = {'qa_dict_list': qa_dict_list,
                   'uploaded_document_list': uploaded_document_list}

        return render_template("index.html", **context)

    elif request.method == "POST":

        file = request.files['file'] if "file" in request.files else None

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(
                application.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('home_page'))

        question = request.form.get('question')
        document = request.form.get('document')

        print(question)
        if question:
            document, answer = predict_answer(document, question)

            new_record = {
                'document': document,
                'question': question,
                'answer': answer}

            if qa_dict_list[0]['document'] == "":
                qa_dict_list[0] = new_record
            else:
                qa_dict_list.applicationend(new_record)
            print(qa_dict_list)

        context = {'qa_dict_list': qa_dict_list,
                   'uploaded_document_list': uploaded_document_list}

        return render_template("index.html", **context)


@application.route('/clear')
def clear_history():
    global qa_dict_list
    qa_dict_list = [

        {'document': "",
         'question': "",
         'answer': ""},
    ]

    return redirect(url_for("home_page"))


if __name__ == '__main__':
    application.run(use_reloader=True, debug=False)