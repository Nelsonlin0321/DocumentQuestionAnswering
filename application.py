from flask import Flask, render_template, url_for, redirect, request
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from utils import utils
from project_modules.config import flask_config, project_config
from project_modules.models import SentenceSelectionModelLoader
from project_modules.config import sentence_selection_model_config
from project_modules.models import AnswerRetrievalModelLoader
from project_modules.config import answer_retrieval_model_config
from project_modules.utils import read_document_dict
from project_modules.predictions import main_predict


application = Flask(__name__)

"""CONFIG"""
LOG_DIR = project_config.log_dir
ALLOWED_EXTENSIONS = project_config.allowed_extensions
UPLOAD_FOLDER = project_config.upload_folder
TEMPLATE_AUTO_RELOAD = flask_config.template_auto_reload
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['TEMPLATE_AUTO_RELOAD'] = TEMPLATE_AUTO_RELOAD
time_format = "%Y_%m_%d_%H_%M_%S_%f"
answer_retrieval_threshold = answer_retrieval_model_config.answer_retrieval_threshold

"""GLOBAL VARIABLES"""
qa_dict_list = [

    {'document': "",
     'question': '',
     'answer': ''},
]
sentence_selection_model = SentenceSelectionModelLoader(
    sentence_selection_model_config)
document_dict = read_document_dict(UPLOAD_FOLDER)
answer_retrieval_model = AnswerRetrievalModelLoader(
    answer_retrieval_model_config)


@application.route('/', methods=['GET', 'POST'])
def home_page():

    global qa_dict_list
    uploaded_document_list = utils.get_uploaded_documents(
        UPLOAD_FOLDER, ALLOWED_EXTENSIONS)

    utils.write_document_list(UPLOAD_FOLDER, ALLOWED_EXTENSIONS)

    if request.method == "GET":

        context = {'qa_dict_list': qa_dict_list,
                   'uploaded_document_list': uploaded_document_list}

        return render_template("index.html", **context)

    elif request.method == "POST":

        """logic to deal with upload file"""
        file = request.files['file'] if "file" in request.files else None
        if file and utils.allowed_file(file.filename, ALLOWED_EXTENSIONS):
            filename = secure_filename(file.filename)
            file.save(os.path.join(
                UPLOAD_FOLDER, filename))
            return redirect(url_for('home_page'))

        """logic to deal with thumsup or down button"""

        post_dict = request.values.to_dict()

        if "feedback" in post_dict:
            log_time = datetime.today().strftime(time_format)
            post_dict['log_time'] = log_time
            print(post_dict)
            utils.save_log(post_dict, LOG_DIR)
            return redirect(url_for('home_page'))

        question = request.form.get('question')
        document = request.form.get('document')

        print("Input document:", document)

        print("Input question:", question)
        if question:
            document, answer = main_predict(
                question, document, document_dict, sentence_selection_model, answer_retrieval_model, answer_retrieval_threshold)

            answer = answer.capitalize()

            new_record = {
                'document': document,
                'question': question,
                'answer': answer}

            if qa_dict_list[0]['document'] == "":
                qa_dict_list[0] = new_record
            else:
                qa_dict_list.append(new_record)

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
