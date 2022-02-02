import os
from werkzeug.utils import secure_filename
import json


def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_uploaded_documents(UPLOAD_FOLDER, ALLOWED_EXTENSIONS):
    file_list = os.listdir(UPLOAD_FOLDER)
    return [file for file in file_list if file.split(".")[1] in ALLOWED_EXTENSIONS]


def write_document_list(UPLOAD_FOLDER, ALLOWED_EXTENSIONS):
    file_list = get_uploaded_documents(UPLOAD_FOLDER,ALLOWED_EXTENSIONS)
    document_names = [file.split(".")[0].replace(
        "_", " ").strip() for file in file_list]
    write_out_path = "./static/document_list.txt"
    with open(write_out_path, mode="w") as f:
        for name in document_names:
            f.write(name)
            f.write("\n")


def save_log(log_dict, save_dir="./log"):
    log_question = log_dict['log_question']
    log_time = log_dict["log_time"]
    log_question = secure_filename(log_question)
    filename = log_question + "_" + log_time + ".json"
    path_name = os.path.join(save_dir, filename)
    with open(path_name, 'w') as f:
        json.dump(log_dict, f)
