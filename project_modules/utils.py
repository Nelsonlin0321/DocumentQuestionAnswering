

import os
from random import random
import pandas as pd
import numpy as np
from project_modules.config import data_config
import json
import random
from werkzeug.utils import secure_filename
from datetime import datetime


def read_qnli_data(file_name, data_dir=data_config.qnli_training_data_path):
    path = os.path.join(data_dir, file_name)
    with open(path, encoding='utf-8-sig') as f:
        text = f.readlines()

    header = text[0].strip().split("\t")
    lines = [line.strip().split("\t") for line in text[1:]]

    df = pd.DataFrame(lines, columns=header)
    return df


def get_qnli_pandas_dataframe():
    qnli_train_df = read_qnli_data("train.tsv")
    qnli_dev_df = read_qnli_data("dev.tsv")
    qnli_train_df['label'] = np.where(
        qnli_train_df['label'] == 'entailment', 1, 0)
    qnli_dev_df['label'] = np.where(qnli_dev_df['label'] == 'entailment', 1, 0)

    return qnli_dev_df, qnli_train_df


def read_document_to_list(document_path):
    with open(document_path, encoding='utf-8-sig') as f:
        document = f.readlines()
        sentence_list = [line.strip()
                         for line in document if len(line.strip()) != 0]
        return sentence_list


def read_document_dict(document_dir):

    document_dict = {}

    for document_file_name in os.listdir(document_dir):
        if document_file_name.endswith(".txt"):
            document_name = document_file_name.replace(
                ".txt", "").replace("_", " ")
            document_path = os.path.join(document_dir, document_file_name)
            document_dict[document_name] = read_document_to_list(document_path)

    return document_dict


def read_json(file_path):
    with open(file_path) as f:
        json_f = json.load(f)
    data = json_f['data']
    return data


def get_random_index(List):
    return random.sample(range(len(List)), 1)[0]


def load_data(data_path, load_impossible_answer=False):

    data = read_json(data_path)

    data_dict = {}
    title_list = []
    context_list = []
    question_list = []
    id_list = []
    answer_text_list = []
    answer_start_list = []
    is_impossible_list = []

    for paragraphs in data:
        title = paragraphs['title']
        context_qas_list = paragraphs['paragraphs']

        for context_qas in context_qas_list:
            context = context_qas['context']
            qas_list = context_qas['qas']

            for qas in qas_list:
                title_list.append(title)
                context_list.append(context)

                is_impossible = qas['is_impossible']
                is_impossible_list.append(is_impossible)

                id_ = qas['id']
                id_list.append(id_)
                question = qas['question']
                question_list.append(question)

                if not is_impossible:
                    answer_list = qas['answers']
                    idx = get_random_index(answer_list)
                    answer_text = answer_list[idx]['text']
                    answer_start = answer_list[idx]['answer_start']

                    answer_text_list.append(answer_text)
                    answer_start_list.append(answer_start)
                else:
                    if load_impossible_answer:
                        answer_list = qas['plausible_answers']
                        idx = get_random_index(answer_list)
                        answer_text = answer_list[idx]['text']
                        answer_start = answer_list[idx]['answer_start']
                        answer_text_list.append(answer_text)
                        answer_start_list.append(answer_start)
                    else:
                        answer_text_list.append("")
                        answer_start_list.append(-1)

    data_dict['id'] = id_list
    data_dict['title'] = title_list
    data_dict['context'] = context_list
    data_dict['question'] = question_list
    data_dict['answer_text'] = answer_text_list
    data_dict['answer_start'] = answer_start_list
    data_dict['is_impossible'] = is_impossible_list

    return data_dict


def get_squad_v2_pandas_dataframe(squad_v2_dir=data_config.squadv2_training_data_path, include_impossible=False, load_impossible_answer=False):
    # download from https://rajpurkar.github.io/SQuAD-explorer/
    train_data_path = os.path.join(squad_v2_dir, "train-v2.0.json")
    dev_data_path = os.path.join(squad_v2_dir, 'dev-v2.0.json')

    train_data_dict = load_data(train_data_path, load_impossible_answer)
    dev_data_dict = load_data(dev_data_path, load_impossible_answer)

    train_data_df = pd.DataFrame(train_data_dict)
    dev_data_df = pd.DataFrame(dev_data_dict)

    if not include_impossible:
        train_data_df = train_data_df[train_data_df['is_impossible'] == False]
        dev_data_df = dev_data_df[dev_data_df['is_impossible'] == False]

    return train_data_df, dev_data_df


def save_model_excel_log(df, type="sentence_selection", save_dir="./models_log/", time_format="%Y_%m_%d_%H_%M_%S_%f"):

    log_question = df['Question'].iloc[0]
    log_time = datetime.today().strftime(time_format)
    log_question = secure_filename(log_question)
    filename = type+"_"+log_question + "_" + log_time + ".xlsx"
    path_name = os.path.join(save_dir, filename)
    df.to_excel(path_name, index=False, encoding='utf-8-sig')
