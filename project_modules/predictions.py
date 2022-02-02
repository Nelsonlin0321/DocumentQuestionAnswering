from sklearn.utils import shuffle
import pandas as pd
from project_modules.config import project_config
from project_modules.utils import save_model_excel_log


def predict_document(question, document_dict, sentence_selection_model):

    document_dict_items = shuffle(list(document_dict.items()))

    for (document, sentence_list) in document_dict_items:

        sentence_selection_dict = sentence_selection_model.predict(
            question, sentence_list)

        pred = max(sentence_selection_dict['pred'])
        prob = max(sentence_selection_dict['prob'])

        pred_df = pd.DataFrame()
        pred_df['Document'] = [document] * len(sentence_list)
        pred_df['Sentence'] = sentence_list
        pred_df['Question'] = question
        pred_df['Prediction'] = sentence_selection_dict['pred']
        pred_df['Probability'] = sentence_selection_dict['prob']

        save_model_excel_log(pred_df, type="predict_document",
                             save_dir=project_config.model_log_dir)

        if pred == 1:
            return document, pred_df

    return None, None


def predict_sentence(question, document, sentence_list, sentence_selection_model):

    sentence_selection_dict = sentence_selection_model.predict(
        question, sentence_list)

    pred_df = pd.DataFrame()
    pred_df['Document'] = [document] * len(sentence_list)
    pred_df['Sentence'] = sentence_list
    pred_df['Question'] = question
    pred_df['Prediction'] = sentence_selection_dict['pred']
    pred_df['Probability'] = sentence_selection_dict['prob']

    save_model_excel_log(pred_df, type="predict_sentence",
                         save_dir=project_config.model_log_dir)

    return pred_df


def predict_answer(question, pred_df, answer_retrieval_model, answer_retrieval_threshold):

    sentence_list = pred_df['Sentence'].to_list()

    retrieval_answer_list = answer_retrieval_model.predict(
        question, sentence_list)

    pred_df['Retrieval_Score'] = [dict_['score']
                                  for dict_ in retrieval_answer_list]
    pred_df['Retrieval_Answer'] = [dict_['answer']
                                   for dict_ in retrieval_answer_list]

    save_model_excel_log(pred_df, type="retrieve_answer",
                         save_dir=project_config.model_log_dir)

    retrieval_pred_df = pred_df[pred_df['Retrieval_Score']
                                > answer_retrieval_threshold]

    if len(retrieval_pred_df) > 0:
        max_id = retrieval_pred_df['Retrieval_Score'].idxmax()
        retrieval_answer = retrieval_pred_df['Retrieval_Answer'].loc[max_id]
        return retrieval_answer
    else:
        max_id = pred_df['Probability'].idxmax()
        sentence_answer = pred_df['Retrieval_Answer'].loc[max_id]
        return sentence_answer


def main_predict(question, document, document_dict, sentence_selection_model, answer_retrieval_model, answer_retrieval_threshold=15):

    default_document = "Unknow Document"
    default_answer = "Sorry, we couldn't find any document related to your question. "
    unfound_answer = "Sorry, we couldn't find any answer for this question in this document: {}"

    if document not in document_dict:
        print("INFO: Input document not in document dictionary!")
        pred_doc, pred_df = predict_document(
            question, document_dict, sentence_selection_model)

        if pred_doc is None or pred_df is None:
            return default_document, default_answer

        else:
            """predict the answer"""
            pred_df = pred_df[pred_df['Prediction'] == 1]
            pred_answer = predict_answer(
                question, pred_df, answer_retrieval_model, answer_retrieval_threshold)
            return pred_doc, pred_answer

    else:
        sentence_list = document_dict[document]

        pred_df = predict_sentence(
            question, document, sentence_list, sentence_selection_model)

        pred_df = pred_df[pred_df['Prediction'] == 1]

        if len(pred_df) == 0:
            return document, unfound_answer.format(document)

        pred_answer = predict_answer(
            question, pred_df, answer_retrieval_model, answer_retrieval_threshold)

        return document, pred_answer
