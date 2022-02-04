from logging import raiseExceptions
import torch
from transformers import AutoTokenizer
from project_modules.datasets import TextPairsDataset
from torch.utils.data import DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AlbertModel
from project_modules.datasets import prepare_dataframe
from project_modules.datasets import LongTextPairDataSet


class SentenceSelectionModelLoader(object):

    def __init__(self, model_config):

        from transformers import AlbertForSequenceClassification as model_structure

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model_structure.from_pretrained(
            model_config.model_path).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)

        self.max_length = model_config.max_length

        self.batch_size = model_config.batch_size

    def predict_from_loader(self, model, data_loader):

        pred_list = []
        prob_list = []

        model.eval()

        for sample in data_loader:

            with torch.no_grad():
                outputs = model(**sample)

                logits = outputs.logits
                probs = torch.sigmoid(logits)

                pred = torch.argmax(logits, axis=1)
                pred = pred.detach().cpu().numpy()
                pred_list.extend(pred)

                prob = probs[:, 1]
                prob = prob.detach().cpu().numpy()
                prob_list.extend(prob)

        return {'pred': pred_list, 'prob': prob_list}

    def predict(self, question, sentence_list):

        if isinstance(question, str):
            question_list = [question] * len(sentence_list)
        elif isinstance(question, list):
            question_list = question
        else:
            raise ValueError("Question requires List of str or str.")

        predict_datasets = TextPairsDataset(
            question_list, sentence_list, self.tokenizer, self.max_length, self.device)

        predict_datasets_loader = DataLoader(
            predict_datasets, batch_size=self.batch_size, shuffle=False)

        return self.predict_from_loader(self.model, predict_datasets_loader)


class AnswerRetrievalModelLoader(object):

    def __init__(self, model_config):

        from transformers import AlbertForQuestionAnswering as model_structure

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model_structure.from_pretrained(
            model_config.model_path).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)

        self.max_length = model_config.max_length

        self.batch_size = model_config.batch_size

        self.max_answer_length = model_config.max_answer_length

        self.sep_token_id = self.tokenizer.sep_token_id

        self.cls_token_id = self.tokenizer.cls_token_id

    def predict_from_loader(self, model, data_loader):

        final_result_list = []

        model.eval()

        for token_inputs in data_loader:

            with torch.no_grad():

                output = model(**token_inputs)

                token_inputs = {k: v.to('cpu')
                                for k, v in token_inputs.items()}

                start_logits = output.start_logits.cpu().detach().numpy()
                end_logits = output.end_logits.cpu().detach().numpy()

                input_ids = token_inputs['input_ids']

                result_dict_list = []

                for idx in range(len(start_logits)):

                    result_dict = {}
                    start_end = (0, 0)
                    start_end_score = (-1, -1)

                    score = -1

                    start_context_id = list(input_ids[idx].detach().numpy()).index(
                        self.sep_token_id) + 2

                    for start, p_start in enumerate(start_logits[idx][start_context_id:]):
                        if p_start > 0:
                            for end, p_end in enumerate(end_logits[idx][start_context_id:]):
                                if p_end > 0:
                                    if end >= start and end < start + self.max_answer_length:
                                        if p_start * p_end > score:
                                            start_end = (start, end)
                                            start_end_score = (p_start, p_end)
                                            score = p_start * p_end

                    start, end = start_end
                    start = start + start_context_id
                    end = end + start_context_id
                    start_score, end_score = start_end_score

                    pred_answer = ""
                    if start != 0 and end != 0:
                        pred_answer = self.tokenizer.decode(
                            input_ids[idx][start:end+1]
                        )

                    result_dict['start_pos'] = start
                    result_dict['start_score'] = start_score
                    result_dict['end_pos'] = end
                    result_dict['end_score'] = end_score

                    result_dict['answer'] = pred_answer
                    result_dict['score'] = score

                    result_dict_list.append(result_dict)

                final_result_list.extend(result_dict_list)

        return final_result_list

    def predict(self, question, sentence_list):

        if isinstance(question, str):
            question_list = [question] * len(sentence_list)
        elif isinstance(question, list):
            question_list = question
        else:
            raise ValueError("Question requires List of str or str.")

        predict_datasets = TextPairsDataset(
            question_list, sentence_list, self.tokenizer, self.max_length, self.device)

        predict_datasets_loader = DataLoader(
            predict_datasets, batch_size=self.batch_size, shuffle=False)

        return self.predict_from_loader(self.model, predict_datasets_loader)


class TwinAlBerts(torch.nn.Module):
    def __init__(self, model_config):

        super(TwinAlBerts, self).__init__()

        self.albert_layer_1 = AlbertModel.from_pretrained(
            model_config.model_name)
        self.albert_layer_2 = AlbertModel.from_pretrained(
            model_config.model_name)

        self.pre_classifier = torch.nn.Linear(768*2, 768)

        self.dropout = torch.nn.Dropout(0.3)

        self.classifer = torch.nn.Linear(768, model_config.num_class)

        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, token_inputs_1, token_inputs_2, labels=None):

        albert_outputs_1 = self.albert_layer_1(**token_inputs_1)
        albert_outputs_2 = self.albert_layer_2(**token_inputs_2)

        pooler_output_1 = albert_outputs_1.pooler_output

        pooler_output_2 = albert_outputs_2.pooler_output

        concat_pooler = torch.cat([pooler_output_1, pooler_output_2], axis=1)

        concat_pooler = self.pre_classifier(concat_pooler)

        concat_pooler = self.dropout(concat_pooler)

        logits = self.classifer(concat_pooler)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class DocumentSelectionModelLoader(object):
    def __init__(self, model_config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = torch.load(model_config.model_path).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_dir)

        self.batch_size = model_config.batch_size

        self.doc_overlap_length = model_config.doc_overlap_length

    def predict_from_loader(self, model, data_loader):
        pred_list = []
        prob_list = []

        model.eval()

        for sample in data_loader:

            with torch.no_grad():
                outputs = model(**sample)

                logits = outputs.logits
                probs = torch.sigmoid(logits)

                pred = torch.argmax(logits, axis=1)
                pred = pred.detach().cpu().numpy()
                pred_list.extend(pred)

                prob = probs[:, 1]
                prob = prob.detach().cpu().numpy()
                prob_list.extend(prob)

        return {'pred': pred_list, 'prob': prob_list}

    def predict(self, question, context_list):
        tokens_left_df, tokens_right_df = prepare_dataframe(
            question, context_list, self.tokenizer, self.doc_overlap_length)

        long_text_pair_dataset = LongTextPairDataSet(
            tokens_left_df, tokens_right_df, device=self.device)

        predict_datasets_loader = DataLoader(
            long_text_pair_dataset, batch_size=self.batch_size, shuffle=False)

        return self.predict_from_loader(self.model, predict_datasets_loader)
