from torch.utils.data import Dataset
import torch
import pandas as pd


class TextPairsDataset(Dataset):

    def __init__(self, question_list, sentence_list, tokenizer, max_length, device):
        self.len = len(question_list)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.question_list = question_list
        self.sentence_list = sentence_list

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        question = self.question_list[index]
        sentence = self.sentence_list[index]

        inputs = self.tokenizer(
            text=question,
            text_pair=sentence,

            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=False,
            truncation="only_second",
        )

        inputs = {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
        }

        inputs = {k: v.to(self.device) for (k, v) in inputs.items()}

        return inputs


def get_token(question, context, tokenizer):

    inputs = tokenizer(
        text=question,
        text_pair=context,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_token_type_ids=True,
        truncation="only_second",
        #             return_tensors=   'pt'
    )

    return inputs


def get_raw_token(question, context, tokenizer):

    inputs = tokenizer(
        text=question,
        text_pair=context,
        add_special_tokens=True,
        max_length=None,
        padding=False,
        return_token_type_ids=True,
        truncation=False,
        return_offsets_mapping=True
    )

    return inputs


def split_long_token(question, context, raw_token, tokenizer, doc_overlap_length=32):

    # because of specical token
    first_context_end_pos = raw_token['offset_mapping'][511][1]-1
    context_1 = context[:first_context_end_pos]

    sencond_char_start_pos = raw_token['offset_mapping'][511 -
                                                         doc_overlap_length][0]-1

    context_2 = context[sencond_char_start_pos:]

    inputs_1 = get_token(question, context_1, tokenizer)

    inputs_2 = get_token(question, context_2, tokenizer)

    return inputs_1, inputs_2


def duplicate_token(question, context, tokenizer):

    inputs = get_token(question, context, tokenizer)

    return inputs, inputs


def prepare_feature(example, tokenizer, doc_overlap_length):

    context = example['context']
    question = example['question']

    # get raw token
    raw_token = get_raw_token(question, context, tokenizer)

    if len(raw_token['input_ids']) <= 512:

        token_inputs_1, token_inputs_2 = duplicate_token(
            question, context, tokenizer)

    else:
        token_inputs_1, token_inputs_2 = split_long_token(
            question, context, raw_token, tokenizer, doc_overlap_length)

    return token_inputs_1, token_inputs_2


def prepare_dataframe(question, context_list, tokenizer, doc_overlap_length):

    df = pd.DataFrame()
    df['context'] = context_list
    df['question'] = question

    token_pairs_list = df.apply(
        lambda x: prepare_feature(x, tokenizer, doc_overlap_length), axis=1)
    tokens_left = [pair[0] for pair in token_pairs_list]

    tokens_left_df = pd.DataFrame(tokens_left)

    tokens_right = [pair[1] for pair in token_pairs_list]

    tokens_right_df = pd.DataFrame(tokens_right)

    return tokens_left_df, tokens_right_df


class LongTextPairDataSet(Dataset):

    def __init__(self, df_pair_1, df_pair_2, label_list=None, device="cpu"):
        self.len = len(df_pair_1)
        self.df_pair_1 = df_pair_1
        self.df_pair_2 = df_pair_2
        self.label_list = label_list
        self.device = device

    def __getitem__(self, index):
        df_1 = self.df_pair_1.iloc[index]
        df_2 = self.df_pair_2.iloc[index]
        if self.label_list is not None:
            labels = self.label_list[index]

        if isinstance(df_1, pd.core.series.Series):
            pair_dict_1 = df_1.to_dict()
            pair_dict_2 = df_2.to_dict()

        else:
            pair_dict_1 = df_1.to_dict(orient="list")
            pair_dict_2 = df_2.to_dict(orient="list")

        inputs_1 = {k: torch.tensor(v).to(self.device)
                    for k, v in pair_dict_1.items()}

        inputs_2 = {k: torch.tensor(v).to(self.device)
                    for k, v in pair_dict_2.items()}

        if self.label_list is not None:
            return {"token_inputs_1": inputs_1, "token_inputs_2": inputs_2, "labels": torch.tensor(labels).to(self.device)}
        else:
            return {"token_inputs_1": inputs_1, "token_inputs_2": inputs_2}

    def __len__(self):
        return self.len
