from torch.utils.data import Dataset
import torch


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
