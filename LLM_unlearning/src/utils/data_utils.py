from typing import List, Dict
from collections import defaultdict

from transformers import PreTrainedTokenizer, BatchEncoding
from datasets import Dataset
import torch
import copy

class DefaultDataCollator:
    def __call__(self, batch_examples: List) -> Dict:
        batch_rslt = defaultdict(list)

        for example in batch_examples:
            for key in example:
                batch_rslt[key].append(example[key])

        return batch_rslt


class DataCollatorWithPadding:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding="longest",
        max_length=None,
        pad_to_multiple_of=None,
        return_attention_mask=True,
        return_tensors="pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask
        self.return_tensors = return_tensors

    def __call__(self, batch_examples: List) -> Dict:
        batch_rslt = defaultdict(list)
        for example in batch_examples:
            for key in example:
                if key not in self.tokenizer.model_input_names:
                    batch_rslt[key].append(example[key])

        features = []
        for example in batch_examples:
            features.append(
                BatchEncoding({k: example[k] for k in self.tokenizer.model_input_names})
            )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors=self.return_tensors,
            verbose=True,
        )

        batch_rslt.update(features)
        return batch_rslt

class ReplaceDataset(Dataset):
    def __init__(self, harmful_sas, replace_sas, tokenizer, max_words=100, pad=True):
        self.ann = {
            "prompt":[],
            "harm_response":[],
            "replace_response":[],
            }
        for prompt, responses in harmful_sas.items():
            for i in range(len(responses)):
                this_harm_resp = responses[i]
                this_rpl_resp = replace_sas[prompt][i]
                self.ann["prompt"].append(prompt)
                self.ann["harm_response"].append(this_harm_resp)
                self.ann["replace_response"].append(this_rpl_resp)
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

    def __len__(self):
        return len(self.ann["prompt"])

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        examples = []
        labels = []
        example_masks = []
        for i in index:
            response = self.ann["replace_response"][i]
            input_id = torch.tensor(
                self.tokenizer.encode(response), dtype=torch.int64
            )
            
            if self.pad:
                padding = self.max_words - input_id.shape[0]
                if padding > 0:
                    input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    input_id = input_id[: self.max_words]

            label_id = copy.deepcopy(input_id)
            label_mask = label_id.ge(0)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            label_id[~label_mask] = IGNORE_INDEX
            att_mask = att_mask.float()
            label_mask = label_mask.float()
            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)
        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask":example_masks,
        }

class KGReplaceDataset(Dataset):
    def __init__(self, replace_sas, tokenizer, max_words=100, pad=True):
        self.ann = {
            "prompt":[],
            "replace_response":[],
            }
        for prompt, responses in replace_sas.items():
            for i in range(len(responses)):
                this_rpl_resp = responses[i]
                self.ann["prompt"].append(prompt)
                self.ann["replace_response"].append(this_rpl_resp)
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad
    
    def update(self, replace_sas):
        for prompt, responses in replace_sas.items():
            for i in range(len(responses)):
                this_rpl_resp = responses[i]
                self.ann["prompt"].append(prompt)
                self.ann["replace_response"].append(this_rpl_resp)
    
    def __len__(self):
        return len(self.ann["prompt"])
    
    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        examples = []
        labels = []
        example_masks = []
        for i in index:
            response = self.ann["replace_response"][i]
            input_id = torch.tensor(
                self.tokenizer.encode(response), dtype=torch.int64
            )
            
            if self.pad:
                padding = self.max_words - input_id.shape[0]
                if padding > 0:
                    input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    input_id = input_id[: self.max_words]

            label_id = copy.deepcopy(input_id)
            label_mask = label_id.ge(0)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            label_id[~label_mask] = IGNORE_INDEX
            att_mask = att_mask.float()
            label_mask = label_mask.float()
            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)
        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask":example_masks,
        }

class KGGradAscDataset(Dataset):
    def __init__(self, harmful_sas, tokenizer, max_words=100, pad=True):
        self.ann = {
            "prompt":[],
            "prefix":[],
            "target":[]
            }
        for prompt, responses in harmful_sas.items():
            for i in range(len(responses)):
                prefix, target = responses[i]
                self.ann["prompt"].append(prompt)
                self.ann["prefix"].append(prefix)
                self.ann["target"].append(target)
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

    def pad_token(self, input_id):
        padding = self.max_words - input_id.shape[0]
        if padding > 0:
            input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input_id = input_id[: self.max_words]
        return input_id

    def __len__(self):
        return len(self.ann["prompt"])

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        examples = []
        labels = []
        example_masks = []
        for i in index:
            prefix = self.ann["prefix"][i]
            target = self.ann["target"][i]
            prefix_id = torch.tensor(
                self.tokenizer.encode(prefix), dtype=torch.int64
            )
            input_id = torch.tensor(
                self.tokenizer.encode(target), dtype=torch.int64
            )
            label_id = copy.deepcopy(input_id)
            label_id[: len(prefix_id)] = -1
            
            if self.pad:
                input_id = self.pad_token(input_id)
                label_id = self.pad_token(label_id)
            
            label_mask = label_id.ge(0)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            label_id[~label_mask] = IGNORE_INDEX
            att_mask = att_mask.float()
            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)
        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask":example_masks,
        }