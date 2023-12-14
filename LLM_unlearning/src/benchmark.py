import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    LlamaForSequenceClassification,
    LlamaTokenizer,
)
from datasets import Dataset
from transformers import default_data_collator
from param import str2bool
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import json
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
import argparse

def get_dataset(data_name):
    if data_name == "Rowan/hellaswag":
        dataset = load_dataset("Rowan/hellaswag")
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        return train_dataset, val_dataset
    elif data_name == "glue_mrpc":
        dataset = load_dataset("glue", "mrpc")
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        return train_dataset, val_dataset
    elif data_name == "glue_rte":
        dataset = load_dataset("glue", "rte")
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        return train_dataset, val_dataset
    elif data_name == "glue_cola":
        dataset = load_dataset("glue", "cola")
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        return train_dataset, val_dataset
    elif data_name == "glue_sst2":
        dataset = load_dataset("glue", "sst2")
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        return train_dataset, val_dataset

def process_dataset(dataset, data_name):
    if data_name == "Rowan/hellaswag":
        ctxs = dataset['ctx']
        ctx_bs = dataset['ctx_b']
        endings = dataset['endings']
        labels = dataset['label']
        return {'context': ctxs, 'context_b': ctx_bs, 
                'ending': endings, 'label': labels}
    elif data_name in ("glue_mrpc", "glue_rte"):
        sentence1 = dataset['sentence1']
        sentence2 = dataset['sentence2']
        labels = dataset['label']
        return {'sentence1': sentence1, 'sentence2': sentence2, 
                'label': labels}

    elif data_name in ("glue_cola", "glue_sst2"):
        sentence = dataset['sentence']
        labels = dataset['label']
        return {'sentence': sentence, 
                'label': labels}

def get_response(output_ids, attention_masks, tokenizer):
    temp = []
    for i in range(len(output_ids)):
        this_output = output_ids[i]
        this_mask = attention_masks[i]
        input_len = this_mask.sum()
        this_output = this_output[input_len:]
        temp.append(this_output)
    output_ids = temp
    
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # extract number from responses
    predictions = []
    for resp in responses:
        this_pred = re.findall(r'\d+', resp)[0]
        predictions.append(int(this_pred))
    return predictions

class UniverseDataset(Dataset):
    def __init__(self, input_dataset, tokenizer, args, max_words=512):
        self.inputs = process_dataset(input_dataset, args.data_name)
        self.tokenizer = tokenizer
        self.max_words = max_words
    
    def __len__(self):
        return len(self.inputs['label'])
    
    def create_prompt(self, index):
        raise NotImplementedError

    def pad_tokens(self, input_id):
        padding = self.max_words - input_id.shape[0]
        if padding > 0:
            input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input_id = input_id[: self.max_words]

        return input_id
    
    def __getitem__(self, index):
        # IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        examples = []
        labels = []
        example_masks = []
        
        for i in index:
            ## create input prompt
            prompt = self.create_prompt(i)
            # string to token
            input_id = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            input_id = self.pad_tokens(input_id)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            example_masks.append(att_mask)
            examples.append(input_id)
            label = int(self.inputs["label"][i])
            labels.append(label)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask":example_masks,
        }


class HswDataset(UniverseDataset):
    def prompt_template(self, ctx, endings):
        template = (
            # "Given a sentence, please provide a next sentence prediction among 4 choices. "
                    # 'Strictly response in the format of "Choice #number". '
                    # "Examples:"
                    # "Context: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then\n"
                    # 'Choice 0: ", the man adds wax to the windshield and cuts it."'
                    # 'Choice 1: ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled."'
                    # 'Choice 2: ", the man puts on a christmas coat, knitted with netting."'
                    # 'Choice 3: ", the man continues removing the snow on his car." ]\n'
                    # 'Prediction of choice: 3\n'
                    "Context: \"{ctx}\" \n"
                    # "which one of the four choices should be the next sentence: \n"
                    'A: {ending0} \n'
                    'B: {ending1} \n'
                    'C: {ending2} \n'
                    'D: {ending3} '
                    # "Please select the predicted sentence among the four choices."
                    )
        return template.format(ctx=ctx, ending0=endings[0], ending1=endings[1], ending2=endings[2], ending3=endings[3])
    
    def create_prompt(self, index):
        ctx = self.inputs["context"][index]
        endings = self.inputs["ending"][index]
        # create complete ending
        context_b = self.inputs['context_b'][index]
        temp = []
        for ending in endings:
            temp.append(context_b+ending)
        endings = temp
        prompt = self.prompt_template(ctx, endings)
        return prompt

class PairDataset(UniverseDataset):    
    def create_prompt(self, index):
        sentence1 = self.inputs["sentence1"][index]
        sentence2 = self.inputs["sentence2"][index]
        prompt = f'{sentence1} <s> {sentence2}'
        return prompt

class ClsDataset(UniverseDataset):    
    def create_prompt(self, index):
        prompt = self.inputs["sentence"][index]
        return prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        default="glue_sst2",
        help="Name of dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Name of llm model"
    )
    parser.add_argument(
        "--unlearn_model",
        type=str,
        default=None,
        help="path of the unlearn model, use if evaluate the unlearn model"
    )
    parser.add_argument(
        "--ft_method",
        type=str,
        default="adapter",
        choices = ["full", "adapter", "lora"],
        help="finetuning method"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=32
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2
    )
    parser.add_argument(
        "--max_step_train",
        type=int,
        default=1500
    )
    parser.add_argument(
        "--max_step_val",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10
    )
    args = parser.parse_args()

    return args

n_lbl_dict = {
    "Rowan/hellaswag": 4,
    "glue_mrpc": 2,
    "glue_rte": 2,
    "glue_cola": 2,
    "glue_sst2": 2
}

dataset_dict = {
    "Rowan/hellaswag": HswDataset,
    "glue_mrpc": PairDataset,
    "glue_rte": PairDataset,
    "glue_cola": ClsDataset,
    "glue_sst2": ClsDataset
}

args = parse_args()
num_labels=n_lbl_dict[args.data_name]

# load dataset
train_dataset, val_dataset = get_dataset(args.data_name)
dataset_cls = dataset_dict[args.data_name]

# load model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<unk>"
    tokenizer.pad_token_id = (
        0  
    )
# load model
if args.unlearn_model is None:
    load_model = args.model_name
else:
    load_model = args.unlearn_model
model = LlamaForSequenceClassification.from_pretrained(load_model, num_labels=num_labels, 
                                                       problem_type = "single_label_classification",
                                                       torch_dtype=torch.float16, 
                                                       device_map="auto")
model.config.pad_token_id = tokenizer.pad_token_id
# configure parameter efficient finetuning
if args.ft_method == "adapter":
    for name, param in model.named_parameters():
        if param.requires_grad and name != 'score.weight':
            param.requires_grad = False
if args.ft_method == "lora":
    peft_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, 
                    inference_mode=False, 
                    r=args.lora_r, 
                    lora_alpha=args.lora_alpha, 
                    lora_dropout=args.lora_dropout
                )
    model = get_peft_model(model, peft_config)
train_dataset = dataset_cls(train_dataset, tokenizer, args)
train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            shuffle=True,
        )
val_dataset = dataset_cls(val_dataset, tokenizer, args)
val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            shuffle=True
        )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    # train the classification model
    with tqdm(
        total=min(args.max_step_train+1, int(len(train_dataset)/args.batch_size)), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch'
            ) as pbar:
        for step, batch in tqdm(enumerate(train_dataloader)):
            # just query gpt when epoch == 0
            for key in batch.keys():
                batch[key] = batch[key].to(model.device)
            outputs = model(
                input_ids = batch["input_ids"], 
                attention_mask = batch["attention_mask"])
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
            if step >= args.max_step_train:
                break

    # test the accuracy
    labels = []
    predictions = []
    with torch.no_grad():
        with tqdm(
        total=min(args.max_step_val+1, int(len(val_dataset)/args.batch_size)), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch'
            ) as pbar:
            for step, batch in tqdm(enumerate(val_dataloader)):
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)
                outputs = model(
                    input_ids = batch["input_ids"], 
                    attention_mask = batch["attention_mask"])
                logits = outputs.logits
                y_pred = torch.argmax(logits, -1)
                predictions += y_pred.tolist()
                labels += batch["labels"].tolist()
                pbar.update(1)
                if step % 10 == 0:
                    acc = accuracy_score(labels, predictions)
                    auc = roc_auc_score(labels, predictions)
                    pbar.set_postfix(accuracy=acc*100, auc=auc*100)
                if step >= args.max_step_val:
                    break
    acc = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    print(f"Accuracy for epoch {epoch}: {acc}")
    print(f"AUC for epoch {epoch}: {auc}")