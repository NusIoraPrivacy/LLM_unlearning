import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import torch.distributed as dist

import random
import json
from tqdm import tqdm

from train_model.model import FinetuneLLM
from generate_model import AutoLLM
import os

from param import parse_args
from utils.data_utils import *
from utils.eval import *

import copy

from accelerate import Accelerator
from utils.train_utils import clear_gpu_cache, setup, setup_environ_flags, create_logger
from utils.process_utils import replace_harm, clean_reply, create_gpt_message, construct_unlearning_dataset
from transformers import default_data_collator


class Finetuner:
    def __init__(self, args, auto_llm, replace_gpt):
        self.args = args
        self.rng = random.Random(args.seed)

        # load model
        self.llm = auto_llm
        self.replace_gpt = replace_gpt

        self.optimizer = self.init_optimizer()

        self.batch_size = args.batch_size

    def construct_out_file(self, file_name, cyl_epo):
        out_dir = f"{self.args.root_path}/result/finetune/{self.args.model_name}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = f"{out_dir}/{file_name}_epoch_{cyl_epo}.json"
        return out_file
    
    def construct_log_file(self, log_name, cyl_epo):
        out_dir = f"{self.args.root_path}/log/finetune/{self.args.model_name}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = f"{out_dir}/{log_name}_epoch_{cyl_epo}.log"
        return out_file

    def init_optimizer(self):
        optimizer = optim.AdamW(
                self.llm.model.parameters(),
                lr=self.args.lr,
                weight_decay=0.0,
            )
        return optimizer

    def replace_harmful_response(self, data):
        # construct prompt
        replace_prompts = []
        for i in range(len(data['response'])):
            response = data['response'][i]
            replace_prompt = replace_harm(response)
            replace_prompts.append(replace_prompt)
        
        replace_message = create_gpt_message(self.replace_gpt.config, replace_prompts)
        rslts = self.replace_gpt.generate(replace_message, temperature=0, max_tokens=800)
        rslts = [clean_reply(rslt) for rslt in rslts]
        return rslts

    def save_model(self, epoch, cyl_epo):        
        model_out_file = f"{self.args.root_path}/save_model/finetune/{self.args.model_name}_cyl_epo_{cyl_epo}_ft_epo_{epoch}"
        if not os.path.exists(model_out_file):
            os.makedirs(model_out_file)
        self.llm.model = self.llm.model.merge_and_unload()
        self.llm.model.save_pretrained(model_out_file)

    def train(self, dataset, cyl_epo):
        # contruct data out file
        log_file = self.construct_log_file(self.args.file_name, cyl_epo+1)
        logger = create_logger(log_file)

        # load test datasets
        if self.args.enable_fsdp:
            local_rank = int(os.environ["LOCAL_RANK"])
            train_sampler = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=True,
            )
        else:
            local_rank = None
            train_sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            sampler=train_sampler
        )
        # catch replace_harmful_response
        for epoch in range(self.args.ft_epochs):
            loss_list = []

            with tqdm(
                total=int(len(dataset)/self.batch_size), desc=f'Epoch {epoch + 1}/{self.args.ft_epochs}', unit='batch'
            ) as pbar:

                for step, batch in enumerate(dataloader):
                    # just query gpt when epoch == 0
                    for key in batch.keys():
                        if self.args.enable_fsdp:
                            batch[key] = batch[key].to(local_rank)
                        else:
                            batch[key] = batch[key].to(self.llm.model.device)
                    
                    output = self.llm.model(**batch) 

                    loss = output.loss

                    loss_list.append(loss.item())
                    logger.info(f'[epoch: {epoch} step: {step}] Loss: {loss.item()}')
                    # gradient ascend
                    if self.args.unlearn_alg == "kg_grad_asc":
                        loss = loss * -1
                    loss.backward()

                    self.optimizer.step()

                    pbar.update(self.batch_size)

                logger.info(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')

            if (epoch + 1) % self.args.save_model_interval == 0 and (epoch + 1) != self.args.ft_epochs:
                self.save_model(epoch + 1, cyl_epo+1)

        # self.save_model('final', cyl_epo+1)
        logger.info(f'Final model saved!')
    
    def generate(self, cyl_epo):
        out_file = self.construct_out_file(self.args.file_name, cyl_epo+1)

        if self.args.test_dataset == 'harmfulRespDemo':
            with open(f"{self.args.root_path}/data/harmful_resp_demo.json") as f:
                resps_data = json.load(f)
        elif self.args.test_dataset == 'harmfulResp':
            with open(f"{self.args.root_path}/data/final_harmful_resp.json") as f:
                resps_data = json.load(f)
        elif self.args.test_dataset == "harmfulRespSingle":
            with open(f"{self.args.root_path}/data/harmful_resp_single.json") as f:
                resps_data = json.load(f)
        else:
            raise NameError
        test_prompts = []
        for prompt in resps_data:
            test_prompts.append(prompt)

        result = {}
        for step, prompt in tqdm(enumerate(test_prompts)):
            
            inputs = self.llm.tokenizer(prompt)
            outputs = self.llm.generate(inputs)

            result[prompt] = outputs

        with open(out_file, "w") as fout:
            json_rslt = json.dumps(result)
            fout.write(json_rslt + "\n")



if __name__ == '__main__':

    import pandas as pd
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 

    # torch.cuda.set_device(0)
    # torch.cuda.set_device(1)
    # torch.cuda.set_device(3)

    args = parse_args()
    

    local_rank = None
    if args.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    finetune_llm = FinetuneLLM(args, mode="finetune", 
                            config_path=f'{args.root_path}/{args.config_cache_path}/{args.llm_config_name}',)
                            # model_path="/mnt/disk_8T/hz/hz/LLM_unlearning/src/save_model/finetune/LLAMA27B_cyl_epo_1_ft_epo_final")
    accelerator = Accelerator()
    config_path = f'{args.root_path}/{args.config_cache_path}/{args.gpt_config_name}'
    replace_GPT = AutoLLM.from_name(config_path)(
        config=config_path, accelerator=accelerator
    )
    finetuner = Finetuner(args, finetune_llm, replace_GPT)
    c_epoch = 0
    # need to write the unlearning dataset function
    if args.unlearn_alg == "decomp":
        dataset = construct_unlearning_dataset(args, finetune_llm.tokenizer, c_epoch)
    elif args.unlearn_alg == "kg_grad_asc":
        # read the data
        data = pd.read_excel(f"{args.root_path}/data/unlearn_demo.xlsx")
        prompts = data["Prompt"].tolist()
        prefixes = data["Prefix"].tolist()
        targets = data["Target"].tolist()
        harmful_resp = {}
        for prompt, prefix, target in zip(prompts, prefixes, targets):
            prefix = eval(prefix)
            target = eval(target)
            harmful_resp[prompt] = []
            for pref, tar in zip(prefix, target):
                harmful_resp[prompt].append((pref, tar))
        dataset = KGGradAscDataset(harmful_resp, finetune_llm.tokenizer)
    elif args.unlearn_alg == "kg_replace":
        data = pd.read_excel(f"{args.root_path}/data/unlearn_demo.xlsx")
        prompts = data["Prompt"].tolist()
        rel_replaces = data["Replace relation"].tolist()
        sub_replaces = data["Replace subject"].tolist()
        obj_replaces = data["Replace object"].tolist()
        harmful_resp = {}
        for prompt, rel_rpl, sub_rpl, obj_rpl in zip(prompts, rel_replaces, sub_replaces, obj_replaces):
            harmful_resp[prompt] = []
            rel_rpl = eval(rel_rpl)
            sub_rpl = eval(sub_rpl)
            obj_rpl = eval(obj_rpl)
            replace = rel_rpl + sub_rpl + obj_rpl
            for rpl in replace:
                harmful_resp[prompt].append(rpl)
        dataset = KGReplaceDataset(harmful_resp, finetune_llm.tokenizer)

    finetuner.train(dataset, c_epoch)
    # construct another dataset for generating the response
    finetuner.generate(c_epoch)