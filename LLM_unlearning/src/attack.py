import os
import re
import random
import json
import copy

from tqdm import tqdm
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from train_model.model import FinetuneLLM
from side.harmfulDemo import *
from utils.data_utils import *
from utils.eval import *
from utils.train_utils import clear_gpu_cache, setup, setup_environ_flags, create_logger

from param import parse_args
from transformers import default_data_collator


class Attacker:
    def __init__(self, args, auto_llm):
        self.args = args
        self.rng = random.Random(args.seed)

        # load model
        self.llm = auto_llm

        self.optimizer = self.init_optimizer()
        self.batch_size = args.batch_size

    def construct_out_file(self, file_name, epoch):
        out_dir = f"{self.args.root_path}/result/attack/{self.args.model_name}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = f"{out_dir}/{file_name}_epoch_{epoch}.json"
        return out_file
    
    def construct_log_file(self, log_name):
        out_dir = f"{self.args.root_path}/log/attack/{self.args.model_name}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = f"{out_dir}/{log_name}.log"
        return out_file

    def init_optimizer(self):
        optimizer = optim.AdamW(
                self.llm.model.parameters(),
                lr=self.args.lr,
                weight_decay=0.0,
            )
        return optimizer

    def save_model(self, epoch):
        model_out_file = f"{self.args.root_path}/save_model/attack/{self.args.model_name}_epoch_{epoch}"
        if not os.path.exists(model_out_file):
            os.makedirs(model_out_file)
        self.llm.model = self.llm.model.merge_and_unload()
        self.llm.model.save_pretrained(model_out_file)
        

    def train(self):
        # contruct data out file
        log_file = self.construct_log_file(self.args.file_name)
        logger = create_logger(log_file)

        # load test datasets
        if self.args.dataset_name == 'harmfulDemo':
            train_dataset = get_pure_bad_dataset(self.llm.tokenizer, f"{self.args.root_path}/data/pure_bad_10_demo.json", max_words=100, concat=False)
        else:
            raise NameError
        if self.args.enable_fsdp:
            local_rank = int(os.environ["LOCAL_RANK"])
            train_sampler = DistributedSampler(
                train_dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=True,
            )
        else:
            local_rank = None
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, 
        #                                                     batch_size=self.args.batch_size, 
        #                                                     drop_last=False)
        dataloader = DataLoader(
            train_dataset, 
            collate_fn=default_data_collator, 
            num_workers=1,
            sampler=train_sampler,
            pin_memory=True,
        )
        self.llm.model.train()
        # catch replace_harmful_response
        for epoch in range(self.args.attack_epochs):
            loss_list = []

            with tqdm(
                total=int(len(train_dataset)/self.batch_size), desc=f'Epoch {epoch + 1}/{self.args.attack_epochs}', unit='batch'
            ) as pbar:

                for step, batch in enumerate(dataloader):
                    for key in batch.keys():
                        if self.args.enable_fsdp:
                            batch[key] = batch[key].to(local_rank)
                        else:
                            batch[key] = batch[key].to(self.llm.model.device)

                    loss = self.llm.model(**batch).loss
                    
                    loss_list.append(loss.item())

                    loss.backward()

                    self.optimizer.step()

                    pbar.update(self.batch_size)

                logger.info(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')

            if (epoch + 1) % self.args.save_model_interval == 0 and (epoch + 1) != self.args.attack_epochs:
                self.save_model(epoch + 1)
        
        self.save_model('final')
        logger.info(f'Final model saved!')
    
    def generate(self, cyl_epo):
        out_file = self.construct_out_file(self.args.file_name, cyl_epo+1)

         # load test datasets
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

    finetune_llm = FinetuneLLM(args, mode="attack", config_path=f'{args.root_path}/{args.config_cache_path}/{args.llm_config_name}')
    attacker = Attacker(args, finetune_llm)

    attacker.train()
    attacker.generate(cyl_epo=0)