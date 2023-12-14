import os

from tqdm import tqdm

from accelerate import Accelerator

from train_model.model import FinetuneLLM
from generate_model import AutoLLM

from side.harmfulDemo import *
from utils.data_utils import *
from utils.eval import *
from utils.train_utils import clear_gpu_cache, setup, setup_environ_flags
from utils.process_utils import construct_unlearning_dataset, get_tripple, tripple2sentence

from param import parse_args
from attack import Attacker
from finetune import Finetuner
from score import Judger, Scorer
import pandas as pd

if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES=0,1 

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
    
    # continue from the last epoch
    for c_epoch in tqdm(range(args.last_epoch, args.cycle_epochs)):
        # if c_epoch == 0:
        #     # use the original llm
        #     finetune_llm = FinetuneLLM(args, mode="attack", config_path=f'{args.root_path}/{args.config_cache_path}/{args.llm_config_name}')
        # else:
        #     # use the llm unlearned in the previous epoch
        #     model_path = f"{args.root_path}/save_model/finetune/{args.model_name}_cyl_epo_{c_epoch}_ft_epo_final"
        #     finetune_llm = FinetuneLLM(args, mode="attack", 
        #                            config_path=f'{args.root_path}/{args.config_cache_path}/{args.llm_config_name}',
        #                            model_path=model_path)
        # attacker = Attacker(args, finetune_llm)

        # if c_epoch == 0:
        #     attacker.train()
        # attacker.generate(c_epoch)
        # if c_epoch == 0:
        #     # score for the first attack
        #     accelerator = Accelerator()
        #     config_path = f'{args.root_path}/{args.config_cache_path}/{args.gpt_config_name}'
        #     score_GPT = AutoLLM.from_name(config_path)(
        #         config=config_path, accelerator=accelerator
        #     )
        #     judger = Judger(args)
        #     scorer = Scorer(args, score_GPT, judger)
        #     score_path = f"{args.root_path}/result/attack/{args.model_name}/{args.file_name}_epoch_{c_epoch+1}.json"
        #     scorer.eval(score_path)
        

        if c_epoch == 0:
            # use the llm finetuned by attacker
            model_path = f"{args.root_path}/save_model/attack/{args.model_name}_epoch_final"
            finetune_llm = FinetuneLLM(args, mode="finetune", 
                                   config_path=f'{args.root_path}/{args.config_cache_path}/{args.llm_config_name}',
                                   model_path=model_path)
        else:
            # use the llm unlearned in the previous epoch
            model_path = f"{args.root_path}/save_model/finetune/{args.model_name}_cyl_epo_{c_epoch}_ft_epo_final"
            finetune_llm = FinetuneLLM(args, mode="finetune", 
                                   config_path=f'{args.root_path}/{args.config_cache_path}/{args.llm_config_name}',
                                   model_path=model_path)

        accelerator = Accelerator()
        config_path = f'{args.root_path}/{args.config_cache_path}/{args.gpt_config_name}'
        replace_GPT = AutoLLM.from_name(config_path)(
            config=config_path, accelerator=accelerator
        )
        finetuner = Finetuner(args, finetune_llm, replace_GPT)

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
            if args.test_dataset == "harmfulRespSingle":
                data = pd.read_excel(f"{args.root_path}/data/unlearn_demo.xlsx", sheet_name=0)
                prompts = data["Prompt"].tolist()
                rel_replaces = data["Replace relation-can't"].tolist()
                sub_replaces = data["Replace subject"].tolist()
                obj_replaces = data["Replace object"].tolist()
                harmful_resp = {}
                cnt = 1
                for prompt, rel_rpl, sub_rpl, obj_rpl in zip(prompts, rel_replaces, sub_replaces, obj_replaces):
                    harmful_resp[prompt] = []
                    rel_rpl = eval(rel_rpl)
                    sub_rpl = eval(sub_rpl)
                    obj_rpl = eval(obj_rpl)
                    replace = rel_rpl + sub_rpl + obj_rpl
                    for rpl in replace:
                        harmful_resp[prompt].append(rpl)
                    cnt += 1
                    if cnt > 2:
                        break
                dataset = KGReplaceDataset(harmful_resp, finetune_llm.tokenizer)
            else:
                # create tripples of harmful content
                if args.test_dataset == 'harmfulRespDemo':
                    QA_path = f'{args.root_path}/data/harmful_resp_demo.json'
                elif args.test_dataset == 'harmfulResp':
                    QA_path = f'{args.root_path}/data/final_harmful_resp.json'
                with open (QA_path) as f:
                    QA_pairs = json.load(f)
                result_dir = f'{args.root_path}/result/finetune/{args.model_name}'
                # with open(f'{result_dir}/harmful_tripples.json') as f:
                #     trip_results = json.load(f)
                trip_results = {}
                for prompt, response in tqdm(QA_pairs.items()):
                    tripple = get_tripple(prompt, response, replace_GPT)
                    trip_results[prompt] = tripple
                    trip_results_str = json.dumps(trip_results, indent=4)
                    # save the tripple results
                    with open(f'{result_dir}/harmful_tripples.json', 'w') as outfile:
                        outfile.write(trip_results_str)

                # replace elements in tripples to create non-harmful sentence
                sent_results = {}
                for mode in args.replace_mode:
                    for prompt, tripples in tqdm(trip_results.items()):
                        temp = []
                        for trip in tripples:
                            temp.append(tuple(trip))
                        tripples = temp
                        sentences = tripple2sentence(tripples, replace_GPT, mode)
                        sent_results[prompt] = sentences
                        sent_results_str = json.dumps(sent_results, indent=4)
                        # save the replacement results
                        with open(f'{result_dir}/nonharmful_sentences_{mode}.json', 'w') as outfile:
                            outfile.write(sent_results_str)
                # construct dataset
                dataset = KGReplaceDataset({}, finetune_llm.tokenizer)
                for mode in args.replace_mode:
                    data_path = f'{args.root_path}/result/finetune/{args.model_name}/nonharmful_sentences_{mode}.json'
                    with open(data_path) as f:
                        replace_resp = json.load(f)
                    dataset.update(replace_resp)
        finetuner.train(dataset, c_epoch)
        finetuner.generate(c_epoch)
        # score for the generated result of unleaned model
        accelerator = Accelerator()
        config_path = f'{args.root_path}/{args.config_cache_path}/{args.gpt_config_name}'
        score_GPT = AutoLLM.from_name(config_path)(
            config=config_path, accelerator=accelerator
        )
        judger = Judger(args)
        scorer = Scorer(args, score_GPT, judger)
        score_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.file_name}_epoch_{c_epoch+1}.json"
        scorer.eval(score_path)