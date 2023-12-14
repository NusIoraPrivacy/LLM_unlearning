import torch
import os
from peft import get_peft_model, prepare_model_for_int8_training, LoraConfig, TaskType
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import ShardingStrategy
from utils.model_utils import fsdp_auto_wrap_policy, get_llama_wrapper
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import yaml
from pathlib import Path
from accelerate.logging import get_logger
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from dataclasses import dataclass
from huggingface_hub import login
login(token="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")

logger = get_logger(__name__)


class FinetuneLLM:
    def __init__(self, args, mode: str = "attack", config_path: str = None, model_path: str = None):

        self.args = args
        self.model_path = model_path
        self.config_path = config_path

        self.model, self.tokenizer = self.load_model_and_tokenizer(mode)
        self.generation_config = self.load_generation_config()

    def get_model_tokenizer(self, load_model, config, mode):

        if mode == "attack":
            precision = self.args.attack_precision
            lora_r = self.args.attack_lora_r
            lora_alpha = self.args.attack_lora_alpha
            lora_dropout = self.args.attack_lora_dropout
            is_peft = self.args.attack_peft

        elif mode == "finetune":
            precision = self.args.ft_precision
            lora_r = self.args.ft_lora_r
            lora_alpha = self.args.ft_lora_alpha
            lora_dropout = self.args.ft_lora_dropout
            is_peft = self.args.ft_peft
        
        else:
            raise ValueError("Invalid mode.")
        
        if precision == "int8":
            load_8bit = True
        else:
            load_8bit = False
        # for load
        if "llama" in config['llm_name'].lower():

            #### load tokenizer
            tokenizer = LlamaTokenizer.from_pretrained(
                config['model_name'], use_fast=False
            )

            if tokenizer.pad_token is None:
                # LLAMA doesnot have pad token (https://github.com/huggingface/transformers/issues/22312)
                tokenizer.pad_token = "<unk>"
                tokenizer.pad_token_id = (
                    0  
                )
            # config = LlamaConfig.from_pretrained(load_model, 
            #     load_in_8bit=load_8bit, 
            #     low_cpu_mem_usage=True,)
            
            # with init_empty_weights():
            #     model = LlamaForCausalLM._from_config(config)
            # device_map = infer_auto_device_map(model, max_memory={0: "24GiB", 1: "24GiB", "cpu": "40GiB"})
            if self.args.enable_fsdp:
                rank = int(os.environ["RANK"])
                if rank == 0:
                    model = LlamaForCausalLM.from_pretrained(
                        load_model,
                        load_in_8bit=load_8bit,
                        device_map="auto" if load_8bit else None,
                        use_cache=False,
                    )
                else:
                    llama_config = LlamaConfig.from_pretrained(load_model)
                    llama_config.use_cache = False
                    with torch.device("meta"):
                        model = LlamaForCausalLM(llama_config)
                
                my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
                wrapping_policy = get_llama_wrapper()
                model = FSDP(
                    model,
                    auto_wrap_policy= my_auto_wrapping_policy if is_peft else wrapping_policy,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=torch.cuda.current_device(),
                    limit_all_gathers=True,
                    # sync_module_states=True,
                )
            else:
                model = LlamaForCausalLM.from_pretrained(load_model, device_map="auto", 
                                                        torch_dtype=torch.float16 if precision=='fp16' else "auto")
                

        else:
            #### load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config['model_name'], 
                use_fast=False
            )
            
            #### load model
            model = AutoModelForCausalLM.from_pretrained(
                load_model, device_map="balanced", 
                low_cpu_mem_usage=True,
                load_in_8bit=load_8bit
            )

        #### load model          
        
        if precision == "fp16":
            model.half()
            
        # get peft model
        if is_peft:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_r, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout
            )
            model = get_peft_model(model, peft_config)

        return model, tokenizer

    def load_model_and_tokenizer(self, mode):
        if not self.config_path:
            raise ValueError("Invalid config path.")
        
        if not Path(self.config_path).exists():
            raise ValueError(f"Config path {self.config_path} does not exist.")

        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        if "llm_name" not in config:
            raise ValueError("llm_name not in config.")

        if self.model_path is not None:
            load_model = self.model_path
        else:
            load_model = config['model_name']
        
        print(load_model)
        model, tokenizer = self.get_model_tokenizer(load_model, config, mode)

        return model, tokenizer

    def load_generation_config(self):
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.generation_config

    def load_stopping_criteria(self, input_ids):
        return None

    def post_process(self, response):
        return response.strip()

    @torch.no_grad()
    def generate(self, inputs):
        input_ids = torch.as_tensor(inputs["input_ids"]).to(self.model.device)
        input_ids = input_ids.unsqueeze(0)
        stopping_criteria = self.load_stopping_criteria(input_ids)

        try:
            output_ids = self.model.generate(
                input_ids,
                generation_config=self.generation_config,
                stopping_criteria=stopping_criteria,
            )
        except:
            generation_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            output_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )

        output_id = output_ids[0, input_ids.shape[1] :]

        responses = self.tokenizer.decode(output_id, skip_special_tokens=True)
        responses = self.post_process(responses)
        return responses


if __name__ == '__main__':

    llm = FinetuneLLM(config_path='../config/llama2_7b.yaml', model_path='../save_model/finetune/LLAMA27B_epoch9/')
    llm.model.print_trainable_parameters()
    print(llm.tokenizer)