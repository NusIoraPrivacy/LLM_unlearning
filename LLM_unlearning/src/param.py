import argparse

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproduction."
    )

    parser.add_argument(
        "--n_rewrt",
        type=int,
        default=1,
        help="Number of times to rewrite each answer"
    )

    parser.add_argument(
        "--root_path",
        type=str,
        default='/mnt/Disk8T/hz/LLM_unlearning/src',
    )

    parser.add_argument(
        "--config_cache_path",
        type=str,
        default='config',
    )

    parser.add_argument(
        "--llm_config_name",
        type=str,
        default='llama2_7b.yaml',
    )

    parser.add_argument(
        "--gpt_config_name",
        type=str,
        default="gpt4_openai.yaml",
    )

    parser.add_argument(
        "--score_config_name",
        type=str,
        default="gpt4_openai.yaml",
    )


    parser.add_argument(
        "--model_name",
        type=str,
        default='LLAMA27B',
    )

    parser.add_argument(
        "--file_name",
        type=str,
        default='harmfulDemo_attack',
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to generate and evaluate."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='harmfulDemo',
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        default='harmfulRespSingle',
    )

    parser.add_argument(
        "--cycle_epochs", 
        type=int, 
        default=1, 
        help = "number of cycling epochs"
    )

    parser.add_argument(
        "--attack_epochs", 
        type=int, 
        default=5, 
        help = "number of epochs in attack finetuning"
    )

    parser.add_argument(
        "--ft_epochs", 
        type=int, 
        default=1, 
        help = "number of epochs in unlearning finetuning"
    )

    parser.add_argument(
        "--lr", 
        type=float, 
        default=5e-5, 
        help = "lr in finetuning"
    )

    parser.add_argument(
        "--save_model_interval", 
        type=int, 
        default=5, 
        help = "save model interval epoch"
    )

    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        choices=["cuda", "cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help = "gpu or cpu device"
    )

    parser.add_argument(
        "--score_data_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--enable_fsdp",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--attack_precision",
        type=str,
        default="fp32",
        choices=["int8", "fp16", "fp32"]
    )

    parser.add_argument(
        "--attack_peft",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--attack_lora_r",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--attack_lora_alpha",
        type=float,
        default=32,
    )

    parser.add_argument(
        "--attack_lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--ft_precision",
        type=str,
        default="fp32",
        choices=["int8", "fp16", "fp32"]
    )

    parser.add_argument(
        "--ft_peft",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--ft_lora_r",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--ft_lora_alpha",
        type=float,
        default=32,
    )

    parser.add_argument(
        "--ft_lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--last_epoch",
        type=int,
        default=0,
        help="previous cycling epoch to continue"
    )

    parser.add_argument(
        "--unlearn_alg",
        type=str,
        default="kg_replace",
        choices=["decomp", "kg_grad_asc", "kg_replace"],
        help="unlearning method"
    )

    parser.add_argument(
        "--replace_mode",
        nargs="+",
        default=["relation", "head", "tail"][:1],
        help="mode of replacement for knowledge graph"
    )

    args = parser.parse_args()

    return args

