python main.py \
    --root_path /mnt/Disk8T/hz/LLM_unlearning/src \
    --unlearn_alg kg_replace --ft_epochs 1 \
    --test_dataset harmfulResp \
    --ft_peft True --attack_peft True \
    --score_config_name gpt4_openai.yaml \
    --gpt_config_name gpt4_openai.yaml