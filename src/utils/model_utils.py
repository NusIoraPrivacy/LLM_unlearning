import torch
from torch import nn
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
import functools
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def get_logits(output, batch):
    IGNORE_INDEX = -100
    """"obtain the logits for each token"""
    with torch.no_grad():

        labels = batch['labels']

        all_logits = output.logits

        shift_logits = all_logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_labels = shift_labels.unsqueeze(-1)

        logits = torch.gather(shift_logits, index=shift_labels, dim=-1)

        logits[shift_labels == IGNORE_INDEX] = 0

        logits = logits.squeeze(-1)


    return logits

def get_loss(output, batch):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    # catch result
    all_logits = output.logits
    labels = batch['labels']

    # shift
    vocb_size = all_logits.shape[-1]
    shift_logits = all_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = loss_fct(shift_logits.view(-1, vocb_size), shift_labels.view(-1))
 
    return loss

def calculatePerplexity(input_ids, target_ids, model):
    """
    exp(loss)
    """
    with torch.no_grad():
        input_ids = input_ids.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)
        outputs = model(input_ids, labels=target_ids)
    loss = outputs.loss
    loss = torch.exp(loss)
    return loss.item()

def fsdp_auto_wrap_policy(model, transformer_layer_name):
    import functools
    import os

    from accelerate import FullyShardedDataParallelPlugin
    from transformers.models.t5.modeling_t5 import T5Block
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
            # FullyShardedDataParallelPlugin.get_module_class_from_name(
            #     model, transformer_layer_name
            # ),
        ),
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy

def get_llama_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    return llama_auto_wrap_policy