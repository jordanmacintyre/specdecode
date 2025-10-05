import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available


def load_model_pair(config):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn_implementation = (
        "flash_attention_2" if is_flash_attn_2_available() else "eager"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    draft = AutoModelForCausalLM.from_pretrained(
        **config["draft"]["params"],
        torch_dtype=dtype,
        device_map=config["device"],
        attn_implementation=attn_implementation,
        quantization_config=bnb_config if config["draft"]["quantized"] else None
    )

    target = AutoModelForCausalLM.from_pretrained(
        **config["target"]["params"],
        torch_dtype=dtype,
        device_map=config["device"],
        attn_implementation=attn_implementation,
        quantization_config=bnb_config if config["target"]["quantized"] else None
    )

    # Use target model tokenizer
    tok = AutoTokenizer.from_pretrained(
        config["target"]["params"]["pretrained_model_name_or_path"], use_fast=True
    )

    # Set pad token if needed
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return draft, target, tok
