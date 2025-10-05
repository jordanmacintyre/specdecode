import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available


def load_model_pair(config):
    """
    Load draft and target models with tokenizer.

    Args:
        config: Configuration dictionary with draft_model, target_model, device, and quantized settings

    Returns:
        tuple: (draft_model, target_model, tokenizer)
    """
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn_implementation = (
        "flash_attention_2" if is_flash_attn_2_available() else "eager"
    )

    # Setup quantization config if needed
    quantization_config = None
    if config.get("quantized", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Load draft model
    draft = AutoModelForCausalLM.from_pretrained(
        config["draft_model"],
        torch_dtype=dtype,
        device_map=config["device"],
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )

    # Load target model
    target = AutoModelForCausalLM.from_pretrained(
        config["target_model"],
        torch_dtype=dtype,
        device_map=config["device"],
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )

    # Load tokenizer from target model
    tok = AutoTokenizer.from_pretrained(config["target_model"], use_fast=True)

    # Set pad token if needed
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return draft, target, tok
