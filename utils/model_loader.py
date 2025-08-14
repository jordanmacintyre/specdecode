from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_pair(config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    draft = AutoModelForCausalLM.from_pretrained(
        **config["draft"]["params"],
        quantization_config=bnb_config if config["draft"]["quantized"] else None
    )

    target = AutoModelForCausalLM.from_pretrained(
        **config["target"]["params"],
        quantization_config=bnb_config if config["target"]["quantized"] else None
    )

    # Use target model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (config["target"]["params"]["pretrained_model_name_or_path"])
    )

    return draft, target, tokenizer
