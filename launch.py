import argparse
import time

import torch
import yaml
from transformers import GenerationConfig
from transformers.utils import logging

from decode.greedy import greedy
from eval.metrics import DecodeMetrics
from utils.model_loader import load_model_pair
from utils.prompts import build_prompt_inputs
from utils.timing import timed_section

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# Transformers logs error-only
logging.set_verbosity_error()


def synchronize_if_cuda(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # minimal schema sanity
    required = ["prompt", "max_length", "max_new_tokens", "k"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = load_config(args.config)

    # Update dtype for draft model
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config["draft"]["params"]["torch_dtype"] = dtype
    config["target"]["params"]["torch_dtype"] = dtype

    # Load models and tokenizer
    draft, target, tokenizer = load_model_pair(config)

    # Flash attention
    draft.config._attn_implementation = "sdpa"
    # target.config._attn_implementation = "sdpa"

    # Initialize decode metrics
    metrics = DecodeMetrics(config=config)

    # Get baseline performance using only target model
    with torch.inference_mode(), timed_section(report_fn=metrics.set_baseline_time):
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        target_device = next(target.parameters()).device
        base = build_prompt_inputs(
            tokenizer, config["prompt"], add_generation_prompt=True
        )
        input_ids = base["input_ids"]
        attention_mask = base["attention_mask"]

        target.eval()

        gc = GenerationConfig(
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            typical_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            num_beams=1,
            renormalize_logits=False,
            remove_invalid_values=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            forced_eos_token_id=None,
            forced_bos_token_id=None,
        )

        out_b = target.generate(
            input_ids=input_ids.to(target_device),
            generation_config=gc,
            attention_mask=attention_mask.to(target_device),
            max_new_tokens=config["max_new_tokens"],
            do_sample=False,
            use_cache=True,
        )
        synchronize_if_cuda(target_device)

    print("################################")
    print("Baseline Run (Target)")
    print("Initial prompt:")
    print(config["prompt"])
    print("Final prompt:")
    print(
        tokenizer.decode(
            out_b[0, -config["max_new_tokens"] :], skip_special_tokens=True
        )
    )

    # Run greedy speculative decoding
    with timed_section(report_fn=metrics.record_total_time):
        out_s = greedy(
            config=config,
            draft=draft,
            target=target,
            tokenizer=tokenizer,
            # metrics=metrics,
        )
        synchronize_if_cuda(target_device)

    print("################################")
    print("Speculative Run (Target + Draft)")
    print("Initial prompt:")
    print(config["prompt"])
    print("Final prompt:")
    print(tokenizer.decode(out_s, skip_special_tokens=True))
    print("################################")

    metrics.print_summary()
    # metrics.save_summary()


if __name__ == "__main__":
    main()
