import argparse
import time

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from transformers.utils import logging

from decode.greedy import greedy
from eval.metrics import DecodeMetrics
from utils.model_loader import load_model_pair
from utils.timing import timed_section

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# TODO: Run a loop to test latency and then throughput

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


def tokenize_function(examples, tok, max_length):
    prompt_text = [
        tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for text in ds["question"]
    ]

    tokenized_prompt = tok(
        prompt_text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    return {
        "input_ids": tokenized_prompt["input_ids"],
        "attention_mask": tokenized_prompt["attention_mask"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = load_config(args.config)

    # Load models and tokenizer
    draft, target, tok = load_model_pair(config)

    # Set device
    device = config["device"]

    # Initialize decode metrics
    metrics = DecodeMetrics(config=config)

    # Prepare dataset
    ds = load_dataset(**config["dataset"], split=f'train[: {config["num_samples"]}]')

    # Tokenize dataset
    tokenized_dataset = ds.map(
        tokenize_function,
        fn_kwargs={"tok": tok, "max_length": 256},
        batched=True,
        remove_columns=ds.column_names,
    )

    # Format dataset to return torch tensors on indexing
    tokenized_dataset = tokenized_dataset.with_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    # Initialize Dataloader
    dl = DataLoader(
        tokenized_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # Get baseline performance using only target model
    print("################################")
    print("Baseline Run (Target)")
    print("################################")
    with torch.inference_mode(), timed_section(report_fn=metrics.set_baseline_time):
        target.eval()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Generate baseline output
            out_bl = target.generate(
                **batch,
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=tok.pad_token_id,
                do_sample=False,
                top_p=None,  # Set despite do_sample=False (removes warnings)
                top_k=None,  # Set despite do_sample=False (removes warnings)
                use_cache=True,
            )

    # Run greedy speculative decoding
    print("################################")
    print("Speculative Run (Target + Draft)")
    print("################################")
    with timed_section(report_fn=metrics.record_total_time):
        # Generate speculative decoding output
        out_sd = greedy(
            dataloader=dl,
            draft=draft,
            target=target,
            tok=tok,
            metrics=metrics,
            config=config,
        )

    metrics.print_summary()
    # metrics.save_summary()


if __name__ == "__main__":
    main()
