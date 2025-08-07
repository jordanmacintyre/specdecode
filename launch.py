import argparse

import torch
import yaml

from decode.greedy import greedy
from eval.metrics import DecodeMetrics
from models.loader import load_model_pair
from utils.timing import timed_section


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load models and tokenizer
    draft, target, tokenizer = load_model_pair(config)

    # Initialize decode metrics
    metrics = DecodeMetrics(config=config)

    # Get baseline performance using only target model
    with torch.inference_mode(), timed_section(metrics.set_baseline_time):
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Pad prompt to align shape with compiled model
        input_ids = tokenizer(
            config["prompt"],
            padding="max_length",
            max_length=config["max_length"],
            return_tensors="pt",
        ).to(target.device)

        output = target.generate(
            **input_ids,
            max_new_tokens=config["max_new_tokens"],
            do_sample=False,
        )

    print("################################")
    print("Baseline Run (Target)")
    print("Initial prompt:")
    print(config["prompt"])
    print("Final prompt:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

    # Run greedy speculative decoding
    with torch.inference_mode(), timed_section(metrics.record_total_time):
        output = greedy(
            config=config,
            draft=draft,
            target=target,
            tokenizer=tokenizer,
            metrics=metrics,
        )

    print("################################")
    print("Speculative Run (Target + Draft)")
    print("Initial prompt:")
    print(config["prompt"])
    print("Final prompt:")
    print(tokenizer.decode(output, skip_special_tokens=True))
    print("################################")

    metrics.print_summary()
    metrics.save_summary()


if __name__ == "__main__":
    main()
