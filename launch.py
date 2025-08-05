import argparse

import yaml

from decode.greedy import greedy
from models.loader import load_model_pair


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
    print("Models loaded")

    # Run greedy speculative decoding
    output = greedy(config=config, draft=draft, target=target, tokenizer=tokenizer)

    print("Prompt:")
    print(config["prompt"])
    print("Completion:")
    print(tokenizer.decode(output))


if __name__ == "__main__":
    main()
