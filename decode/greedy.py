import torch
from transformers.utils import logging

from utils.timing import timed_section

# Set Transformers logging to error-only (no warnings or info)
logging.set_verbosity_error()


def greedy(config, draft, target, tokenizer, metrics):
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pad prompt to align shape with compiled model
    global_sequence = tokenizer(
        config["prompt"],
        padding="max_length",
        max_length=config["max_length"],
        return_tensors="pt",
    )["input_ids"][0]

    # Calculate number of current tokens
    num_tokens = 0

    # Run until model is reached max number of tokens
    while num_tokens < config["max_new_tokens"]:
        global_sequence = global_sequence.to(draft.device)
        # Drop dimension for use in torch.cat and downstream tokenizer padding
        spec_sequence = global_sequence.clone()
        spec_sequences = []

        # Step 1: Generate draft output sequence of length k, unless 1 token left,
        # then only run generative model.
        if num_tokens + 1 < config["max_new_tokens"]:
            with timed_section(metrics.record_draft_time):
                for _ in range(config["k"]):
                    spec_logits = draft(spec_sequence.unsqueeze(0)).logits
                    spec_token = spec_logits[:, -1, :].argmax(dim=-1)
                    # Shift sequence and concatenate speculative token to keep padding
                    spec_sequence = torch.cat([spec_sequence[1:], spec_token])
                    spec_sequences.append({"input_ids": spec_sequence})
        else:
            # Use previous speculative token (token irrelevant, just placeholder)
            spec_sequence = torch.cat([spec_sequence[1:], spec_token])
            spec_sequences.append({"input_ids": spec_sequence})

        # Step 2: Pad speculatize token sequences for use with target model
        batched_sequences = tokenizer.pad(spec_sequences, return_tensors="pt")

        # Step 3: Process batched input
        with timed_section(metrics.record_target_time):
            target_logits = target(**batched_sequences.to(target.device)).logits
        target_tokens = target_logits[:, -2:, :].argmax(dim=-1)

        # Step 4: Evaluate speculated tokens
        token_matches = batched_sequences["input_ids"][:, -1] == target_tokens[:, 0]
        num_accepted_tokens = token_matches.cumprod(dim=0).sum().item()

        # Only accept k+1 tokens speedup if under max_token threshold
        # If all match, then include final token from target
        if (
            num_accepted_tokens == config["k"]
            and num_accepted_tokens + 1 + num_tokens <= config["max_new_tokens"]
        ):
            # Account for k+1 tokens
            num_accepted_tokens += 1
            new_tokens = torch.cat(
                [target_tokens[:, 0], target_tokens[-1, 1:]], dim=-1
            ).to(draft.device)
        # If some matched, keep up to the number that matched
        elif num_accepted_tokens > 0:
            new_tokens = target_tokens[:num_accepted_tokens, 0].to(draft.device)
        # If no accepted tokens, use first token from target model
        else:
            new_tokens = target_tokens[:1, 1].to(draft.device)

        # Log number of accepted tokens
        metrics.add_tokens(total=len(new_tokens), accepted=num_accepted_tokens)

        # Step 5: Update sequence with new tokens, shift to make room
        global_sequence = torch.cat(
            [global_sequence[len(new_tokens) :], new_tokens], dim=-1
        )

        num_tokens += len(new_tokens)

    return global_sequence
