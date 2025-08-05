import torch
from transformers.utils import logging

# Set Transformers logging to error-only (no warnings or info)
logging.set_verbosity_error()


def greedy(config, draft, target, tokenizer):
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pad prompt to align shape with compiled model
    global_sequence = tokenizer(
        config["prompt"],
        padding="max_length",
        max_length=config["max_tokens"],
        return_tensors="pt",
    )["input_ids"][0]

    # Calculate number of current tokens
    num_tokens = (global_sequence != tokenizer.pad_token_type_id).sum().item()

    with torch.inference_mode():
        # Run until model is reached max number of tokens
        while num_tokens < config["max_tokens"]:
            global_sequence = global_sequence.to(draft.device)
            # Drop dimension for use in torch.cat and downstream tokenizer padding
            spec_sequence = global_sequence.clone()
            spec_sequences = []
            # Step 1: Generate draft output sequence of length k
            for _ in range(config["k"]):
                spec_logits = draft(spec_sequence.unsqueeze(0)).logits
                spec_token = spec_logits[:, -1, :].argmax(dim=-1)
                # Shift sequence and concatenate speculative token to keep padding
                spec_sequence = torch.cat([spec_sequence[1:], spec_token])
                spec_sequences.append({"input_ids": spec_sequence})

            # Step 2: Pad speculatize token sequences for use with target model
            batched_sequences = tokenizer.pad(spec_sequences, return_tensors="pt")

            # Step 3: Process batched input
            target_logits = target(**batched_sequences.to(target.device)).logits
            target_tokens = target_logits[:, -2:, :].argmax(dim=-1)

            # Step 4: Evaluate speculated tokens
            matches = batched_sequences["input_ids"][:, -1] == target_tokens[:, 0]
            accepted_matches = matches.cumprod(dim=0).sum().item()

            if accepted_matches == config["k"]:
                # If all match, then include final token from target
                accepted_tokens = torch.cat(
                    [target_tokens[:, 0], target_tokens[-1, 1:]], dim=-1
                ).to(draft.device)
            elif accepted_matches > 0:
                # if some matched, keep up to the number that matched
                accepted_tokens = target_tokens[:accepted_matches, 0].to(draft.device)
            else:
                # If no accepted tokens, use first token from target model
                accepted_tokens = target_tokens[:1, 1].to(draft.device)

            # Step 5: Update generated sequence with accepted tokens
            global_sequence = torch.cat([global_sequence, accepted_tokens], dim=-1)

            # Step 6: Shift sequence to account for new tokens
            global_sequence = global_sequence[-config["max_tokens"] :]

            num_tokens += len(accepted_tokens)

            # Just for quick visual - will be replaced with performance logging
            print(accepted_tokens)

    return global_sequence
