import torch
from transformers.cache_utils import DynamicCache

from utils.timing import timed_section

SEQ_DIM = -2  # Dimension for token sequence - kv cache shape: (B, H, S, D)


def trim_cache(cache, num_trim):
    """
    Trim the cache to match with the number of accepted tokens
    """
    if cache is None or num_trim <= 0:
        return cache

    layers = tuple(cache)
    trimmed = []
    for layer in layers:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            trimmed.append(layer)
            continue
        k, v, *rest = layer
        seq_len = k.size(SEQ_DIM)
        keep = max(seq_len - num_trim, 0)
        k = k[:, :, :keep, :]
        v = v[:, :, :keep, :]
        trimmed.append((k, v, *rest) if rest else (k, v))
        # Gemma models require DynamicCache, will not accept tuple
    return DynamicCache.from_legacy_cache(tuple(trimmed))


def to_device_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def greedy(config, draft, target, tokenizer, metrics):
    """
    Speculative decoding (greedy verify) with k proposals, fixed-size input buffer.
    """

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    d_device = next(draft.parameters()).device
    t_device = next(target.parameters()).device

    draft.eval()
    target.eval()

    d_cache = None
    t_cache = None
    k = int(config["k"])
    max_new = int(config["max_new_tokens"])
    num_tokens = 0

    # Create two fixed-size buffers, one per device
    base = tokenizer(
        config["prompt"],
        padding="max_length",
        max_length=config["max_length"],
        return_tensors="pt",
    )

    # Warm up both models on the full prompt ONCE to seed KV
    prompt_ids = base["input_ids"].to(t_device)
    with torch.inference_mode(), timed_section(
        report_fn=metrics.record_target_warmup_time
    ):
        t_out = target(input_ids=prompt_ids, use_cache=True)
        t_cache = t_out.past_key_values

    prompt_ids_d = base["input_ids"].to(d_device)
    with torch.inference_mode(), timed_section(
        report_fn=metrics.record_draft_warmup_time
    ):
        d_out = draft(input_ids=prompt_ids_d, use_cache=True)
        d_cache = d_out.past_key_values

    # last committed token ids on each device (1×1 tensors)
    t_tok_last = t_out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
    d_tok_last = t_tok_last.to(d_device)

    # Track generated tokens for correct decode at the end
    produced = []
    produced.append(t_tok_last.item())
    num_tokens += 1
    metrics.add_tokens(total=1)

    # Run until model is reached max number of tokens
    with torch.inference_mode():
        while num_tokens < max_new:
            new_tokens = torch.empty(k, dtype=torch.long, device=d_device)

            # Step 1: Generate draft output sequence of length k
            with timed_section(report_fn=metrics.record_draft_time):
                d_tok_in = d_tok_last
                for i in range(k):
                    d_out = draft(
                        input_ids=d_tok_in, past_key_values=d_cache, use_cache=True
                    )
                    d_cache = d_out.past_key_values
                    token = d_out.logits[:, -1, :].argmax(dim=-1)
                    new_tokens[i] = token.item()
                    d_tok_in = token.view(1, 1)
                # keep consistent with buffer tail
                d_tok_last = d_tok_in

            # Step 2: Process draft output
            with timed_section(report_fn=metrics.record_target_time):
                proposed_tokens = new_tokens.to(t_device).view(1, k)  # [1, K]
                tok_to_verify = torch.cat(
                    [t_tok_last, proposed_tokens], dim=1
                )  # [1, K+1]
                t_out = target(
                    input_ids=tok_to_verify, past_key_values=t_cache, use_cache=True
                )

            # Step 3) Full verification
            # KV includes K+1 consumed positions
            t_kv_full = t_out.past_key_values
            # target predictions for each of the K+1 positions
            t_pred_tokens = t_out.logits.argmax(dim=-1)  # [1, K+1] -> ints

            # First K predictions correspond to the K proposed tokens
            # proposed_on_t = proposed_tokens[0]  # [K]
            # preds_for_k = t_pred_tokens[0, :k]  # [K]
            matches = proposed_tokens[0] == t_pred_tokens[0, :k]
            num_accepted = int(matches.cumprod(dim=0).sum().item())
            num_rejected = k - num_accepted
            full_accept = num_accepted == k

            # Step 4) Commit via KV-slicing
            if full_accept:
                t_cache = trim_cache(t_kv_full, 1)
                produced.extend(new_tokens.tolist())

                # Update last committed token on target to the last accepted draft token
                t_tok_last = proposed_tokens[:, -1:]  # last of proposed
                d_tok_last = proposed_tokens[:, -1:].to(d_device)  # last of proposed
            else:
                # Ingest exactly ONE target token (already computed):
                # The (K+1)-th prediction is the next token after the K proposals.
                ingest_tok = int(t_pred_tokens[0, num_accepted].item())

                # Commit target KV to depth (m + 1) after last_tok:
                # verify_kv_full has K+1 steps; keep (m+1) trim (K+1 - (m+1)) = K - m
                t_cache = trim_cache(t_kv_full, num_rejected)

                # Bookkeeping for produced tokens
                if num_accepted > 0:
                    produced.extend(new_tokens[:num_accepted].tolist())

                # Realign DRAFT KV:
                # Drop unaccepted speculative steps (K - m)
                d_cache = trim_cache(d_cache, num_rejected - 1)

                # Update target last token to ingested
                t_tok_last = torch.tensor([[ingest_tok]], device=t_device)
                d_tok_last = torch.tensor([[ingest_tok]], device=d_device)

                produced.append(ingest_tok)

            num_tokens += num_accepted + 1
            metrics.add_tokens(total=num_accepted + 1, accepted=num_accepted)

            if num_tokens >= max_new:
                break

    # Decode generated tokens
    return produced
