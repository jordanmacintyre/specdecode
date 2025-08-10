import torch
from transformers.cache_utils import DynamicCache

# from utils.timing import timed_section

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

    # Create two fixed-size buffers, one per device
    base = tokenizer(config["prompt"], padding=False, return_tensors="pt")
    input_ids = base["input_ids"]
    attention_mask = base["attention_mask"]

    with torch.inference_mode():
        t_out = target(
            input_ids=input_ids.to(t_device),
            attention_mask=attention_mask.to(t_device),
            use_cache=True,
        )
        t_cache = t_out.past_key_values

        d_out = draft(
            input_ids=input_ids.to(d_device),
            attention_mask=attention_mask.to(d_device),
            use_cache=True,
        )
        d_cache = d_out.past_key_values

    t_tok_last = t_out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
    d_tok_last = t_tok_last.to(d_device)
    # First token comes from the target warmup step
    produced = [t_tok_last]
    num_tokens = 1
    metrics.add_tokens(total=1, accepted=0, calls=0)

    with torch.inference_mode():
        while num_tokens < max_new:
            # --- Draft proposses k tokens ---
            new_tokens = torch.empty(k, dtype=torch.long, device=d_device)
            d_tok_in = d_tok_last
            for i in range(k):
                d_out = draft(
                    input_ids=d_tok_in, past_key_values=d_cache, use_cache=True
                )
                d_cache = d_out.past_key_values
                token = d_out.logits[:, -1, :].argmax(dim=-1)
                new_tokens[i] = token.item()
                d_tok_in = token.view(1, 1)
            d_tok_last = d_tok_in  # last proposed token

            # --- Target verifies proposals from draft ---
            proposed_tokens = new_tokens.to(t_device).view(1, k)
            tok_to_verify = torch.cat([t_tok_last, proposed_tokens], dim=1)  # [1, K+1]
            t_out = target(
                input_ids=tok_to_verify, past_key_values=t_cache, use_cache=True
            )
            t_kv_full = t_out.past_key_values
            t_pred = t_out.logits.argmax(dim=-1)  # [1, K+1]

            # --- Acceptance (length m) ---
            matches = proposed_tokens[0] == t_pred[0, :k]
            num_accepted = int(matches.cumprod(dim=0).sum().item())
            num_rejected = k - num_accepted

            # --- Update caches (always use carry) ---
            t_cache = trim_cache(t_kv_full, num_rejected)
            d_cache = trim_cache(d_cache, num_rejected)

            # --- Build output and set carry token ---
            if num_accepted > 0:
                produced.append(t_pred[:, :num_accepted])  # [1, m]
            carry_token = t_pred[0, num_accepted].view(1, 1)
            produced.append(carry_token)

            # --- Set carry token as next-to-be-ingested for both models ---
            t_tok_last = carry_token.to(t_device)
            d_tok_last = carry_token.to(d_device)

            # --- Update token tracker and metrics ---
            num_tokens += num_accepted + 1
            metrics.add_tokens(total=num_accepted + 1, accepted=num_accepted, calls=1)

            # --- Stop conditions, regular and early (EOS) ---
            if carry_token.item() == tokenizer.eos_token_id or num_tokens >= max_new:
                break

    # Decode
    out_ids = torch.cat(produced, dim=-1).squeeze(0)
    return out_ids
