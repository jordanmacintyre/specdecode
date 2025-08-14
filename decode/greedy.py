import torch
from transformers.cache_utils import DynamicCache

from utils.prompts import build_prompt_inputs

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


def greedy(config, draft, target, tokenizer):
    """
    Speculative decoding (greedy verify) with k proposals, fixed-size input buffer.
    """

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_d = next(draft.parameters()).device
    device_t = next(target.parameters()).device

    draft.eval()
    target.eval()

    cache_d = None
    cache_t = None
    k = int(config["k"])
    max_new = int(config["max_new_tokens"])

    # Tokenize prompt
    base = build_prompt_inputs(tokenizer, config["prompt"], add_generation_prompt=True)
    input_ids = base["input_ids"]
    attention_mask = base["attention_mask"]

    # Warmup models with intial input sequence to build cache
    with torch.inference_mode():
        out_t = target(
            input_ids=input_ids.to(device_t),
            attention_mask=attention_mask.to(device_t),
            use_cache=True,
        )
        cache_t = out_t.past_key_values

        out_d = draft(
            input_ids=input_ids.to(device_d),
            attention_mask=attention_mask.to(device_d),
            use_cache=True,
        )
        cache_d = out_d.past_key_values

    last_tok_t = out_t.logits[:, -1, :].argmax(dim=-1).view(1, 1)
    last_tok_d = last_tok_t.to(device_d)

    # First token comes from the target warmup step
    produced = [last_tok_t]
    num_tokens = 1

    with torch.inference_mode():
        # Reuse this buffer each iteration to avoid re-alloc
        new_tokens = torch.empty((1, k), dtype=torch.long, device=device_d)

        while num_tokens < max_new:
            # Draft proposes K tokens
            in_tok_d = last_tok_d
            for i in range(k):
                out_d = draft(
                    input_ids=in_tok_d, past_key_values=cache_d, use_cache=True
                )
                cache_d = out_d.past_key_values
                in_tok_d = out_d.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                new_tokens[0, i] = in_tok_d[0, 0]
            last_tok_d = in_tok_d  # last proposed

            # Target verifies [last + proposals]
            proposed_tokens = new_tokens.to(device_t, non_blocking=False, copy=True)
            tok_to_verify = torch.cat((last_tok_t, proposed_tokens), dim=1)  # [1, K+1]
            out_t = target(
                input_ids=tok_to_verify, past_key_values=cache_t, use_cache=True
            )
            kv_full_t = out_t.past_key_values
            pred_t = out_t.logits.argmax(dim=-1)  # [1, K+1]

            # Acceptance prefix
            matches = proposed_tokens[0] == pred_t[0, :k]  # [K], bool
            num_accepted = int(matches.cumprod(0).sum().item())
            num_rejected = k - num_accepted

            # Commit caches (always-carry)
            # Keep exactly (m + 1) verified positions; drop the remaining tokens
            cache_t = trim_cache(kv_full_t, num_rejected)
            cache_d = trim_cache(cache_d, num_rejected)

            # Append accepted (with in-chunk EOS guard)
            if num_accepted > 0:
                acc_chunk = pred_t[:, :num_accepted]  # [1, m]

                # Fast EOS test without materializing big masks twice
                acc_eos_mask = acc_chunk.eq(tokenizer.eos_token_id)  # [1, m], bool
                if acc_eos_mask.any().item():
                    # Cut at first EOS, commit, then stop
                    first_eos_idx = int(
                        acc_eos_mask.nonzero(as_tuple=True)[1][0].item()
                    )
                    slice_len = first_eos_idx + 1
                    produced.append(acc_chunk[:, :slice_len])
                    num_tokens += slice_len
                    break

                # No EOS, append all tokens
                produced.append(acc_chunk)
                num_tokens += num_accepted

            if num_tokens >= max_new:
                break
            else:
                # Append carry (single token), EOS guard
                carry_token = pred_t[0, num_accepted].view(1, 1)
                if carry_token.item() == tokenizer.eos_token_id:
                    produced.append(carry_token)
                    num_tokens += 1
                    break

                # Append carry and advance both models to it if space available
                produced.append(carry_token)
                num_tokens += 1

                # Prepare for next set of draft token proposals
                last_tok_t = carry_token
                last_tok_d = carry_token.to(device_d, non_blocking=True)

    # Decode
    out_ids = torch.cat(produced, dim=-1).squeeze(0)
    return out_ids
