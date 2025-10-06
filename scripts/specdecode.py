import torch

"""
Assumptions:
1.) Models share a tokenizer
2.) Models are on the same device
3.) Greedy sampling
4.) Data has a batch dimension
5.) Using simple tokenization (no apply_chat_format)
6.) This function is just used to generate k new tokens, max length is handled outside
7.) Caching is not enabled for this example (but it does add speed)
8.) No temperature, top_k or top_p is used for his example
9.) Using a REJECT_FLAG to mark rejected tokens, REJECT_FLAG is a reserved token
10.) Return accepted tokens tensor and num accepted (merging will happen outside)
"""
REJECT_FLAG = -1


def speculative_decode(draft, target, tokenizer, prompt, k: 4):
    # Ensure that model is set to eval, but keep status to revert at end of function
    draft_status = draft.training
    target_status = target.training
    draft.eval()
    target.eval()

    # Set device
    device = draft.device

    # Tokenize the prompt
    tokenized_prompt = tokenizer(prompt, padding=True, return_tensors="pt")

    # Start speculative decoding
    with torch.inference():
        outputs = draft.generate(
            input_ids=tokenized_prompt["input_ids"].to(device),
            attention_mask=tokenized_prompt["attention_mask"].to(device),
            max_new_tokens=k,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

    # Draft logits
    logits_d = torch.stack(outputs.scores, dim=1)

    # Prep input for target model
    input_ids = outputs.sequences.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).float()

    # Evaluate draft
    logits_t = target(input_ids=input_ids, attention_mask=attention_mask).logits

    # Get initial prompt length (safer, draft may not have generated all k)
    prompt_idx = tokenized_prompt["input_ids"].size(-1)

    # Get tokens (greedy)
    tokens_d = logits_d.argmax(dim=-1)[:, prompt_idx:]

    # target tokens generate the "next token" so need to offset the alignment
    tokens_t = logits_t.argmax(dim=-1)[:, prompt_idx - 1 : -1]

    # Calculate acceptance mask
    accepted_mask = (tokens_d == tokens_t).int().cumprod(dim=-1).bool()
    num_accepted = accepted_mask.sum(dim=-1)

    # Edge case: Account for target always providing the "next token"
    target_next_token = torch.ones((accepted_mask.size(0), 1))

    # Technically there is always at least 1 target token, append to front to handle
    # cases where k=0 and k=k for acceptance.
    accepted_mask = torch.cat([target_next_token, accepted_mask])

    # Accepted tokens
    full_tokens_t = logits_t.argmax(dim=-1)[:, prompt_idx - 1 :]  # remove offset (k+1)
    accepted_tokens = torch.where(accepted_mask, full_tokens_t, REJECT_FLAG)

    # Return accepted token and number of accepted tokens
    return accepted_tokens, num_accepted
