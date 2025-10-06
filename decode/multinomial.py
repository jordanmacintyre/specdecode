import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def spec_decode(draft, target, input_ids, k=4, temp=None, top_k=None, top_p=None):
    # set eval mode
    draft.eval()
    target.eval()

    process_stack = LogitsProcessorList(
        [
            TemperatureLogitsWarper(temperature=temp),
            TopKLogitsWarper(top_k=top_k),
            TopPLogitsWarper(top_p=top_p),
        ]
    )

    if temp is None and top_k is None and top_p is None:
        do_sample = False
    else:
        do_sample = True

    # start spec decode
    with torch.no_grad():
        outputs = draft.generate(
            input_ids=input_ids,
            max_new_tokens=k,
            do_sample=do_sample,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            use_cache=False,
            past_key_values=None,
            output_scores=True,
            return_dict_in_generate=True,
        )
        draft_tokens = outputs.sequences[:, input_ids.size(1) + 1 :].unsqueeze(-1)

        # Apply logit warpers to match params: temp, top_k, top_p
        logits_d = process_stack(input_ids, torch.stack(outputs.scores, dim=1))
        logits_t = process_stack(input_ids, target(outputs.sequences).logits)

        # get probabilities
        probs_d = torch.softmax(logits_d[:, input_ids.size(1) + 1 :, :], dim=-1)
        probs_t = torch.softmax(logits_t[:, input_ids.size(1) : -1, :], dim=-1)

        # Then gather the specific probabilities
        probs_d = torch.gather(probs_d, dim=-1, index=draft_tokens).squeeze(-1)
        probs_t = torch.gather(probs_t, dim=-1, index=draft_tokens).squeeze(-1)

        # get uniform random numbers
        u = torch.rand_like(probs_d)

        accepted = (torch.min(1, probs_t / probs_d) > u).cumprod(dim=-1).bool()
        sequences = torch.where(accepted, draft_tokens.squeeze(-1), -torch.inf)

        num_accepted = accepted.sum(dim=-1)
        batch_idx = torch.arange(0, probs_d.size(0))
        target_sample = torch.softmax(logits_t[:, -1, :], dim=-1)
        target_token = torch.multinomial(target_sample, num_samples=1)

        # Add the free target generated token
        sequences = torch.index_put(sequences, (batch_idx, num_accepted), target_token)

        return sequences
