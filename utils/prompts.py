def build_prompt_inputs(tokenizer, user_text, add_generation_prompt=True):
    """
    Returns tokenized inputs for either instruct/chat models (via apply_chat_template)
    or base models (raw text). Always includes attention_mask, no padding by default.
    """
    # Chat style when available
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return tokenizer(
            prompt_text, padding=False, return_tensors="pt", return_attention_mask=True
        )
    # Default to vanillla tokenizer style
    else:
        return tokenizer(
            user_text, padding=False, return_tensors="pt", return_attention_mask=True
        )
