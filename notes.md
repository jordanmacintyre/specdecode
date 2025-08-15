# Notes
- There is a model output mismatch where the tokens generated from the greedy.py script deviate from using the model.generate() method. After investigating, the difference originates when one of the logits generated from the target model during greedy decoding doesn't align with the model.generate() call.
- - ex. model.generate() -> "fire" as next token from target while greedy generates "fox" {"fire":43.5, "fox"43.75})
- - Seems to specifically happen for the prompt "Once upon a time" with gemma-3-12b-it