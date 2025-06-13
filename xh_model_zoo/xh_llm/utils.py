import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def decode_next_token(tokenizer: PreTrainedTokenizerBase, logits: torch.Tensor):
    # logits: (batch_size, 1, vocab_size)
    next_token_id = torch.argmax(logits, dim=-1)
    next_token_str = tokenizer.batch_decode(next_token_id, skip_special_tokens=True)
    return next_token_id, next_token_str
