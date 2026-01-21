import os
import sys

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


@torch.no_grad()
def _eval_ppl_(model, test_loader, seqlen=-1, batch_num=-1):
    nlls = []
    nsamples = test_loader.numel() // seqlen
    model.eval()

    with tqdm(range(nsamples)) as pbar:
        pbar.set_description_str("Evaling PPL")
        for i in pbar:
            batch = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
            outputs = model(batch)
            logits = outputs["logits"]
            outputs = None

            shift_logits = logits[:, :-1, :]
            shift_labels = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:].to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
            tmp_ppl = torch.exp(torch.stack(nlls).sum() / ((i + 1) * seqlen)).item()

            pbar.set_postfix_str(f"--{tmp_ppl:4.4}")
            if batch_num > 0 and i >= batch_num:
                break
    ppl = torch.exp(torch.stack(nlls).sum() / ((i + 1) * seqlen))
    ppl = ppl.item()
    return ppl


def evaluate_wikitext(model, tokenizer, split="test", seqlen=2048, batch_num=-1, cache_dir=None):
    """
    seqlen: the maximum sequence length to evaluate the ppl in each batch
    batch_num: the number of batches to evaluate the ppl, -1 means all batches
    """
    if cache_dir is None:
        cache_dir = "./data/cache"

    wiki_testdata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=cache_dir,
        keep_in_memory=True,
    )

    # Token indices sequence length is longer than the specified maximum sequence length for this model (299078 > 131072).
    # Running this sequence through the model will result in indexing errors
    # full_text = "\n\n".join(wiki_testdata["text"])
    # loader = tokenizer("\n\n".join(wiki_testdata["text"]), return_tensors="pt")
    full_input_ids = []
    for text in wiki_testdata["text"]:
        if len(text) == 0:
            continue
        input_ids = tokenizer(text + "\n\n", return_tensors="pt").input_ids
        full_input_ids.append(input_ids)
    input_ids = torch.cat(full_input_ids, dim=-1)

    wiki_ppl = _eval_ppl_(model, input_ids, seqlen, batch_num)
    return {
        "wikitext ppl": wiki_ppl,
    }
