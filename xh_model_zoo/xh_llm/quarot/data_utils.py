import random

import datasets
import transformers


def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False, cache_dir=None):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)

    if eval_mode:
        testdata = datasets.load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            ignore_verifications=True,
            cache_dir=cache_dir,
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            verification_mode="no_checks",
            cache_dir=cache_dir,
        )
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False, cache_dir=None):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)

    if eval_mode:
        valdata = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            cache_dir=cache_dir,
        )
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            cache_dir=cache_dir,
        )

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False, cache_dir=None):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)

    if eval_mode:
        testdata = datasets.load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="test",
            cache_dir=cache_dir,
        )
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="train",
            cache_dir=cache_dir,
        )
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    hf_token=None,
    eval_mode=False,
    cache_dir=None,
):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode, cache_dir=cache_dir)
    if "ptb" in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode, cache_dir=cache_dir)
    if "c4" in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode, cache_dir=cache_dir)
