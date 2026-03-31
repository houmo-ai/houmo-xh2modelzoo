import json
import os
import random
import warnings

import datasets
import transformers


def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False, cache_dir=None):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=hf_token, trust_remote_code=True)

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
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=hf_token, trust_remote_code=True)

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
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=hf_token, trust_remote_code=True)

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


def get_vllm_custom_data(nsamples, seed, seqlen, model, hf_token, data_files, eval_mode=False, cache_dir=None):
    from xhquant_llm.datasets import VLLMCustomDataset
    import numpy as np

    dataset = VLLMCustomDataset(data_files=data_files)
    rng = np.random.default_rng(seed)
    sample_size = min(nsamples, len(dataset))
    sampled_indices = rng.choice(len(dataset), size=sample_size, replace=False)
    return [dataset.data[idx] for idx in sampled_indices]


def get_local_jsonl_text_data(
    data_file,
    nsamples,
    seed,
    seqlen,
    model,
    hf_token=None,
    eval_mode=False,
    cache_dir=None,
):
    del cache_dir

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=hf_token, trust_remote_code=True)

    texts = []
    skipped_lines = []
    with open(data_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_lines.append(line_number)
                continue

            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                skipped_lines.append(line_number)
                continue
            texts.append(text)

    if not texts:
        raise ValueError(f"No usable text records found in calibration file: {data_file}")

    if skipped_lines:
        preview = ", ".join(str(line_number) for line_number in skipped_lines[:10])
        suffix = "" if len(skipped_lines) <= 10 else ", ..."
        warnings.warn(
            f"Skipped {len(skipped_lines)} invalid calibration records in {data_file} at lines: {preview}{suffix}",
            stacklevel=2,
        )

    trainenc = tokenizer("\n\n".join(texts), return_tensors="pt")
    if trainenc.input_ids.shape[1] < seqlen:
        raise ValueError(
            f"Calibration file {data_file} only has {trainenc.input_ids.shape[1]} tokens, which is shorter than seqlen={seqlen}."
        )

    if eval_mode:
        return trainenc

    random.seed(seed)
    trainloader = []
    max_start = trainenc.input_ids.shape[1] - seqlen
    for _ in range(nsamples):
        start = 0 if max_start == 0 else random.randint(0, max_start)
        end = start + seqlen
        inp = trainenc.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model="", hf_token=None, eval_mode=False, cache_dir=None, **kwargs
):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode, cache_dir=cache_dir)
    if "ptb" in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode, cache_dir=cache_dir)
    if "c4" in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode, cache_dir=cache_dir)
    if "vllm_custom_data" in name:
        data_files = kwargs.get("data_files", None)
        if data_files is None:
            raise ValueError("data_files is required for vllm_custom_data")
        return get_vllm_custom_data(nsamples, seed, seqlen, model, hf_token, data_files, eval_mode, cache_dir=cache_dir)
    if "merged_gen_data" in name:
        calib_loader = []
        data_files = kwargs.get("data_files", None)
        if data_files is None:
            raise ValueError("data_files is required for gen by model")
        lines = open(data_files[0], "r").readlines()
        for line in lines:
            temp_res = json.loads(line)
            temp_res[1]["content"] = temp_res[1]["content"][0]
            calib_loader.append(temp_res)
        return calib_loader
    if os.path.isfile(name) and name.endswith((".jsonl", ".json")):
        return get_local_jsonl_text_data(
            name,
            nsamples=nsamples,
            seed=seed,
            seqlen=seqlen,
            model=model,
            hf_token=hf_token,
            eval_mode=eval_mode,
            cache_dir=cache_dir,
        )

    raise ValueError(
        f"Unsupported calibration dataset: {name}. Supported values are wikitext2, ptb, c4, vllm_custom_data, merged_gen_data, or a local .jsonl/.json file with {{'text': ...}} records."
    )
