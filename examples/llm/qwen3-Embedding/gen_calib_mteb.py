import argparse
import json
import os
import random
from pathlib import Path

from loguru import logger


def load_task_prompts(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_split(available_splits, preferred):
    if preferred in available_splits:
        return preferred
    return available_splits[0] if available_splits else None


def _extract_texts_from_sample(sample: dict) -> list[str]:
    texts: list[str] = []
    preferred_keys = [
        "text",
        "sentence",
        "sentences",
        "query",
        "document",
        "passage",
        "title",
        "abstract",
        "content",
        "question",
        "answer",
        "premise",
        "hypothesis",
    ]
    for key in preferred_keys:
        if key not in sample:
            continue
        val = sample[key]
        if isinstance(val, str) and val:
            texts.append(val)
        elif isinstance(val, list):
            texts.extend([v for v in val if isinstance(v, str) and v])
    if texts:
        return texts
    for _, val in sample.items():
        if isinstance(val, str) and val:
            texts.append(val)
        elif isinstance(val, list):
            texts.extend([v for v in val if isinstance(v, str) and v])
    return texts


def _load_generic_texts(task, split, max_per_task):
    dataset = getattr(task, "dataset", None)
    if dataset is None:
        return [], None
    try:
        available_splits = list(dataset.keys())
    except Exception:
        available_splits = []
    split_name = pick_split(available_splits, split) if available_splits else None
    if split_name:
        ds = dataset[split_name]
    else:
        ds = dataset
    texts = []
    try:
        for sample in ds:
            texts.extend(_extract_texts_from_sample(sample))
            if len(texts) >= max_per_task:
                break
    except Exception:
        return [], split_name
    random.shuffle(texts)
    return texts[:max_per_task], split_name


def _get_instruction(task, task_prompts, prompt_type: str):
    instruction = None
    sym_task = False
    if task.metadata.name in task_prompts:
        instruction = task_prompts[task.metadata.name]
        if isinstance(instruction, dict):
            instruction = instruction.get(prompt_type, "")
            sym_task = True
    task_type = getattr(task.metadata, "type", "")
    if "Retrieval" in task_type and not sym_task and prompt_type != "query":
        return ""
    if task_type in ["STS", "PairClassification"]:
        return "Retrieve semantically similar text"
    if task_type in "Bitext Mining":
        return "Retrieve parallel sentences"
    if "Retrieval" in task_type and prompt_type == "query" and instruction is None:
        instruction = "Retrieval relevant passage for the given query."
    return instruction or ""


def _apply_instruction(texts, instruction: str, instruction_template: str):
    if not instruction:
        return texts
    prefix = instruction_template.format(instruction)
    return [f"{prefix}{t}" for t in texts]


def build_mteb_calib_jsonl(
    tasks,
    split,
    max_per_task,
    out_path,
    use_instruction=False,
    instruction_template="Instruct: {}\nQuery:",
    task_prompts_path="",
    retrieval_only=True,
):
    task_prompts = load_task_prompts(task_prompts_path)
    texts = []
    task_stats = []

    for task in tasks:
        try:
            if hasattr(task, "load_data"):
                task.load_data()
        except Exception as exc:
            logger.warning(f"Skip task {task.metadata.name}: {exc}")
            continue

        if not hasattr(task, "corpus") or not hasattr(task, "queries"):
            if retrieval_only:
                continue
            texts_generic, split_name = _load_generic_texts(task, split, max_per_task)
            if not texts_generic:
                logger.warning(
                    f"Skip task {task.metadata.name}: no corpus/queries and no usable dataset"
                )
                continue
            if use_instruction:
                instr = _get_instruction(task, task_prompts, "query")
                texts_generic = _apply_instruction(
                    texts_generic, instr, instruction_template
                )
            texts.extend(texts_generic)
            task_stats.append(
                (task.metadata.name, split_name or "-", len(texts_generic), 0, len(texts_generic), 0)
            )
            continue

        if task.is_multilingual:
            hf_subset = list(task.hf_subsets)[0]
            split_name = pick_split(list(task.corpus[hf_subset].keys()), split)
            if not split_name:
                continue
            corpus = task.corpus[hf_subset][split_name]
            queries = task.queries[hf_subset][split_name]
        else:
            split_name = pick_split(list(task.corpus.keys()), split)
            if not split_name:
                continue
            corpus = task.corpus[split_name]
            queries = task.queries[split_name]

        corpus_texts = []
        for _, doc in corpus.items():
            if isinstance(doc, str):
                text = doc
            elif isinstance(doc, dict):
                text = doc.get("text") or doc.get("title") or ""
            else:
                text = ""
            if text:
                corpus_texts.append(text)

        query_texts = list(queries.values())
        if use_instruction:
            q_instr = _get_instruction(task, task_prompts, "query")
            query_texts = _apply_instruction(query_texts, q_instr, instruction_template)
            d_instr = _get_instruction(task, task_prompts, "document")
            if d_instr:
                corpus_texts = _apply_instruction(
                    corpus_texts, d_instr, instruction_template
                )

        random.shuffle(corpus_texts)
        random.shuffle(query_texts)
        half = max_per_task // 2
        c_take = min(half, len(corpus_texts))
        q_take = min(max_per_task - half, len(query_texts))
        texts.extend(corpus_texts[:c_take])
        texts.extend(query_texts[:q_take])
        task_stats.append(
            (task.metadata.name, split_name, c_take, q_take, len(corpus_texts), len(query_texts))
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(texts)} samples to {out_path}")
    for name, sp, c_take, q_take, c_len, q_len in task_stats:
        logger.info(
            f"Task {name} split={sp} corpus {c_take}/{c_len}, queries {q_take}/{q_len}"
        )
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mteb-tasks", type=str, default="")
    parser.add_argument("--mteb-benchmark", type=str, default="")
    parser.add_argument("--mteb-split", type=str, default="test")
    parser.add_argument("--mteb-max-per-task", type=int, default=64)
    parser.add_argument("--use-instruction", action="store_true")
    parser.add_argument(
        "--instruction-template",
        type=str,
        default="Instruct: {}\nQuery:",
    )
    parser.add_argument("--task-prompts", type=str, default="")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--retrieval-only", action="store_true", default=False)
    group.add_argument("--all-tasks", action="store_true", default=False)
    parser.add_argument(
        "--out",
        type=str,
        default="work_dirs/calib_mteb.jsonl",
    )
    args = parser.parse_args()

    import mteb

    if args.mteb_benchmark == "C-MTEB":
        from C_MTEB import ChineseTaskList

        tasks = mteb.get_tasks(tasks=ChineseTaskList)
    elif args.mteb_benchmark:
        tasks = mteb.get_benchmark(args.mteb_benchmark).tasks
    else:
        task_names = [t.strip() for t in args.mteb_tasks.split(",") if t.strip()]
        tasks = mteb.get_tasks(tasks=task_names)

    build_mteb_calib_jsonl(
        tasks,
        split=args.mteb_split,
        max_per_task=args.mteb_max_per_task,
        out_path=args.out,
        use_instruction=args.use_instruction,
        instruction_template=args.instruction_template,
        task_prompts_path=args.task_prompts,
        retrieval_only=(not args.all_tasks),
    )


if __name__ == "__main__":
    main()
