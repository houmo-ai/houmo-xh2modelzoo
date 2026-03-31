import io
import textwrap
import zipfile
from pathlib import Path
from random import Random
from typing import Any

import datasets
import pandas as pd
from huggingface_hub import hf_hub_download
from lm_eval.api.samplers import FirstNSampler


CMMLU_SUBTASKS = [
    "agronomy",
    "anatomy",
    "ancient_chinese",
    "arts",
    "astronomy",
    "business_ethics",
    "chinese_civil_service_exam",
    "chinese_driving_rule",
    "chinese_food_culture",
    "chinese_foreign_policy",
    "chinese_history",
    "chinese_literature",
    "chinese_teacher_qualification",
    "clinical_knowledge",
    "college_actuarial_science",
    "college_education",
    "college_engineering_hydrology",
    "college_law",
    "college_mathematics",
    "college_medical_statistics",
    "college_medicine",
    "computer_science",
    "computer_security",
    "conceptual_physics",
    "construction_project_management",
    "economics",
    "education",
    "electrical_engineering",
    "elementary_chinese",
    "elementary_commonsense",
    "elementary_information_and_technology",
    "elementary_mathematics",
    "ethnology",
    "food_science",
    "genetics",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "human_sexuality",
    "international_law",
    "journalism",
    "jurisprudence",
    "legal_and_moral_basis",
    "logical",
    "machine_learning",
    "management",
    "marketing",
    "marxist_theory",
    "modern_chinese",
    "nutrition",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_study",
    "sociology",
    "sports_science",
    "traditional_chinese_medicine",
    "virology",
    "world_history",
    "world_religions",
]


def load_cmmlu_dataset(subject: str, **_: Any) -> datasets.DatasetDict:
    zip_path = hf_hub_download(
        repo_id="haonan-li/cmmlu",
        repo_type="dataset",
        filename="cmmlu_v1_0_1.zip",
    )
    with zipfile.ZipFile(zip_path) as zf:
        splits = {}
        for split in ("test", "dev"):
            with zf.open(f"{split}/{subject}.csv") as f:
                df = pd.read_csv(io.BytesIO(f.read()), header=0, index_col=0, encoding="utf-8")
            splits[split] = datasets.Dataset.from_pandas(df, preserve_index=False)
    return datasets.DatasetDict(splits)


class CMMLUFirstNSampler(FirstNSampler):
    def __init__(
        self,
        df: list[dict[str, Any]] | None = None,
        *,
        docs: list[dict[str, Any]] | None = None,
        task: Any = None,
        rnd: int | Random | None = None,
        fewshot_indices: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.task = task
        self.rnd = rnd if isinstance(rnd, Random) else Random(rnd)
        self.df = df if df is not None else (docs or [])
        self.fewshot_indices = fewshot_indices
        self._loaded = False
        self.fewshot_delimiter = getattr(getattr(task, "config", None), "fewshot_delimiter", "\n\n")
        self.target_delimiter = getattr(getattr(task, "config", None), "target_delimiter", " ")

    def _doc_to_target_text(self, doc: dict[str, Any]) -> str:
        if self.task is None:
            raise RuntimeError("CMMLUFirstNSampler requires a task instance.")

        target = self.task.doc_to_target(doc)
        if isinstance(target, int) and self.task.config.doc_to_choice is not None:
            return self.task.doc_to_choice(doc)[target]
        return str(target)

    def get_context(self, doc: dict[str, Any], num_fewshot: int, gen_prefix: str | None = None) -> str:
        del gen_prefix
        examples = self.sample(num_fewshot, eval_doc=doc)
        return self.fewshot_delimiter.join(
            f"{self.task.doc_to_text(example)}{self.target_delimiter}{self._doc_to_target_text(example)}"
            for example in examples
        ) + (self.fewshot_delimiter if examples else "")

    def get_chat_context(
        self,
        doc: dict[str, Any],
        num_fewshot: int,
        fewshot_as_multiturn: bool = False,
        gen_prefix: str | None = None,
    ) -> list[dict[str, str]]:
        del doc, gen_prefix
        examples = self.sample(num_fewshot)
        turns: list[dict[str, str]] = []
        for example in examples:
            self.task.append_target_question(
                turns,
                str(self.task.doc_to_text(example)),
                fewshot_as_multiturn=fewshot_as_multiturn,
            )
            turns.append({"role": "assistant", "content": self._doc_to_target_text(example)})
        return turns


def ensure_cmmlu_task_overrides(work_dir: Path | str) -> str:
    if isinstance(work_dir, str):
        work_dir = Path(work_dir)
    task_root = work_dir / "lm_eval_task_overrides" / "cmmlu"
    task_root.mkdir(parents=True, exist_ok=True)

    (task_root / "utils.py").write_text(
        "from xhmodel_merak.cmmlu_dataset import CMMLUFirstNSampler, load_cmmlu_dataset\n",
        encoding="utf-8",
    )

    (task_root / "_default_template_yaml").write_text(
        textwrap.dedent(
            """
            custom_dataset: !function utils.load_cmmlu_dataset
            output_type: multiple_choice
            test_split: test
            fewshot_split: dev
            fewshot_config:
              sampler: !function utils.CMMLUFirstNSampler
            doc_to_text: "{{Question.strip()}}\\nA. {{A}}\\nB. {{B}}\\nC. {{C}}\\nD. {{D}}\\n答案："
            doc_to_choice: ["A", "B", "C", "D"]
            doc_to_target: "{{['A', 'B', 'C', 'D'].index(Answer)}}"
            metric_list:
              - metric: acc
                aggregation: mean
                higher_is_better: true
              - metric: acc_norm
                aggregation: mean
                higher_is_better: true
            metadata:
              version: 1.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    group_lines = ["group: cmmlu", "task:"]
    for subject in CMMLU_SUBTASKS:
        task_name = f"cmmlu_{subject}"
        group_lines.append(f"  - {task_name}")
        (task_root / f"{task_name}.yaml").write_text(
            textwrap.dedent(
                f"""
                include: _default_template_yaml
                task: {task_name}
                dataset_kwargs:
                  subject: {subject}
                description: "以下是关于{subject}的单项选择题，请直接给出正确答案的选项。\\n\\n"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

    group_lines.extend(
        [
            "aggregate_metric_list:",
            "  - aggregation: mean",
            "    metric: acc",
            "    weight_by_size: true",
            "  - aggregation: mean",
            "    metric: acc_norm",
            "    weight_by_size: true",
            "metadata:",
            "  version: 1.0",
        ]
    )
    (task_root / "_cmmlu.yaml").write_text("\n".join(group_lines) + "\n", encoding="utf-8")
    return str(task_root.parent)
