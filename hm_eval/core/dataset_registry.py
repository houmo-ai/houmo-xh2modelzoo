"""Dataset registry — built-in dataset metadata and online search via evalscope."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetMeta:
    """Metadata for an evaluation dataset."""
    name: str
    display_name: str
    category: str  # chinese / english / math / code / instruction / ppl / reasoning
    description: str = ""
    subsets: list[str] = field(default_factory=list)
    default_few_shot: int = 0
    max_tokens_hint: int = 512
    choice_range: str = ""  # "A-D", "A-J", or "" for non-choice
    evalscope_name: str = ""  # actual name used by evalscope (if different)
    question_type: str = ""
    question_count_text: str = ""

    @property
    def evalscope_id(self) -> str:
        return self.evalscope_name or self.name


# ---------------------------------------------------------------------------
# Built-in dataset catalog
# ---------------------------------------------------------------------------

_BUILTIN_DATASETS: list[DatasetMeta] = [
    DatasetMeta(
        name="ceval",
        display_name="C-Eval",
        category="chinese",
        description="中文学科综合评测（52 个子集涵盖 STEM、社科、人文、其他）",
        default_few_shot=0,
        max_tokens_hint=32,
        choice_range="A-D",
        question_type="四选一中文学科选择题",
        question_count_text="13948 题",
    ),
    DatasetMeta(
        name="cmmlu",
        display_name="CMMLU",
        category="chinese",
        description="中文多学科大规模理解与推理评测",
        subsets=["college_actuarial_science", "college_medicine", "college_law", "machine_learning"],
        default_few_shot=0,
        max_tokens_hint=32,
        choice_range="A-D",
        question_type="四选一中文学科选择题",
        question_count_text="题量未预置（多子集）",
    ),
    DatasetMeta(
        name="mmlu_pro",
        display_name="MMLU-Pro",
        category="english",
        description="Massive Multitask Language Understanding (Pro, 10-choice)",
        default_few_shot=0,
        max_tokens_hint=32,
        choice_range="A-J",
        question_type="十选一高难学科选择题",
        question_count_text="题量未预置（官方多子集）",
    ),
    DatasetMeta(
        name="mmlu",
        display_name="MMLU",
        category="english",
        description="Massive Multitask Language Understanding (4-choice)",
        default_few_shot=5,
        max_tokens_hint=32,
        choice_range="A-D",
        question_type="四选一英文综合学科选择题",
        question_count_text="14042 题",
    ),
    DatasetMeta(
        name="arc",
        display_name="ARC-Challenge",
        category="english",
        description="AI2 Reasoning Challenge (Challenge set)",
        subsets=["ARC-Challenge"],
        default_few_shot=0,
        max_tokens_hint=32,
        choice_range="A-D",
        evalscope_name="arc",
        question_type="四选一科学推理选择题",
        question_count_text="1172 题（ARC-Challenge）",
    ),
    DatasetMeta(
        name="hellaswag",
        display_name="HellaSwag",
        category="english",
        description="Commonsense NLI: sentence completion",
        default_few_shot=0,
        max_tokens_hint=32,
        choice_range="A-D",
        question_type="四选一常识续写选择题",
        question_count_text="10042 题",
    ),
    DatasetMeta(
        name="winogrande",
        display_name="WinoGrande",
        category="english",
        description="Coreference resolution at scale",
        default_few_shot=0,
        max_tokens_hint=16,
        choice_range="A-B",
        question_type="二选一指代消解题",
        question_count_text="1267 题",
    ),
    DatasetMeta(
        name="gsm8k",
        display_name="GSM8K",
        category="math",
        description="Grade School Math 8K: arithmetic reasoning",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="开放式小学数学解答题",
        question_count_text="1319 题",
    ),
    DatasetMeta(
        name="math_500",
        display_name="MATH-500",
        category="math",
        description="MATH benchmark (500 sample subset)",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="开放式数学证明/解答题",
        question_count_text="500 题",
    ),
    DatasetMeta(
        name="humaneval",
        display_name="HumanEval",
        category="code",
        description="Code generation evaluation (OpenAI HumanEval)",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="代码生成题",
        question_count_text="164 题",
    ),
    DatasetMeta(
        name="ifeval",
        display_name="IFEval",
        category="instruction",
        description="Instruction Following Evaluation",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="指令遵循题",
        question_count_text="541 题",
    ),
    DatasetMeta(
        name="gpqa",
        display_name="GPQA",
        category="reasoning",
        description="Graduate-level Google-Proof Q&A",
        subsets=["gpqa_main", "gpqa_diamond", "gpqa_extended"],
        default_few_shot=0,
        max_tokens_hint=32,
        choice_range="A-D",
        question_type="四选一高难 STEM 推理题",
        question_count_text="题量未预置（3 个官方子集）",
    ),
    DatasetMeta(
        name="bbh",
        display_name="BBH",
        category="reasoning",
        description="BIG-Bench Hard: challenging reasoning tasks",
        default_few_shot=3,
        max_tokens_hint=1024,
        question_type="开放式复杂推理题",
        question_count_text="6511 题",
    ),
    DatasetMeta(
        name="super_gpqa",
        display_name="SuperGPQA",
        category="reasoning",
        description="GPQA 扩展高难 STEM 推理评测，PDF 中作为 GPQA family 的补充信号出现",
        default_few_shot=0,
        max_tokens_hint=64,
        choice_range="A-D",
        question_type="四选一高难 STEM 推理题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="truthfulqa",
        display_name="TruthfulQA",
        category="english",
        description="Measuring truthfulness in language models",
        default_few_shot=0,
        max_tokens_hint=32,
        choice_range="A-D",
        evalscope_name="truthful_qa",
        question_type="四选一真实性判断题",
        question_count_text="817 题",
    ),
    DatasetMeta(
        name="aime24",
        display_name="AIME24",
        category="math",
        description="AIME 2024 竞赛数学评测",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="开放式竞赛数学解答题",
        question_count_text="30 题",
    ),
    DatasetMeta(
        name="aime25",
        display_name="AIME25",
        category="math",
        description="AIME 2025/2026 家族竞赛数学评测",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="开放式竞赛数学解答题",
        question_count_text="30 题（2 个子集）",
    ),
    DatasetMeta(
        name="hle",
        display_name="HLE",
        category="reasoning",
        description="Humanity's Last Exam，高难综合知识与推理评测",
        default_few_shot=0,
        max_tokens_hint=64,
        question_type="开放式高难综合推理题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="aa_lcr",
        display_name="AA-LCR",
        category="long_context",
        description="长上下文长程推理评测",
        default_few_shot=0,
        max_tokens_hint=512,
        question_type="长上下文推理题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="live_code_bench",
        display_name="LiveCodeBench",
        category="code",
        description="代码生成与最新题目泛化能力评测",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="代码生成题",
        question_count_text="题量按 LiveCodeBench 版本滚动更新",
    ),
    DatasetMeta(
        name="swe_bench",
        display_name="SWE-bench",
        category="code",
        description="真实软件工程问题修复评测",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="软件工程缺陷修复题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="terminal_bench",
        display_name="Terminal-Bench",
        category="agent",
        description="终端 Agent 操作与执行评测",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="终端 Agent 操作题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="tau2_bench",
        display_name="TAU2-Bench",
        category="agent",
        description="通用 Agent 工具使用评测",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="Agent 工具使用题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="tool_bench",
        display_name="ToolBench",
        category="agent",
        description="工具调用与函数调用能力评测",
        default_few_shot=0,
        max_tokens_hint=1024,
        question_type="函数/工具调用题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="cmmmu",
        display_name="CMMMU",
        category="multimodal",
        description="中文多模态大学学科综合评测，包含图表、地图、乐谱、化学结构等视觉题型",
        default_few_shot=0,
        max_tokens_hint=256,
        choice_range="A-D",
        question_type="中文多模态选择/判断/填空题",
        question_count_text="题量未预置（30 个学科子集）",
    ),
    DatasetMeta(
        name="mmmu",
        display_name="MMMU",
        category="multimodal",
        description="多模态通识与视觉推理评测",
        default_few_shot=0,
        max_tokens_hint=256,
        question_type="多模态理解与视觉推理题",
        question_count_text="约 11500 题",
    ),
    DatasetMeta(
        name="mmmu_pro",
        display_name="MMMU-Pro",
        category="multimodal",
        description="MMMU 的更高难度版本",
        default_few_shot=0,
        max_tokens_hint=256,
        question_type="多模态高难推理题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="omnidoc_bench",
        display_name="OmniDoc Bench",
        category="multimodal",
        description="文档理解与版面理解评测",
        default_few_shot=0,
        max_tokens_hint=256,
        question_type="文档理解题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="math_vision",
        display_name="MathVision",
        category="multimodal",
        description="视觉数学理解评测",
        default_few_shot=0,
        max_tokens_hint=512,
        question_type="视觉数学解答题",
        question_count_text="题量未预置",
    ),
    DatasetMeta(
        name="wmt24",
        display_name="WMT24",
        category="multilingual",
        description="WMT 2024 机器翻译评测",
        default_few_shot=0,
        max_tokens_hint=512,
        question_type="机器翻译题",
        question_count_text="题量未预置",
    ),
]

_CATEGORY_TYPE_FALLBACK: dict[str, str] = {
    "chinese": "中文综合题",
    "english": "英文综合题",
    "math": "开放式数学题",
    "code": "代码生成题",
    "instruction": "指令遵循题",
    "reasoning": "高难推理题",
    "agent": "Agent 工具使用题",
    "multimodal": "多模态理解题",
    "multilingual": "多语种语言题",
    "long_context": "长上下文推理题",
    "discovered": "按 evalscope 默认格式计分",
}

_CHOICE_RANGE_TYPE_FALLBACK: dict[str, str] = {
    "A-B": "二选一选择题",
    "A-D": "四选一选择题",
    "A-J": "十选一选择题",
}

# Index by name for fast lookup
_DATASET_MAP: dict[str, DatasetMeta] = {d.name: d for d in _BUILTIN_DATASETS}


class DatasetRegistry:
    """Registry for evaluation datasets with search capability."""

    def __init__(self) -> None:
        self._datasets: dict[str, DatasetMeta] = dict(_DATASET_MAP)
        self._evalscope_datasets: list[str] = []

    def _ensure_evalscope_datasets_loaded(self) -> None:
        if self._evalscope_datasets:
            return

        self._evalscope_datasets = _discover_evalscope_datasets()
        for ds_name in self._evalscope_datasets:
            if ds_name not in self._datasets:
                self._datasets[ds_name] = DatasetMeta(
                    name=ds_name,
                    display_name=ds_name,
                    category="discovered",
                    description=f"Discovered from evalscope registry: {ds_name}",
                )

    def list_all(self) -> list[DatasetMeta]:
        self._ensure_evalscope_datasets_loaded()
        return list(self._datasets.values())

    def get(self, name: str) -> Optional[DatasetMeta]:
        self._ensure_evalscope_datasets_loaded()
        return self._datasets.get(name)

    def resolve_evalscope_id(self, dataset_name: str) -> str:
        meta = self.get(dataset_name)
        if meta:
            return meta.evalscope_id
        return dataset_name

    def get_recommended(self, dataset_names: list[str]) -> list[DatasetMeta]:
        """Return DatasetMeta objects for a list of names (preserving order)."""
        result = []
        for name in dataset_names:
            meta = self.get(name)
            if meta:
                result.append(meta)
        return result

    def search(self, query: str) -> list[DatasetMeta]:
        """Search datasets by keyword (fuzzy matching on name and display_name).

        Also tries to discover from evalscope benchmarks registry.
        """
        query_lower = query.strip().lower()
        if not query_lower:
            return self.list_all()

        self._ensure_evalscope_datasets_loaded()

        results = []
        seen: set[str] = set()
        for d in self._datasets.values():
            if (query_lower in d.name.lower()
                    or query_lower in d.display_name.lower()
                    or query_lower in d.description.lower()):
                if d.name not in seen:
                    results.append(d)
                    seen.add(d.name)
            elif d.subsets and any(query_lower in s.lower() for s in d.subsets):
                if d.name not in seen:
                    results.append(d)
                    seen.add(d.name)

        return results

    def _search_evalscope(self, query: str) -> list[str]:
        """Try to discover datasets from evalscope benchmarks."""
        self._ensure_evalscope_datasets_loaded()

        matches = []
        for ds_name in self._evalscope_datasets:
            if query in ds_name.lower():
                matches.append(ds_name)
        return matches

    def build_dataset_args(
        self,
        dataset_name: str,
        subsets: Optional[list[str]] = None,
        few_shot_num: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build evalscope dataset_args dict for a given dataset."""
        meta = self.get(dataset_name)
        evalscope_id = meta.evalscope_id if meta else dataset_name
        args: dict[str, Any] = {}

        if evalscope_id == "arc":
            args["subset_list"] = subsets or ["ARC-Challenge"]
        elif subsets:
            args["subset_list"] = subsets
        elif meta and meta.subsets:
            # Use default subsets if defined and no override
            pass  # let evalscope use all subsets

        if few_shot_num is not None:
            args["few_shot_num"] = few_shot_num
        elif meta and meta.default_few_shot > 0:
            args["few_shot_num"] = meta.default_few_shot

        if args:
            return {evalscope_id: args}
        return {}

    def max_tokens_for(self, dataset_name: str) -> int:
        """Return the recommended upper-bound max_tokens for a dataset."""
        meta = self.get(dataset_name)
        if meta:
            return meta.max_tokens_hint
        return 512

    def build_selected_summaries(self, dataset_names: list[str]) -> list[dict[str, str]]:
        summaries: list[dict[str, str]] = []
        for dataset_name in dataset_names:
            meta = self.get(dataset_name)
            if meta is None:
                summaries.append({
                    "title": dataset_name,
                    "question_type": "按 evalscope 默认格式计分",
                    "question_count": "题量未预置",
                    "description": "当前仅发现数据集名称，未补充本地摘要。",
                })
                continue

            summaries.append({
                "title": f"{meta.display_name} / {meta.name}",
                "question_type": self._resolve_question_type(meta),
                "question_count": self._resolve_question_count(meta),
                "description": meta.description or "当前未填写数据集简介。",
            })
        return summaries

    def _resolve_question_type(self, meta: DatasetMeta) -> str:
        if meta.question_type:
            return meta.question_type
        if meta.choice_range:
            return _CHOICE_RANGE_TYPE_FALLBACK.get(meta.choice_range, "选择题")
        return _CATEGORY_TYPE_FALLBACK.get(meta.category, "综合评测题")

    def _resolve_question_count(self, meta: DatasetMeta) -> str:
        if meta.question_count_text:
            return meta.question_count_text
        if meta.subsets:
            return f"题量未预置（{len(meta.subsets)} 个官方子集）"
        return "题量未预置"


def _discover_evalscope_datasets() -> list[str]:
    """Try to list available evalscope benchmark names."""
    try:
        import evalscope.benchmarks as benchmarks
        if hasattr(benchmarks, "BENCHMARK_REGISTRY"):
            return sorted(benchmarks.BENCHMARK_REGISTRY.keys())
        # Fallback: scan submodules
        import pkgutil
        names = []
        for importer, modname, ispkg in pkgutil.walk_packages(
            benchmarks.__path__, prefix=benchmarks.__name__ + "."
        ):
            short = modname.split(".")[-1]
            if short.endswith("_adapter"):
                names.append(short.replace("_adapter", ""))
        return sorted(set(names))
    except Exception:
        logger.debug("evalscope benchmarks discovery failed, using built-in list only")
        return []
