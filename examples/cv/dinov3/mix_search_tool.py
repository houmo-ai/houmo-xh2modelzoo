# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Thin bridge: xhquant MixPrecisionSearch + detection-model extensions.

Inherits from xhquant's ``MixPrecisionSearch`` and reuses:

- ``run()`` – graph interpreter + candidate collection
- ``calculate_sensitivity()`` – L1 / SQNR / KL metrics

Only ``compute_loss`` (detection proxy vector) and ``search`` are overridden,
because the base class hard-codes candidate classes and misses several
XH2a ops used by LT-DETR.

The detector search is two-stage:

1. Build a W16A8 upper reference, then degrade each weighted node to W8A8 and
   select enough W16 nodes to cover the observed degradation budget.
2. Fix the selected W bits, build a W-fixed/A16 upper reference, then degrade
   each TE/weighted candidate's A side to A8 and select enough A16 nodes to
   cover the observed degradation budget.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any, Sequence

import torch
from tqdm import tqdm

_XHQUANT_IMPORT_ERROR: ImportError | None = None
try:
    from xhquant.common.types import PrecisionMode
    from xhquant.core import QBaseModule
    from xhquant.core.graph.quant_graph import enable_quanted_module
    from xhquant.mix_precision.mix_precision import MixPrecisionSearch
    from xhquant.quantization.xh2a.qmodules import QConv2d, QConvTranspose2d, QLinear, QMoeBlock
    from xhquant.utils.logger import get_root_logger
except ImportError as exc:
    _XHQUANT_IMPORT_ERROR = exc
    PrecisionMode = None  # type: ignore[assignment]
    QBaseModule = ()  # type: ignore[assignment]
    enable_quanted_module = None  # type: ignore[assignment]
    QConv2d = QConvTranspose2d = QLinear = QMoeBlock = ()  # type: ignore[assignment]

    class MixPrecisionSearch:  # type: ignore[no-redef]
        """Import-time placeholder so detector_output_vector remains testable without xhquant."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_xhquant()

    def get_root_logger():  # type: ignore[no-redef]
        import logging

        return logging.getLogger(__name__)


def _require_xhquant() -> None:
    """Raise the original xhquant import error when search functionality is used."""
    if _XHQUANT_IMPORT_ERROR is not None:
        raise ImportError("xhquant is required for DINOv3 mixed precision search.") from _XHQUANT_IMPORT_ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# 默认加权候选算子（6 类，覆盖 QMatMul/QGroupMatMul）
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTED_CANDIDATES = (
    "XH2aQuantQLinear",
    "XH2aQuantQConv2d",
    "XH2aQuantQConvTranspose2d",
    "XH2aQuantQMoeBlock",
    "XH2aQuantQMatMul",
    "XH2aQuantQGroupMatMul",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 检测模型输出向量化（sensitivity proxy）
# ═══════════════════════════════════════════════════════════════════════════════


def detector_output_vector(
    output: Any,
    *,
    box_scale: float = 640.0,
    score_weight: float = 1.0,
    label_weight: float = 0.0,
) -> torch.Tensor:
    """将检测输出 (labels, boxes, scores) 展平为一维向量作为 sensitivity proxy。"""

    def _as_tensor(v: Any) -> Any:
        return v._data if hasattr(v, "_data") else v

    labels = None
    if isinstance(output, dict):
        labels = _as_tensor(output.get("labels"))
        boxes = _as_tensor(output.get("boxes"))
        scores = _as_tensor(output.get("scores"))
    elif isinstance(output, (list, tuple)) and len(output) >= 3:
        labels, boxes, scores = _as_tensor(output[0]), _as_tensor(output[1]), _as_tensor(output[2])
    elif isinstance(output, torch.Tensor):
        return torch.nan_to_num(output.float()).reshape(-1)
    else:
        raise ValueError(f"Unsupported detector output type: {type(output)}")

    if boxes is None or scores is None:
        raise ValueError("Detector output must provide boxes and scores.")
    vectors = [
        torch.nan_to_num(boxes.float() / float(box_scale)).reshape(-1),
        torch.nan_to_num(scores.float() * float(score_weight)).reshape(-1),
    ]
    if labels is not None and label_weight > 0:
        vectors.append(torch.nan_to_num(labels.float() / 80.0 * float(label_weight)).reshape(-1))
    return torch.cat(vectors, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# DetectorMixPrecisionSearchV2
# ═══════════════════════════════════════════════════════════════════════════════


class DetectorMixPrecisionSearchV2(MixPrecisionSearch):
    """继承 xhquant MixPrecisionSearch，最小化定制检测模型混合精度搜索。

    复用基类: run() / calculate_sensitivity() / config 管理
    重写:   compute_loss / search / get_search_result

    第一阶段只搜索加权算子的 W16，并按逐节点降到 W8 的退化量逼近 W16A8 上限。
    第二阶段在固定 W 结果后只搜索 TE/weighted 候选的 A16，并按逐节点降到 A8 的退化量逼近 W-fixed/A16 上限。
    """

    weighted_candidate_ops: list[Any]

    # ------------------------------------------------------------------
    # compute_loss: 检测输出向量化
    # ------------------------------------------------------------------

    def compute_loss(self, output: Any, label: Any = None):
        """重写：检测模型用输出向量 proxy 代替标量 loss。"""
        if self.config.get("task") == "cv_det":
            return detector_output_vector(
                output,
                box_scale=float(self.config.get("box_scale", 640.0)),
                score_weight=float(self.config.get("score_weight", 1.0)),
                label_weight=float(self.config.get("label_weight", 0.0)),
            )
        return super().compute_loss(output, label)

    # ------------------------------------------------------------------
    # 候选收集
    # ------------------------------------------------------------------

    @staticmethod
    def _module_name(module: Any) -> str:
        return module._get_name() if hasattr(module, "_get_name") else type(module).__name__

    def _is_weighted(self, module: Any) -> bool:
        """有权重候选：在搜索列表中且有 w_cfg 或 i_cfg_2。"""
        candidates = set(self.config.get("search_candidates", []))
        if "*" in candidates:
            return hasattr(module, "w_cfg") or hasattr(module, "i_cfg_2")
        name = self._module_name(module)
        bare = name.removeprefix("XH2aQuant")
        if name not in candidates and bare not in candidates and type(module).__name__ not in candidates:
            return False
        return hasattr(module, "w_cfg") or hasattr(module, "i_cfg_2")

    def _collect_candidates(self) -> None:
        """遍历量化图，只收集 TE/weighted 候选节点。"""
        weighted = []
        for node in self.graph.nodes:
            if node.op != "call_module":
                continue
            module = self.module.get_submodule(str(node.target))
            if isinstance(module, QBaseModule) and self._is_weighted(module):
                weighted.append(node)
        self.weighted_candidate_ops = weighted

    # ------------------------------------------------------------------
    # 量化配置快照 / 恢复 / 设置
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot(module: Any) -> dict[str, Any]:
        """快照模块的所有量化配置（含 o_quantizer）。"""
        snap: dict[str, Any] = {}
        for attr in ("w_cfg", "i_cfg", "i_cfg_1", "i_cfg_2", "o_cfg"):
            if hasattr(module, attr):
                snap[attr] = copy.deepcopy(getattr(module, attr))
        if hasattr(module, "o_quantizer"):
            snap["o_quantizer"] = copy.deepcopy(module.o_quantizer)
        return snap

    @staticmethod
    def _restore(module: Any, snap: dict[str, Any]) -> None:
        """从快照恢复量化配置。"""
        for attr, value in snap.items():
            setattr(module, attr, value)
        if "o_cfg" in snap and hasattr(module, "o_quantizer"):
            from xhquant.core.builder import build_quantizer

            module.o_quantizer = build_quantizer(module.o_cfg)

    @staticmethod
    def _unique_cfg_attrs(module: Any, attrs: Sequence[str]) -> list[str]:
        """去重 cfg 属性（i_cfg/i_cfg_1 可能指向同一对象）。"""
        unique, seen = [], set()
        for attr in attrs:
            if not hasattr(module, attr):
                continue
            cfg_id = id(getattr(module, attr))
            if cfg_id in seen:
                continue
            seen.add(cfg_id)
            unique.append(attr)
        return unique

    @staticmethod
    def _rebuild_output_quantizer(module: Any) -> None:
        """修改 o_cfg 后重建 o_quantizer。"""
        if hasattr(module, "o_cfg") and hasattr(module, "o_quantizer"):
            from xhquant.core.builder import build_quantizer

            module.o_quantizer = build_quantizer(module.o_cfg)

    @staticmethod
    def _set_weight_bit(module: Any, bit: int) -> bool:
        """设置权重量化 bit 宽度（w_cfg 或 i_cfg_2）。"""
        changed = False
        for attr in DetectorMixPrecisionSearchV2._unique_cfg_attrs(module, ("w_cfg", "i_cfg_2")):
            cfg = getattr(module, attr, None)
            qspec = getattr(cfg, "qspec", None) if cfg else None
            if qspec and hasattr(qspec, "man_bit"):
                qspec.man_bit = int(bit)
                changed = True
        return changed

    @staticmethod
    def _set_activation_bit(module: Any, bit: int) -> bool:
        """设置激活量化 bit 宽度（含 o_cfg 重建）。"""
        changed = False
        for attr in DetectorMixPrecisionSearchV2._unique_cfg_attrs(module, ("i_cfg", "i_cfg_1", "o_cfg")):
            cfg = getattr(module, attr, None)
            qspec = getattr(cfg, "qspec", None) if cfg else None
            if qspec and hasattr(qspec, "man_bit"):
                qspec.man_bit = int(bit)
                changed = True
                if attr == "o_cfg":
                    DetectorMixPrecisionSearchV2._rebuild_output_quantizer(module)
        return changed

    @staticmethod
    def _fix_module(module: Any) -> None:
        """固定模块量化状态：清除 → 触发量化 → 权重静态量化 → 固定。"""
        if isinstance(module, (QLinear, QConv2d, QConvTranspose2d, QMoeBlock)):
            module.clear_quant_state()
            module.fire_quant()
            module.weight_static_quant()
            module.fixed_quant()
        elif hasattr(module, "fixed_quant"):
            module.fixed_quant()

    @staticmethod
    def _refresh_weight_quant_for_scan(module: Any) -> None:
        """扫描 W bit 时重建权重量化缓存，确保 W8/W16 forward 真正生效。"""
        if hasattr(module, "clear_quant_state"):
            module.clear_quant_state()
        if hasattr(module, "fire_quant"):
            module.fire_quant()
        if hasattr(module, "weight_static_quant"):
            module.weight_static_quant()

    @contextlib.contextmanager
    def _temporary_quant_nodes(self, nodes: Sequence[Any]):
        """搜索 forward 专用：临时启用多个候选节点量化，并把每个节点输出反量化。"""
        self.module.disable_quant()
        seen: set[int] = set()
        with contextlib.ExitStack() as stack:
            for node in nodes:
                module = self.module.get_submodule(str(node.target))
                if not isinstance(module, QBaseModule):
                    continue
                module_id = id(module)
                if module_id in seen:
                    continue
                seen.add(module_id)
                stack.enter_context(enable_quanted_module(module))
            yield

    # ------------------------------------------------------------------
    # search: W/A 两阶段解耦搜索
    # ------------------------------------------------------------------

    @torch.no_grad()
    def search(
        self,
        data_loader: Sequence[Any],
        origin_mode: Any,
        labels: Any = None,
        wrap_cfg: Any = None,
        initial_env: Any = None,
        enable_io_processing: bool = True,
    ) -> None:
        """主搜索入口：先选 W16，再在固定 W 后选择 A16。"""
        if labels is not None or wrap_cfg is not None or initial_env is not None or not enable_io_processing:
            raise NotImplementedError("Detector search only supports plain sampled image batches.")
        assert isinstance(data_loader, (list, tuple)) and len(data_loader) > 0

        self._collect_candidates()
        max_w_bit = max(self.config["weight_bits"])
        min_w_bit = min(self.config["weight_bits"])
        max_a_bit = max(self.config["act_bits"])
        min_a_bit = min(self.config["act_bits"])

        # --- Stage 1: full W16A8 reference, then degrade one weighted node to W8 ---
        stage1_nodes = list(self.weighted_candidate_ops)
        for node in self.weighted_candidate_ops:
            module = self.module.get_submodule(node.target)
            self._set_weight_bit(module, max_w_bit)
            self._set_activation_bit(module, min_a_bit)
            module.precision_mode = PrecisionMode.FAST
            self._fix_module(module)

        weighted_sens: dict[Any, dict[str, float]] = {}
        w_degrade_key = f"w{min_w_bit}_degradation_to_w{max_w_bit}a{min_a_bit}_upper"
        weighted_space = len(data_loader) * len(self.weighted_candidate_ops)
        pbar = tqdm(total=max(1, weighted_space), desc="stage1 W8 degradation vs W16A8 upper", dynamic_ncols=True)

        try:
            with self._temporary_quant_nodes(stage1_nodes):
                for data in data_loader:
                    if hasattr(data, "shape"):
                        data = [data]

                    output_w_upper = self.run(*data)
                    if self.config["key_name"] == "loss":
                        output_w_upper = self.compute_loss(output_w_upper, None)
                        assert output_w_upper is not None

                    for node in self.weighted_candidate_ops:
                        module = self.module.get_submodule(node.target)
                        old_mode = module.precision_mode
                        module.precision_mode = PrecisionMode.FAST
                        try:
                            self._set_weight_bit(module, min_w_bit)
                            self._set_activation_bit(module, min_a_bit)
                            self._fix_module(module)
                            output_w_low = self.run(*data)
                            if self.config["key_name"] == "loss":
                                output_w_low = self.compute_loss(output_w_low, None)
                                assert output_w_low is not None
                            weighted_sens.setdefault(node, {})
                            weighted_sens[node][w_degrade_key] = (
                                weighted_sens[node].get(w_degrade_key, 0.0)
                                + self.calculate_sensitivity(output_w_upper, output_w_low)
                            )
                        finally:
                            module.precision_mode = old_mode
                            self._set_weight_bit(module, max_w_bit)
                            self._set_activation_bit(module, min_a_bit)
                            self._fix_module(module)
                            pbar.update(1)
        finally:
            pbar.close()

        w_scores = {
            node: float(values.get(w_degrade_key, 0.0)) / float(len(data_loader))
            for node, values in weighted_sens.items()
        }
        selected_w = self._select_by_scores(
            w_scores,
            count=self.config.get("weighted_count"),
            topk=self.config.get("weighted_topk"),
            target_ratio=self.config.get("weighted_target_ratio"),
            default_topk=self.config["topk"],
            default_target_ratio=self.config["target_ratio"],
        )

        # --- Stage 2: selected W fixed, full A16 reference, then degrade one TE node's A side to A8 ---
        te_a_nodes = list(self.weighted_candidate_ops)
        for node in self.weighted_candidate_ops:
            module = self.module.get_submodule(node.target)
            self._set_weight_bit(module, max_w_bit if node in selected_w else min_w_bit)
            self._set_activation_bit(module, max_a_bit)
            module.precision_mode = PrecisionMode.FAST
            self._fix_module(module)

        te_a_sens: dict[Any, dict[str, float]] = {}
        a_degrade_key = f"a{min_a_bit}_degradation_to_a{max_a_bit}_upper"
        te_a_space = len(data_loader) * len(te_a_nodes)
        pbar = tqdm(total=max(1, te_a_space), desc="stage2 TE A8 degradation vs W-fixed A16 upper", dynamic_ncols=True)

        try:
            with self._temporary_quant_nodes(te_a_nodes):
                for data in data_loader:
                    if hasattr(data, "shape"):
                        data = [data]

                    output_a_upper = self.run(*data)
                    if self.config["key_name"] == "loss":
                        output_a_upper = self.compute_loss(output_a_upper, None)
                        assert output_a_upper is not None

                    for node in te_a_nodes:
                        module = self.module.get_submodule(node.target)
                        old_mode = module.precision_mode
                        module.precision_mode = PrecisionMode.FAST
                        w_bit = max_w_bit if node in selected_w else min_w_bit
                        try:
                            self._set_weight_bit(module, w_bit)
                            self._set_activation_bit(module, min_a_bit)
                            self._fix_module(module)

                            output_a_low = self.run(*data)
                            if self.config["key_name"] == "loss":
                                output_a_low = self.compute_loss(output_a_low, None)
                                assert output_a_low is not None
                            te_a_sens.setdefault(node, {})
                            te_a_sens[node][a_degrade_key] = (
                                te_a_sens[node].get(a_degrade_key, 0.0)
                                + self.calculate_sensitivity(output_a_upper, output_a_low)
                            )
                        finally:
                            module.precision_mode = old_mode
                            self._set_weight_bit(module, w_bit)
                            self._set_activation_bit(module, max_a_bit)
                            self._fix_module(module)
                            pbar.update(1)
        finally:
            pbar.close()

        te_a_scores = {
            node: float(values.get(a_degrade_key, 0.0)) / float(len(data_loader))
            for node, values in te_a_sens.items()
        }
        selected_te_a = self._select_by_scores(
            te_a_scores,
            count=self.config.get("te_a_count"),
            topk=self.config.get("te_a_topk"),
            target_ratio=self.config.get("te_a_target_ratio"),
            default_topk=self.config["topk"],
            default_target_ratio=self.config["target_ratio"],
        )

        self.module.enable_quant(origin_mode)
        for node in self.weighted_candidate_ops:
            module = self.module.get_submodule(node.target)
            self._set_weight_bit(module, max_w_bit if node in selected_w else min_w_bit)
            self._set_activation_bit(module, max_a_bit if node in selected_te_a else min_a_bit)
            module.precision_mode = PrecisionMode.FAST
            self._fix_module(module)

        weighted_selection = self._selection_summary(
            w_scores,
            selected_w,
            count=self.config.get("weighted_count"),
            topk=self.config.get("weighted_topk"),
            target_ratio=self.config.get("weighted_target_ratio"),
            default_topk=self.config["topk"],
            default_target_ratio=self.config["target_ratio"],
        )
        te_a_selection = self._selection_summary(
            te_a_scores,
            selected_te_a,
            count=self.config.get("te_a_count"),
            topk=self.config.get("te_a_topk"),
            target_ratio=self.config.get("te_a_target_ratio"),
            default_topk=self.config["topk"],
            default_target_ratio=self.config["target_ratio"],
        )

        # --- 保存报告 ---
        self.sensitivity_report = {
            "stage": "w_then_a",
            "selection_policy": self.config["policy"],
            "target_ratio": self.config["target_ratio"],
            "weighted_target_ratio": self.config.get("weighted_target_ratio"),
            "te_a_target_ratio": self.config.get("te_a_target_ratio"),
            "weighted_selection": weighted_selection,
            "te_a_selection": te_a_selection,
            "weighted_sensitivity": {
                str(n.target): {k: v / len(data_loader) for k, v in vals.items()}
                for n, vals in weighted_sens.items()
            },
            "te_a_sensitivity": {
                str(n.target): {k: v / len(data_loader) for k, v in vals.items()}
                for n, vals in te_a_sens.items()
            },
            "weighted_gain": {str(n.target): score for n, score in w_scores.items()},
            "te_a_gain": {str(n.target): score for n, score in te_a_scores.items()},
            "selected_weight": [str(n.target) for n in sorted(selected_w, key=lambda n: n.target)],
            "selected_te_a": [str(n.target) for n in sorted(selected_te_a, key=lambda n: n.target)],
        }

        # --- 日志 ---
        from prettytable import PrettyTable

        w_high = sum(1 for n in self.weighted_candidate_ops if n in selected_w)
        w_low = len(self.weighted_candidate_ops) - w_high
        a_high = sum(1 for n in te_a_nodes if n in selected_te_a)
        a_low = len(te_a_nodes) - a_high
        table = PrettyTable(["stage", "high", "low", "total"])
        table.add_row([f"weight w{max_w_bit}", w_high, w_low, len(self.weighted_candidate_ops)])
        table.add_row([f"te a{max_a_bit}", a_high, a_low, len(te_a_nodes)])
        get_root_logger().info("detector mix precision result:\n%s", table)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 选择策略：coverage / topk / threshold
    # ------------------------------------------------------------------

    def _select_by_scores(
        self,
        score_map: dict[Any, float],
        *,
        count: int | None,
        topk: float | None,
        target_ratio: float | None,
        default_topk: float,
        default_target_ratio: float,
    ) -> set[Any]:
        """按收益降序选节点。

        - threshold: 选择收益超过阈值的节点。
        - topk:      保留固定比例/数量。
        - coverage:  保留最少节点，使累计正收益达到目标比例，用于逼近当前阶段上限。
        """
        scores = [(node, max(0.0, float(score))) for node, score in score_map.items()]
        if self.config["policy"] == "threshold":
            return {node for node, score in scores if score > self.config["threshold"]}

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        if count is not None and count >= 0:
            cutoff = min(len(sorted_scores), count)
        elif self.config["policy"] == "coverage":
            ratio = default_target_ratio if target_ratio is None else float(target_ratio)
            ratio = min(1.0, max(0.0, ratio))
            total_gain = sum(score for _node, score in sorted_scores if score > 0)
            if total_gain <= 0 or ratio <= 0:
                cutoff = 0
            else:
                target_gain = total_gain * ratio
                cumulative_gain = 0.0
                cutoff = 0
                for cutoff, (_node, score) in enumerate(sorted_scores, start=1):
                    cumulative_gain += max(0.0, score)
                    if cumulative_gain >= target_gain:
                        break
        elif topk is not None:
            cutoff = min(len(sorted_scores), int(len(sorted_scores) * topk) + 1)
        else:
            cutoff = min(len(sorted_scores), int(len(sorted_scores) * default_topk) + 1)
        return {node for node, _ in sorted_scores[:cutoff]}

    def _selection_summary(
        self,
        score_map: dict[Any, float],
        selected: set[Any],
        *,
        count: int | None,
        topk: float | None,
        target_ratio: float | None,
        default_topk: float,
        default_target_ratio: float,
    ) -> dict[str, Any]:
        """记录选择结果的覆盖率，用于判断是否逼近对应阶段上限。"""
        scores = {node: max(0.0, float(score)) for node, score in score_map.items()}
        total_gain = sum(score for score in scores.values() if score > 0)
        selected_gain = sum(scores.get(node, 0.0) for node in selected)
        effective_target_ratio = default_target_ratio if target_ratio is None else float(target_ratio)
        effective_topk = default_topk if topk is None else float(topk)
        if count is not None and count >= 0:
            selector = "count"
        else:
            selector = self.config["policy"]
        return {
            "selector": selector,
            "target_ratio": min(1.0, max(0.0, effective_target_ratio)),
            "topk": effective_topk,
            "count": count,
            "num_candidates": len(scores),
            "num_positive_gain": sum(1 for score in scores.values() if score > 0),
            "num_selected": len(selected),
            "total_positive_gain": total_gain,
            "selected_positive_gain": selected_gain,
            "selected_gain_ratio": (selected_gain / total_gain) if total_gain > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # get_search_result
    # ------------------------------------------------------------------

    def get_search_result(self) -> dict[str, Any]:
        """导出搜索结果：每个加权节点的最终精度配置。"""
        if not hasattr(self, "weighted_candidate_ops"):
            raise ValueError("Call search() before get_search_result().")
        result: dict[str, Any] = {}
        for op in self.weighted_candidate_ops:
            module = self.module.get_submodule(op.target)
            settings: dict[str, Any] = {"kind": "weighted"}
            for attr in ("w_cfg", "i_cfg", "i_cfg_1", "i_cfg_2", "o_cfg"):
                if hasattr(module, attr):
                    settings[attr] = getattr(module, attr)
            result[op.target] = settings
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷工厂
# ═══════════════════════════════════════════════════════════════════════════════


def make_detector_searcher(
    module: torch.nn.Module,
    config: dict[str, Any] | None = None,
) -> DetectorMixPrecisionSearchV2:
    """创建检测模型的混合精度搜索器，预设合理的默认配置（6 类加权候选）。"""
    _require_xhquant()
    default_config = {
        "weight_bits": [8, 16],
        "act_bits": [8, 16],
        "policy": "coverage",
        "topk": 0.2,
        "target_ratio": 1.0,
        "weighted_topk": None,
        "weighted_count": None,
        "weighted_target_ratio": None,
        "te_a_topk": None,
        "te_a_count": None,
        "te_a_target_ratio": None,
        "metric": "l1",
        "key_name": "loss",
        "task": "cv_det",
        "box_scale": 640.0,
        "score_weight": 1.0,
        "label_weight": 0.25,
        "search_candidates": list(DEFAULT_WEIGHTED_CANDIDATES),
    }
    if config:
        default_config.update(config)
    return DetectorMixPrecisionSearchV2(module, default_config)
