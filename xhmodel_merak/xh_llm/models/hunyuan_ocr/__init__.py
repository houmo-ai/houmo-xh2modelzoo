# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from .dflash_draft import (
    HunyuanOCRDFlashCheckpoint,
    HunyuanOCRDFlashConfig,
    HunyuanOCRDFlashModel,
    HunyuanOCRDFlashOutputHead,
    HunyuanOCRDraftCacheController,
    build_dflash_noise_embedding,
    build_draft_position_ids,
    build_hunyuan_ocr_dflash_contract,
    build_hunyuan_ocr_dflash_export_adapter,
    dflash_graph_io_contract,
    load_hunyuan_ocr_dflash_checkpoint,
)
from .hunyuan_ocr_dflash_model import XHHunYuanOCRDFlashModel
from .hunyuan_ocr_llm_model import (
    HunyuanOCRModelMeta,
    HunyuanOCRTextExportMeta,
    HunyuanOCRVisualMeta,
    XHHunYuanOCRModel,
)
from .hunyuan_ocr_processor import HunyuanOCRMultiBucketProcessor
from .hunyuan_ocr_speculative_runtime import (
    HunyuanOCRDraftGraphs,
    HunyuanOCRSpeculativeDecoder,
    HunyuanOCRSpeculativeRuntimeError,
    SpeculativeStats,
    load_draft_graphs,
    resolve_draft_runtime_contract,
)
from .hunyuan_ocr_visual_model import XHHunYuanOCRVisualModel
from .target_verify_runtime import (
    HunyuanOCRTargetVerifyController,
    TargetVerifyResult,
    commit_verify_prefix_length,
)
from .xh_hunyuan_ocr_config import (
    XHHunYuanOCRDFlashConfig,
    XHHunYuanOCRModelConfig,
    XHHunYuanOCRVisualConfig,
)


if TYPE_CHECKING:
    from .workflow import HunyuanOCRWorkflow


__all__ = [
    "HunyuanOCRWorkflow",
    "HunyuanOCRModelMeta",
    "HunyuanOCRMultiBucketProcessor",
    "HunyuanOCRTextExportMeta",
    "HunyuanOCRTargetVerifyController",
    "HunyuanOCRDraftGraphs",
    "HunyuanOCRSpeculativeDecoder",
    "HunyuanOCRSpeculativeRuntimeError",
    "SpeculativeStats",
    "HunyuanOCRDFlashCheckpoint",
    "HunyuanOCRDFlashConfig",
    "HunyuanOCRDFlashModel",
    "HunyuanOCRDFlashOutputHead",
    "HunyuanOCRDraftCacheController",
    "HunyuanOCRVisualMeta",
    "TargetVerifyResult",
    "XHHunYuanOCRModel",
    "XHHunYuanOCRModelConfig",
    "XHHunYuanOCRDFlashConfig",
    "XHHunYuanOCRDFlashModel",
    "XHHunYuanOCRVisualConfig",
    "XHHunYuanOCRVisualModel",
    "commit_verify_prefix_length",
    "load_draft_graphs",
    "resolve_draft_runtime_contract",
    "build_draft_position_ids",
    "build_dflash_noise_embedding",
    "build_hunyuan_ocr_dflash_contract",
    "build_hunyuan_ocr_dflash_export_adapter",
    "dflash_graph_io_contract",
    "load_hunyuan_ocr_dflash_checkpoint",
]


def __getattr__(name: str):
    if name == "HunyuanOCRWorkflow":
        from .workflow import HunyuanOCRWorkflow

        return HunyuanOCRWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
