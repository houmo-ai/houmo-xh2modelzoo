"""Structured help and model documentation for the public Qwen3.5/Qwen3.6 workflow API."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .workflow_defaults import (
    export_config_template,
    get_default_quant_config,
    quant_config_template,
    repo_root,
)


_DOC_ROOT = Path("docs")
_HMONNX_IO_DOC = _DOC_ROOT / "qwen3_5_hmonnx_io_spec.md"

_QUANT_FIELD_HELP: dict[str, dict[str, Any]] = {
    "algorithm": {
        "type": "str",
        "default": "autoround",
        "description": "量化入口。默认运行 AutoRound，并保存 GPTQModel 兼容的 HuggingFace 目录。",
    },
    "output_format": {
        "type": "str",
        "default": "gptqmodel_hf",
        "description": "AutoRound 保存格式别名；当前 Qwen3.5/Qwen3.6 工作流要求与 artifact_format 一致。",
    },
    "artifact_format": {
        "type": "str",
        "default": "gptqmodel_hf",
        "description": "quant() 返回产物的语义格式；export() 按该格式选择加载路径。",
    },
    "bits": {
        "type": "int",
        "default": 4,
        "description": "AutoRound 目标权重量化 bit。xhquant 内部导出算子会再根据权重范围选择 w4/w8 细节。",
    },
    "group_size": {
        "type": "int",
        "default": 64,
        "description": "Qwen3.5/Qwen3.6 量化强约束，必须为 64。",
        "required": True,
    },
    "sym": {
        "type": "bool",
        "default": True,
        "description": "AutoRound 对称量化开关。",
    },
    "iters": {
        "type": "int",
        "default": 200,
        "description": "AutoRound 每层优化迭代次数。",
    },
    "seed": {
        "type": "int",
        "default": 42,
        "description": "AutoRound 随机种子，与现有 mode1 shell 脚本保持一致。",
    },
    "quant_nontext_module": {
        "type": "bool",
        "default": False,
        "description": "LLM-only 量化开关：False 表示不量化视觉/非文本模块。",
    },
    "autoround_format": {
        "type": "str",
        "default": "auto_gptq",
        "description": "AutoRound 上游保存格式名；dense 脚本默认为 auto_gptq，MoE YAML 使用 auto_round:gptqmodel。",
    },
    "save_path": {
        "type": "str|null",
        "default": None,
        "description": "可选：强制指定量化产物目录。默认使用 output_dir/<model>-autoround-gptqmodel。",
    },
    "calibration.dataset": {
        "type": "str",
        "default": "NeelNanda/pile-10k",
        "description": "AutoRound 校准数据集名称，与 mode1 LLM-only shell 脚本保持一致。",
    },
    "calibration.nsamples": {
        "type": "int",
        "default": 128,
        "description": "AutoRound 校准样本数。",
    },
    "calibration.seqlen": {
        "type": "int",
        "default": 2048,
        "description": "AutoRound 校准序列长度。",
    },
    "runtime.batch_size": {
        "type": "int",
        "default": 8,
        "description": "AutoRound 校准 batch size，与 mode1 LLM-only shell 脚本保持一致。",
    },
    "runtime.trust_remote_code": {
        "type": "bool",
        "default": True,
        "description": "加载 HuggingFace 模型时是否允许 remote code。",
    },
    "runtime.device_map": {
        "type": "str|null",
        "default": "MoE: balanced",
        "description": "MoE AutoRound 多卡切分策略；dense YAML 默认不设置。",
    },
    "runtime.low_gpu_mem_usage": {
        "type": "bool",
        "default": "MoE: True",
        "description": "MoE 低显存加载开关；dense YAML 默认不设置。",
    },
    "moe.attn_bits": {
        "type": "int|null",
        "default": "MoE: 8",
        "description": "MoE attention linears 的 bit override，对齐 scripts_qwen35moe mode1。",
    },
    "moe.shared_expert_bits": {
        "type": "int|null",
        "default": "MoE: 8",
        "description": "MoE shared_expert gate/up/down projection 的 bit override。",
    },
    "moe.expert_bits": {
        "type": "int|null",
        "default": None,
        "description": "可选：MoE 非 shared experts gate/up/down 的统一 bit override。",
    },
    "moe.expert_up_gate_bits": {
        "type": "int|null",
        "default": None,
        "description": "可选：MoE 非 shared experts gate_proj/up_proj 的 bit override。",
    },
    "moe.expert_down_bits": {
        "type": "int|null",
        "default": None,
        "description": "可选：MoE 非 shared experts down_proj 的 bit override。",
    },
    "existing_hf.algorithm": {
        "type": "str",
        "default": "existing_hf",
        "description": "跳过量化，复用外部已经量化好的 HF/GPTQModel 目录。",
    },
    "existing_hf.existing_hf_model_dir": {
        "type": "str",
        "default": "weights/<existing-gptqmodel-hf-dir>",
        "description": "外部量化产物目录，例如 9B mode1 或 35B-A3B autoround/gptqmodel 结果。",
    },
    "existing_hf.source_algorithm": {
        "type": "str",
        "default": "autoround",
        "description": "说明外部权重来源算法；结果写入 QuantResult.algorithm 方便上游记录。",
    },
}

_EXPORT_FIELD_HELP: dict[str, dict[str, Any]] = {
    "export.model.model_type": {
        "type": "str",
        "default": "Qwen3_5ForConditionalGeneration 或 Qwen3_5MoeForConditionalGeneration",
        "description": "模型实现路由。dense、MoE、visual-only 使用不同 model_type。",
    },
    "export.model.hf_model": {
        "type": "str",
        "default": "来自推荐 YAML",
        "description": "原始 HuggingFace 模型目录。Workflow.build_export_dict 会用 quant 结果覆盖导出源。",
    },
    "export.model.model_name": {
        "type": "str",
        "default": "xh2_<model>_<variant>_256_2k",
        "description": "导出产物命名前缀，随推荐 YAML 固定。",
    },
    "export.model.context_max_length": {
        "type": "int",
        "default": 2048,
        "description": "KV cache 最大长度，影响 prefill/decode HMONNX cache shape。",
    },
    "export.model.prefill_chunk_length": {
        "type": "int",
        "default": 256,
        "description": "Prefill 图静态输入长度。",
    },
    "export.model.max_pe_length": {
        "type": "int",
        "default": 262144,
        "description": "RoPE/position embedding 上限。",
    },
    "export.model.quant_scheme.quant_type": {
        "type": "str",
        "default": "w8a8h1_sefp",
        "description": "Merak 导出默认 xhquant scheme。不要把 w4/w8 写进 YAML 文件名。",
    },
    "export.model.fuse_gdr_ops": {
        "type": "bool",
        "default": False,
        "description": "GDR fuse 开关。默认 False；编译器支持成熟后可用 override 或 YAML 改为 True。",
    },
    "export.model.visual_config.max_size_w": {
        "type": "int",
        "default": 448,
        "description": "full 导出内置视觉分支的最大宽度；推荐桶为 448 和 896。",
    },
    "export.model.visual_config.max_size_h": {
        "type": "int",
        "default": 448,
        "description": "full 导出内置视觉分支的最大高度；推荐桶为 448 和 896。",
    },
    "export.model.max_size_w": {
        "type": "int",
        "default": "448 或 896",
        "description": "visual_only YAML 的视觉塔宽度。",
    },
    "export.model.max_size_h": {
        "type": "int",
        "default": "448 或 896",
        "description": "visual_only YAML 的视觉塔高度。",
    },
    "export.model.spec_decode_mode": {
        "type": "str|null",
        "default": None,
        "description": "None=常规 full；mtp=MTP 投机解码；dflash=DFlash 投机解码。",
    },
    "export.model.num_draft_tokens": {
        "type": "int",
        "default": "MTP 4；DFlash 9",
        "description": "每轮 draft token 数。",
    },
    "export.model.mtp_config": {
        "type": "dict",
        "default": "*_full_mtp.yaml 中提供",
        "description": "MTP draft 模型配置；默认 hf_model 由主模型路径继承。",
    },
    "export.model.dflash_config.target_model_dir": {
        "type": "str|null",
        "default": None,
        "description": "保持 None；运行时自动解析为 export.model.hf_model，避免写死 base 路径。",
    },
}

_QUANT_CONFIG_HELP: dict[str, Any] = {
    "default": get_default_quant_config(),
    "fields": _QUANT_FIELD_HELP,
    "supported_algorithms": {
        "autoround": "Run AutoRound and save a GPTQModel-compatible HuggingFace artifact.",
        "existing_hf": "Reuse an already quantized HuggingFace/GPTQModel directory via existing_hf_model_dir.",
        "none": "Pass config_overrides={'quant': None} to validate/export the base HF model without quantization.",
    },
    "required_constraints": [
        "group_size must be 64",
        "artifact_format/output_format must be gptqmodel_hf",
        "autoround_format defaults to auto_gptq for dense YAML; MoE YAML uses auto_round:gptqmodel",
        "quant_nontext_module defaults to False for mode1 LLM-only quantization",
        "default YAML quant is not null; use config_overrides={'quant': None} only for explicit base validation",
    ],
    "template": quant_config_template(),
}

_EXPORT_CONFIG_HELP: dict[str, Any] = {
    "variants": ["full", "mtp", "dflash", "visual_only"],
    "visual_sizes": [448, 896],
    "fields": _EXPORT_FIELD_HELP,
    "model_types": {
        "full": ["Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"],
        "visual_only": ["Qwen3_5ForConditionalGeneration_visual", "Qwen3_5MoeForConditionalGeneration_visual"],
    },
    "notes": [
        "full exports LLM prefill/decode and includes visual_config when the YAML has it.",
        "mtp extends full with output_post_norm_hidden and mtp_config.",
        "dflash extends full with output_hidden_state_indices and dflash_config.",
        "visual_only exports the visual tower at max_size_w/max_size_h 448 or 896.",
        "fuse_gdr_ops is config/override-only; default is False until compiler support is ready.",
    ],
    "template": export_config_template(),
}

_MODEL_DOCS: dict[str, Any] = {
    "name": "Qwen3.5 / Qwen3.6 Merak HMONNX workflow",
    "api_import": "from xhmodel_merak.xh_llm.models.qwen3_5 import quant, export",
    "recommended_config_api": [
        "list_recommended_configs()",
        "get_default_quant_config()",
        "get_default_export_config(name=... or family/model_size/variant/visual_size)",
        "get_default_workflow_config(name=... or family/model_size/variant/visual_size)",
    ],
    "supported_models": [
        {
            "family": "qwen3_5",
            "model_size": "9b",
            "hf_model": "weights/Qwen3.5-9B",
            "verified_quant_model": "weights/Qwen3.5-9B-mode1-llm-only",
            "variants": ["full", "mtp", "dflash", "visual_only_448", "visual_only_896"],
        },
        {
            "family": "qwen3_5",
            "model_size": "27b",
            "hf_model": "weights/Qwen3.6-27B",
            "verified_quant_model": None,
            "variants": ["full", "mtp", "dflash", "visual_only_448", "visual_only_896"],
        },
        {
            "family": "qwen3_5_moe",
            "model_size": "35b_a3b",
            "hf_model": "weights/Qwen3.6-35B-A3B",
            "verified_quant_model": "weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400",
            "variants": ["full", "mtp", "dflash", "visual_only_448", "visual_only_896"],
        },
    ],
    "default_quant": get_default_quant_config(),
    "default_export_summary": {
        "chip_arch": "XH2a",
        "context_max_length": 2048,
        "prefill_chunk_length": 256,
        "quant_type": "w8a8h1_sefp",
        "visual_sizes": [448, 896],
        "fuse_gdr_ops": False,
    },
    "validation_scope": {
        "full": "9B and 35B-A3B are validated for base and existing_hf quant with fuse_gdr_ops False/True.",
        "spec_decode": "9B and 35B-A3B MTP/DFlash validation uses existing_hf quant artifacts only.",
    },
    "hmonnx_io_doc": str(_HMONNX_IO_DOC),
    "readme": "examples_merak/llm/qwen3_5/README.md",
}


def get_quant_config_help() -> dict[str, Any]:
    """Return structured help for supported quant config fields and defaults."""
    return copy.deepcopy(_QUANT_CONFIG_HELP)


def get_export_config_help() -> dict[str, Any]:
    """Return structured help for supported export variants and visual sizes."""
    return copy.deepcopy(_EXPORT_CONFIG_HELP)


def get_model_docs() -> dict[str, Any]:
    """Return structured model-family documentation and links to the full Markdown guide."""
    docs = copy.deepcopy(_MODEL_DOCS)
    docs["hmonnx_io_doc_exists"] = (repo_root() / _HMONNX_IO_DOC).is_file()
    return docs


__all__ = ["get_export_config_help", "get_model_docs", "get_quant_config_help"]
