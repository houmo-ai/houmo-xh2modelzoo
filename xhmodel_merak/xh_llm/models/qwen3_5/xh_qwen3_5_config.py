import json
from collections.abc import Mapping
from pathlib import Path

from xhmodel_merak.configuration_utils import HFModelConfig
from xhquant.api import QuantScheme

from ...vision_llm_model import VisionLLMModelConfig
from .lora import XHQwen3_5LoRAConfig, coerce_lora_config


DRAFT_BASE_QUANT_TYPE = "w8a8h1_sefp"


def build_spec_draft_quant_scheme(
    head_weight_bits: int = 4,
    *,
    head_node_name: str = "lm_head",
) -> dict:
    """Build the default MTP/DFlash draft quant scheme.

    Draft graphs use the regular W8A8 base quantization, but the large logits
    head is configurable.  Keep the default at W4 for spec-decode draft heads
    unless callers provide an explicit per-draft ``quant_scheme``.
    """
    if head_weight_bits == 8:
        return dict(quant_type=DRAFT_BASE_QUANT_TYPE)
    if head_weight_bits != 4:
        raise ValueError(f"Unsupported spec draft head weight bits: {head_weight_bits}. Expected 4 or 8.")
    return dict(
        quant_type=DRAFT_BASE_QUANT_TYPE,
        nodes_cfg={
            head_node_name: dict(
                w_schema=dict(
                    bits=4,
                    fp_mode="ssfp",
                    hidden_bit=False,
                )
            )
        },
    )


class XHQwen3_5_VisualConfig(HFModelConfig):  # noqa: N801
    def __init__(
        self,
        *,
        max_size_w: int,
        max_size_h: int,
        max_size_t: int = 2,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        lora: Mapping[str, object] | None = None,
        **kwargs,
    ):
        if lora is not None:
            raise ValueError("Qwen3.5 ViT/visual export does not support LoRA")
        super().__init__(**kwargs)
        self.max_size_w = max_size_w
        self.max_size_h = max_size_h
        self.max_size_t = max_size_t
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size


class XHQwen3_5_MTPConfig(HFModelConfig):  # noqa: N801
    def __init__(
        self,
        *,
        hidden_size: int = 3584,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        batch_size: int = 1,
        input_sequence_length: int = 1,
        context_max_length: int = 2048,
        max_pe_length: int = 262144,
        use_cache: bool = True,
        flash_attention: Mapping | None = None,
        draft_head_weight_bits: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.input_sequence_length = input_sequence_length
        self.context_max_length = context_max_length
        self.max_pe_length = max_pe_length
        self.use_cache = use_cache
        self.flash_attention = flash_attention
        self.draft_head_weight_bits = draft_head_weight_bits
        if getattr(self, "quant_scheme", None) is None:
            self.quant_scheme = build_spec_draft_quant_scheme(draft_head_weight_bits)

    @property
    def hf_model_dir(self) -> str:
        return self.hf_model


class XHQwen3_5_DFlashConfig(HFModelConfig):  # noqa: N801
    def __init__(
        self,
        *,
        mode: str = "decode",
        target_model_dir: str | None = None,
        hidden_size: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        num_hidden_layers: int | None = None,
        num_target_layers: int | None = None,
        block_size: int | None = None,
        batch_size: int = 1,
        input_sequence_length: int = 1,
        max_pe_length: int = 262144,
        max_sequence_length: int = 256,
        flash_attention: Mapping | None = None,
        draft_head_weight_bits: int = 4,
        noise_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assistant_config_path = Path(str(self.hf_model)) / "config.json"
        assistant_config: dict = {}
        if assistant_config_path.is_file():
            assistant_config = json.loads(
                assistant_config_path.read_text(encoding="utf-8")
            )
        dflash_hf_config = assistant_config.get("dflash_config") or {}

        def resolve_checkpoint_int(
            name: str,
            configured: int | None,
            *,
            legacy_default: int,
            checkpoint_value: int | None = None,
        ) -> int:
            value = (
                assistant_config.get(name)
                if checkpoint_value is None
                else checkpoint_value
            )
            if value is None:
                return (
                    legacy_default
                    if configured is None
                    else int(configured)
                )
            expected = int(value)
            if configured is not None and int(configured) != expected:
                raise ValueError(
                    f"Qwen3.5 DFlash {name} must match assistant checkpoint "
                    f"{assistant_config_path}: configured={configured}, "
                    f"checkpoint={expected}"
                )
            return expected

        target_layer_ids = dflash_hf_config.get("target_layer_ids") or []
        hidden_size = resolve_checkpoint_int(
            "hidden_size",
            hidden_size,
            legacy_default=3584,
        )
        num_attention_heads = resolve_checkpoint_int(
            "num_attention_heads",
            num_attention_heads,
            legacy_default=28,
        )
        num_key_value_heads = resolve_checkpoint_int(
            "num_key_value_heads",
            num_key_value_heads,
            legacy_default=4,
        )
        head_dim = resolve_checkpoint_int(
            "head_dim",
            head_dim,
            legacy_default=128,
        )
        num_hidden_layers = resolve_checkpoint_int(
            "num_hidden_layers",
            num_hidden_layers,
            legacy_default=1,
        )
        num_target_layers = resolve_checkpoint_int(
            "num_target_layers",
            num_target_layers,
            legacy_default=4,
            checkpoint_value=(
                len(target_layer_ids)
                if target_layer_ids
                else None
            ),
        )
        checkpoint_block_size = assistant_config.get("block_size")
        if checkpoint_block_size is None:
            checkpoint_block_size = dflash_hf_config.get("block_size")
        block_size = resolve_checkpoint_int(
            "block_size",
            block_size,
            legacy_default=16,
            checkpoint_value=checkpoint_block_size,
        )

        self.mode = mode
        self.target_model_dir = target_model_dir
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_target_layers = num_target_layers
        self.block_size = block_size
        self.batch_size = batch_size
        self.input_sequence_length = input_sequence_length
        self.max_pe_length = max_pe_length
        self.max_sequence_length = max_sequence_length
        self.flash_attention = flash_attention
        self.draft_head_weight_bits = draft_head_weight_bits
        if noise_token_id is None:
            if not assistant_config_path.is_file():
                raise ValueError(
                    "Qwen3.5 DFlash requires noise_token_id or an assistant "
                    f"config.json containing dflash_config.mask_token_id: "
                    f"{assistant_config_path}"
                )
            noise_token_id = dflash_hf_config.get("mask_token_id")
        if noise_token_id is None:
            raise ValueError("Qwen3.5 DFlash assistant config has no dflash_config.mask_token_id")
        self.noise_token_id = int(noise_token_id)
        if self.noise_token_id < 0:
            raise ValueError("Qwen3.5 DFlash noise_token_id must be non-negative")
        if getattr(self, "quant_scheme", None) is None:
            # The TorchFX adapter owns the assistant under ``core`` and names
            # its head node ``core_lm_head``. Quant overrides match FX node
            # names (not the dotted module target ``core.lm_head``).
            self.quant_scheme = build_spec_draft_quant_scheme(
                draft_head_weight_bits,
                head_node_name="core_lm_head",
            )

    @property
    def dflash_model_dir(self) -> str:
        return self.hf_model


class XHQwen3_5ModelConfig(VisionLLMModelConfig):  # noqa: N801
    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str | None = None,
        quant_scheme: dict | QuantScheme | None = None,
        quant_weight: str | None = None,
        hf_model: str | None = None,
        batch_size: int = 1,
        context_max_length: int = 2048,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: int | None = 1,
        mix_search: bool = False,
        use_cache: bool = True,
        linear_chunk_size: int = 64,
        flash_attention: Mapping | None = None,
        fuse_gdr_ops: bool = False,
        fuse_gdr_block_recurrent_ops: bool = False,
        split_conv_cache: bool = True,
        normalize_force_fp32: bool = False,
        use_manual_depthwise_conv1d: bool = False,
        cumsum_matmul_quant_config: dict | None = None,
        visual_config: dict | XHQwen3_5_VisualConfig | None = None,
        spec_decode_mode: str | None = None,
        mtp_config: dict | XHQwen3_5_MTPConfig | None = None,
        dflash_config: dict | XHQwen3_5_DFlashConfig | None = None,
        num_draft_tokens: int | None = None,
        spec_draft_head_weight_bits: int = 4,
        mtp_head_k: int | None = None,
        reranked_repo_dir: str | None = None,
        force_rerank: bool = False,
        lora: dict | XHQwen3_5LoRAConfig | None = None,
        **kwargs,
    ):
        if isinstance(visual_config, Mapping) and visual_config.get("lora") is not None:
            raise ValueError(
                "Qwen3.5 visual_config does not support LoRA; configure language-model adapters "
                "under export.model.lora only"
            )
        super().__init__(
            model_name=model_name,
            chip_arch=chip_arch,
            model_type=model_type,
            hf_model=hf_model,
            quant_scheme=quant_scheme,
            quant_weight=quant_weight,
            batch_size=batch_size,
            context_max_length=context_max_length,
            prefill_chunk_length=prefill_chunk_length,
            num_logits_to_keep=num_logits_to_keep,
            mix_search=mix_search,
            use_cache=use_cache,
            **kwargs,
        )
        if isinstance(visual_config, Mapping):
            visual_config = dict(visual_config)
            if "model_name" not in visual_config:
                visual_config["model_name"] = f"{model_name}_visual"
            if "hf_model" not in visual_config:
                visual_config["hf_model"] = hf_model
            visual_config = XHQwen3_5_VisualConfig(**visual_config)
        self.visual_config = visual_config

        self.linear_chunk_size = linear_chunk_size
        self.flash_attention = flash_attention
        self.fuse_gdr_ops = fuse_gdr_ops
        self.fuse_gdr_block_recurrent_ops = fuse_gdr_block_recurrent_ops
        self.split_conv_cache = split_conv_cache
        self.normalize_force_fp32 = normalize_force_fp32
        self.use_manual_depthwise_conv1d = use_manual_depthwise_conv1d
        self.cumsum_matmul_quant_config = cumsum_matmul_quant_config

        if spec_decode_mode in {"mtp", "dflash"} and int(batch_size) != 1:
            raise ValueError(
                "Qwen3.5 multi-batch export does not support "
                f"spec_decode_mode={spec_decode_mode!r}; use batch_size=1 for MTP/DFlash "
                "or disable spec_decode_mode for batch_size>1."
            )

        self.spec_decode_mode = spec_decode_mode
        if num_draft_tokens is None:
            # MTP historically drafts four tokens. DFlash checkpoints are
            # trained/exported with nine draft tokens unless a workflow run
            # explicitly overrides the target verify width.
            num_draft_tokens = 9 if spec_decode_mode == "dflash" else 4
        if int(num_draft_tokens) <= 0:
            raise ValueError("Qwen3.5 num_draft_tokens must be positive")
        self.num_draft_tokens = int(num_draft_tokens)
        self.spec_draft_head_weight_bits = spec_draft_head_weight_bits
        self.mtp_head_k = mtp_head_k
        self.reranked_repo_dir = reranked_repo_dir
        self.force_rerank = force_rerank
        self.lora = coerce_lora_config(lora)

        if isinstance(mtp_config, Mapping):
            mtp_config = dict(mtp_config)
            if "model_name" not in mtp_config:
                mtp_config["model_name"] = f"{model_name}_mtp"
            # Qwen3.5 MTP tensors live in the target checkpoint. The draft
            # graph therefore cannot independently select a checkpoint or
            # cache capacity.
            mtp_config["hf_model"] = hf_model
            mtp_config["context_max_length"] = context_max_length
            if "draft_head_weight_bits" not in mtp_config:
                mtp_config["draft_head_weight_bits"] = spec_draft_head_weight_bits
            mtp_config["flash_attention"] = flash_attention
            mtp_config = XHQwen3_5_MTPConfig(**mtp_config)
        elif mtp_config is not None:
            mtp_config.hf_model = hf_model
            mtp_config.context_max_length = context_max_length
            mtp_config.flash_attention = flash_attention
        self.mtp_config = mtp_config

        if isinstance(dflash_config, Mapping):
            dflash_config = dict(dflash_config)
            if "model_name" not in dflash_config:
                dflash_config["model_name"] = f"{model_name}_dflash"
            if "hf_model" not in dflash_config:
                dflash_config["hf_model"] = hf_model
            # The assistant checkpoint is independent, but its shared target
            # cache is not: one top-level context value owns both capacities.
            dflash_config["target_model_dir"] = hf_model
            dflash_config["max_sequence_length"] = context_max_length
            dflash_config["max_pe_length"] = context_max_length
            dflash_config["batch_size"] = batch_size
            dflash_config["input_sequence_length"] = prefill_chunk_length
            if "draft_head_weight_bits" not in dflash_config:
                dflash_config["draft_head_weight_bits"] = spec_draft_head_weight_bits
            # The unified workflow flag applies to the target and the DFlash
            # decoder. A second hidden switch can silently produce mixed graph
            # contracts.
            dflash_config["flash_attention"] = flash_attention
            dflash_config = XHQwen3_5_DFlashConfig(**dflash_config)
        elif dflash_config is not None:
            dflash_config.target_model_dir = hf_model
            dflash_config.max_sequence_length = context_max_length
            dflash_config.max_pe_length = context_max_length
            dflash_config.batch_size = batch_size
            dflash_config.input_sequence_length = prefill_chunk_length
            dflash_config.flash_attention = flash_attention
        self.dflash_config = dflash_config
        if self.dflash_config is not None:
            dflash_block_size = int(
                getattr(
                    self.dflash_config,
                    "block_size",
                    self.num_draft_tokens + 1,
                )
            )
            if self.num_draft_tokens + 1 > dflash_block_size:
                raise ValueError(
                    "Qwen3.5 DFlash num_draft_tokens exceeds checkpoint "
                    f"capacity: proposals={self.num_draft_tokens}, "
                    f"block_size={dflash_block_size}"
                )
        if self.dflash_config is not None and getattr(self.dflash_config, "target_model_dir", None) is None:
            self.dflash_config.target_model_dir = hf_model

        self.image_token_id: int | None = None
        self.video_token_id: int | None = None
        self.vision_start_token_id: int | None = None

        self.vision_end_token_id: int | None = None
        self.spatial_merge_size: int | None = None
