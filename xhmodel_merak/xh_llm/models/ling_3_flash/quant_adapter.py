"""GPTQModel and AutoRound weight-only quantization for Ling-3-Flash."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._compat import load_ling_config, patch_ling_remote_code_compatibility


_QUANTIZATION_ATTENTION_BACKEND = "eager"


@dataclass
class Ling3FlashQuantSpec:
    model_dir: str
    output_dir: str
    method: str
    dataset: str = "NeelNanda/pile-10k"
    calibration_jsonl: str | None = None
    text_key: str = "text"
    nsamples: int = 128
    seqlen: int = 2048
    batch_size: int = 1
    moe_batch_size: int | None = None
    iters: int = 200
    group_size: int = 64
    base_bits: int = 8
    expert_bits: int = 4
    device: str = "cuda:0"
    device_map: str = "0"
    offload_to_disk: bool = False
    offload_dir: str = "work_dirs/ling_3_flash_offload"
    wait_for_submodule_finalizers: bool = True
    hessian_mse: bool = True
    max_quant_layers: int | None = None
    seed: int = 42
    dry_run: bool = False

    def validate(self) -> None:
        self.method = str(self.method).lower().replace("-", "_")
        if self.method == "auto_round":
            self.method = "autoround"
        if self.method not in {"gptq", "autoround"}:
            raise ValueError(f"Unsupported Ling quantization method: {self.method!r}")
        if self.group_size != 64:
            raise ValueError("Ling-3-Flash deployment profile requires group/block size 64")
        if self.base_bits != 8 or self.expert_bits != 4:
            raise ValueError(
                "Ling-3-Flash profile is fixed to routed experts W4 and all other quantized linears W8"
            )
        if (
            self.nsamples <= 0
            or self.seqlen <= 0
            or self.batch_size <= 0
        ):
            raise ValueError("nsamples, seqlen and batch_size must be positive")
        if self.moe_batch_size is not None and self.moe_batch_size <= 0:
            raise ValueError("moe_batch_size must be positive when specified")
        if self.iters < 0:
            raise ValueError("iters must be non-negative")


def _is_kda_decay_or_beta_projection(name: str) -> bool:
    return name.endswith(
        (
            ".attention.f_proj",
            ".attention.f_a_proj",
            ".attention.f_b_proj",
            ".attention.b_proj",
        )
    )


def _ling_layer_layout(
    model: torch.nn.Module,
) -> tuple[list[str], list[str]]:
    """Return decoder and MTP block names without relying on a fixed prefix."""

    config = getattr(model, "config", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    language_model = getattr(model, "model", None)
    layers = getattr(language_model, "layers", None)
    if num_hidden_layers is None or layers is None:
        raise RuntimeError("Ling model is missing config.num_hidden_layers or model.layers")

    num_hidden_layers = int(num_hidden_layers)
    if num_hidden_layers <= 0 or len(layers) < num_hidden_layers:
        raise RuntimeError(
            "Ling layer layout is invalid: "
            f"num_hidden_layers={num_hidden_layers}, total_layers={len(layers)}"
        )

    layer_prefixes = [
        name for name, module in model.named_modules() if module is layers
    ]
    if len(layer_prefixes) != 1 or not layer_prefixes[0]:
        raise RuntimeError(
            f"Unable to resolve the Ling model.layers prefix: {layer_prefixes}"
        )
    layer_prefix = layer_prefixes[0]
    decoder_blocks = [
        f"{layer_prefix}.{index}" for index in range(num_hidden_layers)
    ]
    mtp_blocks = [
        f"{layer_prefix}.{index}"
        for index in range(num_hidden_layers, len(layers))
    ]
    return decoder_blocks, mtp_blocks


def build_autoround_block_names(model: torch.nn.Module) -> list[list[str]]:
    """Select only normal decoder blocks; MTP has a two-input forward contract."""

    decoder_blocks, _ = _ling_layer_layout(model)
    return [decoder_blocks]


def _build_autoround_regex_policy(
    decoder_blocks: list[str],
    mtp_blocks: list[str],
    *,
    expert_bits: int,
    group_size: int,
) -> dict[str, dict[str, int | str]]:
    """Build non-overlapping regex rules in AutoRound's input syntax."""

    if not decoder_blocks:
        raise RuntimeError("Ling AutoRound policy requires decoder blocks")
    all_blocks = [*decoder_blocks, *mtp_blocks]
    if any(re.search(r"[^A-Za-z0-9_.]", name) for name in all_blocks):
        raise RuntimeError(f"Ling block names cannot be represented safely: {all_blocks}")

    decoder_group = "(?:" + "|".join(decoder_blocks) + ")"
    policy: dict[str, dict[str, int | str]] = {}
    if mtp_blocks:
        mtp_group = "(?:" + "|".join(mtp_blocks) + ")"
        policy[f"^{mtp_group}(?:[.].*)$"] = {
            "bits": 16,
            "data_type": "fp",
        }
    policy[
        f"^{decoder_group}.attention."
        r"(?:f_proj|f_a_proj|f_b_proj|b_proj)$"
    ] = {"bits": 16, "data_type": "fp"}
    policy[
        f"^{decoder_group}.mlp.experts."
        r"\d+.(?:gate_proj|up_proj|down_proj)$"
    ] = {"bits": expert_bits, "group_size": group_size}
    return policy


def build_autoround_layer_config(
    model: torch.nn.Module,
    *,
    expert_bits: int = 4,
    group_size: int = 64,
) -> dict[str, dict[str, int | str]]:
    """Return three regex policies instead of enumerating every expert Linear.

    Ling's f/b projections correspond to Qwen3.5's excluded
    ``in_proj_a/in_proj_b`` decay/beta path. Every other Linear inherits the
    AutoRound base W8/G64/symmetric scheme. Supplying regexes here is
    important: AutoRound preserves them in ``regex_config`` and can collapse
    the expanded per-module state when exporting GPTQModel metadata.
    """

    decoder_blocks, mtp_blocks = _ling_layer_layout(model)
    decoder_prefixes = tuple(f"{name}." for name in decoder_blocks)
    mtp_prefixes = tuple(f"{name}." for name in mtp_blocks)
    routed_experts = 0
    excluded_kda = 0
    excluded_mtp = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.startswith(mtp_prefixes):
            excluded_mtp += 1
        elif name.startswith(decoder_prefixes) and _is_kda_decay_or_beta_projection(name):
            excluded_kda += 1
        elif name.startswith(decoder_prefixes) and ".mlp.experts." in name and name.endswith(
            (".gate_proj", ".up_proj", ".down_proj")
        ):
            routed_experts += 1
    if routed_experts == 0:
        raise RuntimeError("AutoRound found no routed expert Linear modules")
    if excluded_kda == 0:
        raise RuntimeError("AutoRound found no Ling KDA decay/beta projections to exclude")
    if mtp_blocks and excluded_mtp == 0:
        raise RuntimeError("AutoRound found Ling MTP blocks but no MTP Linear modules")
    return _build_autoround_regex_policy(
        decoder_blocks,
        mtp_blocks,
        expert_bits=expert_bits,
        group_size=group_size,
    )


def _autoround_device_map(value: str):
    value = str(value).strip()
    if value.isdigit():
        return int(value)
    if "," in value and all(part.strip().isdigit() for part in value.split(",")):
        raise ValueError(
            "Pass one local AutoRound GPU index per experiment, for example device_map='0'"
        )
    return value


def _install_autoround_nonfinite_loss_guard(autoround) -> None:
    """Fail immediately instead of exporting after an invalid tuning loss."""

    original_get_loss = autoround._get_loss

    def _guarded_get_loss(self, *args, **kwargs):
        loss = original_get_loss(*args, **kwargs)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(
                "Ling-3-Flash AutoRound produced a non-finite tuning loss. "
                "The quantized checkpoint was not exported."
            )
        return loss

    autoround._get_loss = MethodType(_guarded_get_loss, autoround)


def canonicalize_autoround_gptqmodel_config(output_dir: str | Path) -> None:
    """Validate native regex export and remove redundant expanded metadata.

    ``auto_round:gptqmodel`` consumers use ``dynamic``. AutoRound also writes
    its expanded ``extra_config`` for compatibility with its native loader;
    that data is redundant because the same loader can expand ``dynamic``.
    Removing it keeps Ling's configuration proportional to the policy rather
    than to its 512-expert module count. AutoRound writes the metadata both
    inside ``config.json`` and, on some versions, into a standalone
    ``quantization_config.json``. Canonicalize both so loaders cannot
    accidentally select the still-expanded copy.
    """

    config_path = Path(output_dir).expanduser().resolve() / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise RuntimeError(f"AutoRound output has no quantization_config: {config_path}")
    dynamic = quantization_config.get("dynamic")
    if not isinstance(dynamic, dict) or not dynamic:
        raise RuntimeError(f"AutoRound output has no dynamic rules: {config_path}")

    if (
        int(quantization_config.get("bits", 0)) != 8
        or int(quantization_config.get("group_size", 0)) != 64
        or quantization_config.get("sym") is not True
    ):
        raise RuntimeError("Ling AutoRound output is not the required W8/G64/sym profile")

    num_hidden_layers = int(config.get("num_hidden_layers", 0))
    num_mtp_layers = int(config.get("num_nextn_predict_layers", 0))
    if num_hidden_layers <= 0 or num_mtp_layers <= 0:
        raise RuntimeError(
            "Ling AutoRound output must declare positive num_hidden_layers and "
            "num_nextn_predict_layers"
        )
    block_prefixes = quantization_config.get("block_name_to_quantize")
    if not (
        isinstance(block_prefixes, list)
        and len(block_prefixes) == 1
        and isinstance(block_prefixes[0], str)
    ):
        raise RuntimeError(
            "Ling AutoRound output must contain one block_name_to_quantize prefix"
        )
    layer_prefix = block_prefixes[0]
    decoder_blocks = [
        f"{layer_prefix}.{index}" for index in range(num_hidden_layers)
    ]
    mtp_blocks = [
        f"{layer_prefix}.{index}"
        for index in range(num_hidden_layers, num_hidden_layers + num_mtp_layers)
    ]
    policy = _build_autoround_regex_policy(
        decoder_blocks,
        mtp_blocks,
        expert_bits=4,
        group_size=64,
    )

    from auto_round.utils import to_standard_regex

    expected_dynamic: dict[str, dict[str, int]] = {}
    expected_regex_values: dict[str, dict[str, int | str]] = {}
    for pattern, layer_cfg in policy.items():
        standardized = to_standard_regex(pattern)
        expected_regex_values[standardized] = layer_cfg
        if int(layer_cfg["bits"]) >= 16:
            expected_dynamic[f"-:{standardized}"] = {}
        else:
            expected_dynamic[f"+:{standardized}"] = {"bits": int(layer_cfg["bits"])}

    def resolve(rules: dict, module_name: str):
        for pattern, overrides in rules.items():
            if pattern.startswith("-:"):
                if re.match(pattern.removeprefix("-:"), module_name):
                    return False
            elif re.match(pattern.removeprefix("+:"), module_name):
                return overrides
        return None

    def validate_and_canonicalize(candidate: dict, *, source: Path) -> None:
        candidate_dynamic = candidate.get("dynamic")
        if not isinstance(candidate_dynamic, dict) or not candidate_dynamic:
            raise RuntimeError(f"Ling AutoRound config has no dynamic rules: {source}")
        for field in ("bits", "group_size", "sym", "block_name_to_quantize"):
            if candidate.get(field) != quantization_config.get(field):
                raise RuntimeError(
                    f"Ling AutoRound metadata mismatch for {field!r} in {source}: "
                    f"expected {quantization_config.get(field)!r}, got {candidate.get(field)!r}"
                )

        extra_config = candidate.get("extra_config", {})
        if extra_config is not None and not isinstance(extra_config, dict):
            raise RuntimeError(f"Ling AutoRound extra_config must be a mapping: {source}")

        exact_entries = 0
        for name, layer_cfg in (extra_config or {}).items():
            if name in expected_regex_values:
                expected = (
                    False
                    if int(expected_regex_values[name]["bits"]) >= 16
                    else {"bits": int(expected_regex_values[name]["bits"])}
                )
            else:
                expected = resolve(expected_dynamic, name)
                exact_entries += 1
            if expected is False:
                valid = int(layer_cfg.get("bits", 16)) >= 16
            elif isinstance(expected, dict):
                valid = int(layer_cfg.get("bits", 8)) == int(expected["bits"])
            else:
                valid = False
            if not valid:
                raise RuntimeError(
                    f"Unexpected Ling AutoRound extra_config entry in {source}: "
                    f"{name!r}: {layer_cfg!r}"
                )

        if extra_config and exact_entries == 0:
            raise RuntimeError(
                f"Ling AutoRound extra_config did not contain expanded module state: {source}"
            )

        if candidate_dynamic != expected_dynamic:
            if len(candidate_dynamic) > 100:
                exact_rule = re.compile(
                    r"^(?P<kind>[+-]):\^(?P<name>[A-Za-z0-9_]+(?:\\\.[A-Za-z0-9_]+)*)\$$"
                )
                for pattern, overrides in candidate_dynamic.items():
                    match = exact_rule.fullmatch(pattern)
                    if match is None:
                        raise RuntimeError(
                            f"Unexpected expanded Ling AutoRound dynamic rule in {source}: "
                            f"{pattern!r}"
                        )
                    name = match.group("name").replace(r"\.", ".")
                    expected = resolve(expected_dynamic, name)
                    actual = False if match.group("kind") == "-" else overrides
                    if actual != expected:
                        raise RuntimeError(
                            f"Ling AutoRound dynamic mismatch for {name!r} in {source}: "
                            f"expected {expected!r}, got {actual!r}"
                        )
            else:
                for name in (extra_config or {}):
                    if name in expected_regex_values:
                        continue
                    if resolve(candidate_dynamic, name) != resolve(expected_dynamic, name):
                        raise RuntimeError(
                            f"Ling AutoRound dynamic mismatch for expanded module "
                            f"{name!r} in {source}"
                        )

        candidate["dynamic"] = expected_dynamic
        candidate.pop("extra_config", None)

    validate_and_canonicalize(quantization_config, source=config_path)

    documents_to_write = [(config_path, config)]
    standalone_path = config_path.with_name("quantization_config.json")
    if standalone_path.is_file():
        standalone_document = json.loads(standalone_path.read_text(encoding="utf-8"))
        standalone_quantization_config = standalone_document.get(
            "quantization_config", standalone_document
        )
        if not isinstance(standalone_quantization_config, dict):
            raise RuntimeError(
                f"AutoRound standalone quantization metadata is not a mapping: {standalone_path}"
            )
        validate_and_canonicalize(
            standalone_quantization_config,
            source=standalone_path,
        )
        documents_to_write.append((standalone_path, standalone_document))

    temporary_documents = []
    for path, document in documents_to_write:
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_documents.append((temporary_path, path))
    for temporary_path, path in temporary_documents:
        temporary_path.replace(path)


def _quantize_gptq(
    spec: Ling3FlashQuantSpec,
    calibration_texts: Iterable[str] | None,
) -> None:
    from gptqmodel.recipes.ling3_flash import (
        quantize_ling3_flash as quantize_ling3_flash_recipe,
    )

    quantize_ling3_flash_recipe(
        model_dir=spec.model_dir,
        output_dir=spec.output_dir,
        method="gptq",
        artifact_format="gptqmodel_hf",
        bits=spec.base_bits,
        expert_bits=spec.expert_bits,
        group_size=spec.group_size,
        sym=True,
        batch_size=spec.batch_size,
        nsamples=spec.nsamples,
        seqlen=spec.seqlen,
        calibration_jsonl=spec.calibration_jsonl,
        calibration_text_key=spec.text_key,
        calibration_dataset=spec.dataset,
        calibration_data=calibration_texts,
        device=spec.device,
        trust_remote_code=True,
        offload_to_disk=spec.offload_to_disk,
        offload_path=spec.offload_dir if spec.offload_to_disk else None,
        hessian_mse=spec.hessian_mse,
        wait_for_submodule_finalizers=spec.wait_for_submodule_finalizers,
        moe_routing_batch_size=spec.moe_batch_size,
        auto_forward_data_parallel=False,
        max_quant_layers=spec.max_quant_layers,
        seed=spec.seed,
    )


def autoround_calibration_source(
    spec: Ling3FlashQuantSpec,
    calibration_texts: Iterable[str] | None,
) -> str | list[str]:
    """Let AutoRound build exactly ``nsamples`` full-length token blocks.

    Passing only the first ``nsamples`` raw documents is not equivalent: the
    AutoRound dataloader drops every document shorter than ``seqlen``.  For the
    Ling calibration JSONL that left only 17 valid rows out of 128.  Passing
    the dataset path/name lets AutoRound scan, shuffle, filter, and select the
    requested number of full-length rows itself.
    """

    if calibration_texts is not None:
        texts = list(calibration_texts)
        if len(texts) < spec.nsamples:
            raise ValueError(
                f"calibration_texts contains {len(texts)} samples, need {spec.nsamples}"
            )
        return texts[: spec.nsamples]
    if spec.calibration_jsonl is not None:
        return str(Path(spec.calibration_jsonl).expanduser().resolve())
    return spec.dataset


def _quantize_autoround(
    spec: Ling3FlashQuantSpec,
    calibration_source: str | list[str],
) -> None:
    from auto_round import AutoRound

    patch_ling_remote_code_compatibility()
    config = load_ling_config(spec.model_dir)
    config._attn_implementation = _QUANTIZATION_ATTENTION_BACKEND
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_dir,
        config=config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation=_QUANTIZATION_ATTENTION_BACKEND,
    )
    patch_ling_remote_code_compatibility()
    layer_config = build_autoround_layer_config(
        model,
        expert_bits=spec.expert_bits,
        group_size=spec.group_size,
    )
    block_names = build_autoround_block_names(model)
    autoround = AutoRound(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained(spec.model_dir, trust_remote_code=True),
        bits=spec.base_bits,
        group_size=spec.group_size,
        sym=True,
        data_type="int_sym_gptq",
        dataset=calibration_source,
        nsamples=spec.nsamples,
        seqlen=spec.seqlen,
        iters=spec.iters,
        batch_size=spec.batch_size,
        gradient_accumulate_steps=1,
        low_gpu_mem_usage=True,
        low_cpu_mem_usage=True,
        device_map=_autoround_device_map(spec.device_map),
        seed=spec.seed,
        layer_config=layer_config,
        to_quant_block_names=block_names,
        enable_quanted_input=True,
    )
    _install_autoround_nonfinite_loss_guard(autoround)
    autoround.quantize()
    patch_ling_remote_code_compatibility()
    autoround.save_quantized(spec.output_dir, format="auto_round:gptqmodel")
    canonicalize_autoround_gptqmodel_config(spec.output_dir)


def quantization_plan(spec: Ling3FlashQuantSpec) -> dict:
    return {
        "method": spec.method,
        "model": spec.model_dir,
        "output": spec.output_dir,
        "precision": {
            "routed_experts": f"W{spec.expert_bits}",
            "all_other_quantized_linears": f"W{spec.base_bits}",
            "kda_decay_and_beta": "BF16 (Qwen3.5 proj_a/proj_b policy)",
            "mtp_layers": "BF16 (excluded from decoder block quantization)",
            "group_size": spec.group_size,
            "symmetric": True,
        },
        "calibration": {"nsamples": spec.nsamples, "seqlen": spec.seqlen},
        "moe_batch_size": spec.moe_batch_size,
        "offload_to_disk": spec.offload_to_disk,
        "offload_dir": spec.offload_dir if spec.offload_to_disk else None,
        "wait_for_submodule_finalizers": spec.wait_for_submodule_finalizers,
        "attention_backend": _QUANTIZATION_ATTENTION_BACKEND,
        "device": spec.device if spec.method == "gptq" else spec.device_map,
    }


def quantize_ling3_flash(
    spec: Ling3FlashQuantSpec,
    *,
    calibration_texts: Iterable[str] | None = None,
) -> Path:
    spec.validate()
    random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if spec.dry_run:
        return Path(spec.output_dir).expanduser().resolve()
    output_dir = Path(spec.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if spec.method == "gptq":
        _quantize_gptq(spec, calibration_texts)
    else:
        _quantize_autoround(
            spec,
            autoround_calibration_source(spec, calibration_texts),
        )
    return output_dir


__all__ = [
    "Ling3FlashQuantSpec",
    "autoround_calibration_source",
    "build_autoround_block_names",
    "build_autoround_layer_config",
    "canonicalize_autoround_gptqmodel_config",
    "quantization_plan",
    "quantize_ling3_flash",
]
