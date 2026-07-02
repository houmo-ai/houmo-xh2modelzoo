import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import WorkflowConfig
from .result import ExportResult, QuantResult
from .utils import same_abs_path


class BaseLLMWorkflow:
    expected_model_config_cls_name: str | None = None
    expected_model_cls_name: str | None = None

    def __init__(
        self,
        model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ):
        if not model_dir:
            raise ValueError("model_dir must be provided!")
        self.workflow_config = WorkflowConfig.from_file(config_path)
        self.model_dir = os.path.abspath(os.path.normpath(str(model_dir)))
        self.seed = seed
        self.debug = debug

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError(f"{type(self).__name__}.quant() must implement model-specific quantization")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        """
        绝大多数情况可复用基类export()方法导出HMONNX
        特殊情况例如mineru2_5需要导出多个vit，则由子类自己实现
        """
        import torch

        from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
        from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
        from xhquant.utils import MemoryTracker, TimeProfiler

        if quant_result is None:
            raise ValueError("quant_result can't be None")

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = self._build_export_config(quant_result, workflow_config)

        work_dir_path = Path(output_dir)
        if work_dir_path.exists():
            raise FileExistsError(f"{str(work_dir_path)} already exists!")
        work_dir_path.mkdir(parents=True)

        xhquant_init(str(work_dir_path / "export_hmonnx.log"), self.debug)
        set_random_seed(self.seed)
        logger = get_xhquant_logger()

        config_file = str(work_dir_path / f"{workflow_config.name}.yaml")
        workflow_config.dump(config_file)
        cfg = Config(export_cfg)

        logger.info(f"Workflow config: {workflow_config.name}")
        logger.info(f"Using device: {device}, cuda_available: {torch.cuda.is_available()}")
        logger.info(f"Config:\n{cfg.pretty_text}")

        model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
        logger.info(f"Resolved model config type: {type(model_cfg).__name__}")
        if (
            self.expected_model_config_cls_name is not None
            and type(model_cfg).__name__ != self.expected_model_config_cls_name
        ):
            raise TypeError(
                f"Expected model config type {self.expected_model_config_cls_name}, "
                f"but got {type(model_cfg).__name__}"
            )
        logger.info(f"Model Config:\n{model_cfg.to_json_string()}")

        xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
        logger.info(f"Resolved model type: {type(xh_model).__name__}")
        if self.expected_model_cls_name is not None and type(xh_model).__name__ != self.expected_model_cls_name:
            raise TypeError(f"Expected model type {self.expected_model_cls_name}, but got {type(xh_model).__name__}")
        # TODO work_dir是否必要？
        if hasattr(xh_model, "work_dir"):
            xh_model.work_dir = str(work_dir_path)

        with TimeProfiler("convert", logger), MemoryTracker(device, "convert2hmonnx", logger):
            meta = xh_model.export_hmonnx(str(work_dir_path))
        return ExportResult(work_dir=str(work_dir_path), config_file=config_file, meta=meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
    ) -> str:
        raise NotImplementedError(f"{type(self).__name__}.dump_golden() must be implemented")

    def _resolve_export_model_dir(self, quant_result: QuantResult) -> str:
        if not same_abs_path(quant_result.raw_model_dir, self.model_dir):
            raise ValueError("QuantResult.raw_model_dir must be the same as self.model_dir")
        if quant_result.skipped:
            return self.model_dir
        if not quant_result.quanted_model_dir:
            raise ValueError("QuantResult.quanted_model_dir must be provided when quant is not skipped")
        return quant_result.quanted_model_dir

    @staticmethod
    def _find_golden_meta_file(export_result: ExportResult) -> str:
        work_dir = Path(export_result.work_dir)
        if not work_dir.is_dir():
            raise FileNotFoundError(f"Export work_dir does not exist or is not a directory: {export_result.work_dir}")

        meta_files = []
        for path in work_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("hmquant"):
                continue
            meta_file = path / "golden_meta_info.json"
            if meta_file.is_file():
                meta_files.append(meta_file)

        if not meta_files:
            raise FileNotFoundError(f"No golden_meta_info.json found under hmquant* directories in {work_dir}")
        if len(meta_files) > 1:
            meta_file_list = ", ".join(str(path) for path in meta_files)
            raise ValueError(f"Found multiple golden_meta_info.json files under {work_dir}: {meta_file_list}")
        return str(meta_files[0])

    def _build_export_config(
        self,
        quant_result: QuantResult,
        workflow_config: WorkflowConfig,
    ) -> dict[str, Any]:
        export_model_dir = self._resolve_export_model_dir(quant_result)
        formatted_model_name = self._format_model_name(workflow_config, export_model_dir)
        export_cfg = workflow_config.build_export_dict()
        export_cfg["model"]["hf_model"] = export_model_dir
        export_cfg["model"]["model_name"] = formatted_model_name
        return export_cfg

    def _format_model_name(
        self,
        workflow_config: WorkflowConfig,
        export_model_dir: str,
    ) -> str:
        """
        规则如下
        修改后model_name格式：{chip_arch}_{原始model_name}_{spec_decode_mode}_{quant_scheme}_{prefill_chunk_length}_{context_max_length}_{max_pe_length}，全部转小写
        供参考的config.yaml：configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_full_dflash.yaml
        如果有字段或其他必要信息缺失，直接终止format，返回原始model_name

        各个字段规则
        原始model_name: yaml中export.model.model_name字段，不存在直接raiseError
        spec_decode_mode: yaml中export.model.spec_decode_mode字段，如果字段不存在"_{spec_decode_mode}"直接从format_model_name中删除
        chip_arch: yaml中export.model.chip_arch字段，XH2a映射到xh2，其余情况不变
        quant_scheme: yaml中export.model.quant_scheme.quant_type字段，只取w{数字}a{数字}，例如w8a16，w8a8。如果quant.bits字段存在，则w后的数字改为quant.bits数值
        prefill_chunk_length: yaml中export.model.prefill_chunk_length字段
        context_max_length: yaml中export.model.context_max_length字段
        max_pe_length: yaml中export.model.max_pe_length字段。如果缺失，从export_model_dir路径下的config.json中递归搜索max_position_embeddings字段
        """
        model_cfg = workflow_config.export["model"]
        try:
            model_name_token = model_cfg["model_name"]
        except KeyError as exc:
            raise ValueError("BaseLLMWorkflow requires `export.model.model_name` in workflow config") from exc

        try:
            chip_arch = model_cfg["chip_arch"]
            quant_type = model_cfg["quant_scheme"]["quant_type"]
            prefill_chunk_length = int(model_cfg["prefill_chunk_length"])
            context_max_length = int(model_cfg["context_max_length"])
        except (KeyError, TypeError):
            return str(model_name_token)

        chip_token = "xh2" if chip_arch.lower() == "xh2a" else chip_arch

        spec_decode_mode = model_cfg.get("spec_decode_mode")
        spec_decode_token = f"_{spec_decode_mode}" if spec_decode_mode else ""

        match = re.search(r"w(\d+)a(\d+)", quant_type.lower())
        if match is None:
            raise ValueError(
                "export.model.quant_scheme.quant_type must contain a w{bits}a{bits} token, "
                f"got {quant_type!r}"
            )
        weight_bits = int(match.group(1))
        activation_bits = int(match.group(2))
        quant_cfg = workflow_config.quant
        if isinstance(quant_cfg, Mapping) and quant_cfg.get("bits") is not None:
            weight_bits = int(quant_cfg["bits"])
        quant_token = f"w{weight_bits}a{activation_bits}"

        max_pe_length = model_cfg.get("max_pe_length")
        if max_pe_length is not None:
            max_pe_length = int(max_pe_length)
        else:
            def find_max_position_embeddings(data: Any) -> Any:
                if isinstance(data, Mapping):
                    if "max_position_embeddings" in data:
                        return data["max_position_embeddings"]
                    for value in data.values():
                        found = find_max_position_embeddings(value)
                        if found is not None:
                            return found
                elif isinstance(data, list):
                    for value in data:
                        found = find_max_position_embeddings(value)
                        if found is not None:
                            return found
                return None

            hf_model_path = Path(export_model_dir)
            if hf_model_path.is_file() and hf_model_path.name == "config.json":
                config_paths = [hf_model_path]
            elif hf_model_path.exists():
                config_paths = sorted(hf_model_path.rglob("config.json"))
            else:
                config_paths = []

            for config_path in config_paths:
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                max_pe_length = find_max_position_embeddings(config_data)
                if max_pe_length is not None:
                    max_pe_length = int(max_pe_length)
                    break

        if max_pe_length is None:
            return str(model_name_token)

        def length_token(length: int) -> str:
            return f"{length // 1024}k" if length % 1024 == 0 else str(length)

        return (
            f"{chip_token}_{model_name_token}{spec_decode_token}_{quant_token}_{length_token(prefill_chunk_length)}_"
            f"{length_token(context_max_length)}_mpe{length_token(max_pe_length)}"
        ).lower()
