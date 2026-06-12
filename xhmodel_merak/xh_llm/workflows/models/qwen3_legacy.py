from collections.abc import Mapping
from typing import Any
from pathlib import Path


from ..base import BaseHMONNXWorkflow
from ..result import ExportResult, QuantResult


class XHQwen3LegacyHMONNXWorkflow(BaseHMONNXWorkflow):
    expected_model_config_cls_name = "XHQwen3LegacyModelConfig"
    expected_model_cls_name = "XHQwen3LegacyModel"

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is None:
            # 量化配置为None表示无需量化
            return QuantResult(hf_model_dir=self.hf_model_dir, skipped=True)
        
        # step1 导入依赖
        from datasets import load_dataset
        from gptqmodel import GPTQModel, QuantizeConfig

        # step2 获取quant配置
        bits = workflow_config.quant["bits"]
        save_path = str(Path(output_dir) / f"{Path(self.hf_model_dir).name}-gptqmodel-{bits}bit")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        calibration_dataset = [
            text for text in dataset["text"] 
            if text.strip() and len(text.strip()) > 50  
        ][:128]
        
        # step3 执行量化
        quant_config = QuantizeConfig(bits=bits, group_size=64)
        model = GPTQModel.load(self.hf_model_dir, quant_config, device=device)
        model.quantize(calibration_dataset, batch_size=128)
        model.save(save_path)

        # step4 构造QuantResult并返回
        return QuantResult(hf_model_dir=self.hf_model_dir, quanted_model_dir=save_path)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        export_result = super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )
        return export_result

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
    ) -> str:
        from transformers import TextStreamer

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
        from xhquant.api import get_xhquant_logger
        from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

        meta_file = self._find_golden_meta_file(export_result)
        logger = get_xhquant_logger()
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(meta_file)
        # processor = hmonnx_model.get_tf_processor()
        # tokenizer = processor.tokenizer
        tokenizer = hmonnx_model.get_tokenizer()

        messages = self.build_input_message(input_messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt", truncation=True)
        model_inputs = model_inputs.to(device)

        streamer = TextStreamer(tokenizer)
        hmonnx_model.to(device)
        hmonnx_model.enable_golden = True
        # logger.warning("Golden outputs should be generated in aligned precision for stability.")

        contexts = [
            TimeProfiler("hmonnx_generate_golden", logger),
            MemoryTracker(device=device, name="generate_golden", logger=logger),
            LLMInferenceContextManager(hmonnx_model),
        ]
        with ContextManagers(contexts):
            generated_ids = hmonnx_model.generate(
                **model_inputs,
                max_new_tokens=2,
                streamer=streamer,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 模型输出对于dump_golden非必须
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        logger.info(f"{'-' * 20} Golden output {'-' * 20}")
        logger.info(f"{content}")

        return meta_file

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

    def build_input_message(self, input_messages: Any) -> list[dict[str, Any]]:
        if isinstance(input_messages, str):
            prompt = input_messages
        elif isinstance(input_messages, Mapping):
            if "text" not in input_messages:
                raise ValueError("Qwen3 legacy input_messages must contain 'text'")
            prompt = input_messages["text"]
        else:
            raise ValueError("Qwen3 legacy input_messages must be a string or a mapping with 'text'")

        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Qwen3 legacy input text must be a non-empty string")
        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]


__all__ = ["XHQwen3LegacyHMONNXWorkflow"]
