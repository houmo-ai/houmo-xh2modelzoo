import copy
import shutil
from pathlib import Path
from typing import Any, cast

from xhmodel_merak.utils import calculate_file_md5
from xhmodel_merak.xh_llm.builder import register_llm_model
from xhmodel_merak.xh_llm.models.qwen3_vl import (
    XHQwen3VLModel,
    XHQwen3VLModelConfig,
)
from xhmodel_merak.xh_llm.types import (
    ExportData,
    LLMModelMeta,
    LLMModelState,
    ModelSwitcher,
    VLLMModelMeta,
)
from xhmodel_merak.xh_llm.utils import unfold_args
from xhquant.api import (
    get_xhquant_logger,
    to_export_graph,
    to_export_hmonnx_v2,
)
from xhquant.utils import log_function_call

from .hmonnx_inference import XHQwen3VLEmbeddingHMONNXModel


class XHQwen3VLEmbeddingModelConfig(XHQwen3VLModelConfig):
    def __init__(
        self,
        *,
        output_hidden_states_for_export: bool = True,
        num_logits_to_keep: int = 0,
        **kwargs: Any,
    ):
        super().__init__(
            num_logits_to_keep=num_logits_to_keep,
            output_hidden_states_for_export=(
                output_hidden_states_for_export
            ),
            **kwargs,
        )
        self.output_hidden_states_for_export = (
            output_hidden_states_for_export
        )


@register_llm_model("Qwen3VLForConditionalGeneration_embedding")
class XHQwen3VLEmbeddingModel(XHQwen3VLModel):
    transformers_min_version = "4.57.3"
    CONFIG_CLS = XHQwen3VLEmbeddingModelConfig
    HMONNXINFERENCE_CLS = XHQwen3VLEmbeddingHMONNXModel
    WORKFLOW_CLS = (
        "xhmodel_merak.xh_llm.models.qwen3_vl_embedding.workflow:"
        "XHQwen3VLEmbeddingWorkflow"
    )

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = super().get_export_cfg()
        export_cfg["output_names"] = ["hidden_states"]
        return export_cfg

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._llm_model_impl import embedding_wrap_cls_scope

        with embedding_wrap_cls_scope():
            return super().init_wrap_model(hf_model)

    def to_wrap(self):
        from ._vision_model_impl import (
            embedding_vision_wrap_cls_scope,
        )

        with embedding_vision_wrap_cls_scope():
            return super().to_wrap()

    @log_function_call()
    def export_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        """Export runtime artifacts without a separate metadata file."""
        logger = get_xhquant_logger()
        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.fixed()
        self.visual.quanted_model.fixed()

        exported_info = self.get_export_info(output_dir)
        self.config.model_name = exported_info.model_name
        visual_output_dir = str(
            Path(exported_info.exported_dir) / "visual"
        )
        self.visual.config.model_name = (
            f"{exported_info.model_name}_"
            f"{self.visual.config.max_size_w}x"
            f"{self.visual.config.max_size_h}"
        )
        visual_meta = self.visual.export_hmonnx(
            visual_output_dir
        )
        visual_meta.hmonnx = str(
            Path(visual_meta.hmonnx)
            .relative_to(exported_info.exported_dir)
            .as_posix()
        )

        meta_info = cast(VLLMModelMeta, exported_info.meta)
        if not isinstance(meta_info, VLLMModelMeta):
            raise TypeError(
                "Expected VLLMModelMeta, got "
                f"{type(meta_info).__name__}"
            )
        meta_info.visual_config = visual_meta
        self._export_hmonnx(exported_info)
        logger.info(
            "Exporting completed! Exported model is saved at: "
            f"{output_dir}"
        )
        return meta_info

    @log_function_call()
    def _export_hmonnx(self, exported_info: ExportData):
        """Export the embedding Prefill graph without a Decode graph."""
        meta_info = exported_info.meta
        logger = get_xhquant_logger()
        model_name = exported_info.model_name
        output_dir = Path(exported_info.exported_dir)
        meta_info.kv_cache = self.kvcache_config

        with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
            self.set_input_sequence_length(
                self.wrap_cfg.prefill_chunk_length
            )
            if isinstance(self._quanted_model, ModelSwitcher):
                quanted_model = self._quanted_model.prefill
            else:
                quanted_model = self._quanted_model

            if not quanted_model.is_fixed():
                raise ValueError(
                    "prefill_quanted_model is not fixed; "
                    "call `fixed` before export"
                )
            quanted_model.to("cpu")

            logger.info(
                "Exporting Prefill for %s model .........",
                model_name,
            )
            self.set_prefill()
            data_processor = self.get_data_preprocessor()
            dummy_input = self.get_dummy_inputs()
            data_processor.input_sequence_length = (
                self.config.prefill_chunk_length
            )
            inputs = unfold_args(data_processor(dummy_input))
            exported_model = to_export_graph(quanted_model, inputs)

            logger.info(
                "Exporting Prefill to HMONNX format for %s .........",
                model_name,
            )
            prefill_dir = output_dir / "prefill"
            prefill_dir.mkdir(parents=True, exist_ok=True)
            prefill_file = (
                prefill_dir / f"{model_name}_prefill.onnx"
            )
            export_cfg = copy.deepcopy(self.get_export_cfg())
            if "input_names" in export_cfg or hasattr(
                export_cfg,
                "input_names",
            ):
                export_cfg["input_names"] = self.xh1_hmonnx_compatible(
                    export_cfg["input_names"]
                )
            prefill_file = to_export_hmonnx_v2(
                exported_model,
                inputs,
                str(prefill_file),
                export_cfg,
                normalize_onnx_name=True,
            )
            meta_info.prefill_hmonnx_md5 = calculate_file_md5(
                prefill_file
            )
            meta_info.prefill_hmonnx = str(
                Path(prefill_file).relative_to(output_dir)
            )
        return exported_info

    def _extra_export_metadata(
        self,
        output_dir: str,
        meta_info: LLMModelMeta,
    ) -> LLMModelMeta:
        meta_info = super()._extra_export_metadata(
            output_dir,
            meta_info,
        )
        hf_config_dir = Path(output_dir) / meta_info.hf_config
        for filename in (
            "merges.txt",
            "tokenizer.model",
            "spiece.model",
        ):
            source = Path(self.config.hf_model) / filename
            if source.is_file():
                shutil.copyfile(source, hf_config_dir / filename)
        meta_info.embedding_output = "hidden_states"
        meta_info.embedding_pooling = "last_token"
        meta_info.embedding_normalize = True
        meta_info.image_token_id = self.config.image_token_id
        meta_info.video_token_id = self.config.video_token_id
        meta_info.vision_start_token_id = (
            self.config.vision_start_token_id
        )
        meta_info.vision_end_token_id = self.config.vision_end_token_id
        meta_info.spatial_merge_size = self.config.spatial_merge_size
        return meta_info
