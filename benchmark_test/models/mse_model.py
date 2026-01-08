from typing import Dict, List, Optional, Sequence, Union
import torch
from torch import Tensor
from xhmodel_zoo.data_preprocessors import BaseDataPreprocessor
from xhmodel_zoo.registry import MODELS
from xhmodel_zoo.structures import DetSampleList, InstanceData
from xhmodel_zoo.utils.typing_utils import OptConfigType
from xhmodel_zoo.models.onnx_model import ONNXModel
from xhmodel_zoo.models.utils.box_utils import non_max_suppression_v8
from xhmodel_zoo.models.utils.det_utils import add_pred_to_datasample
from xhquant.api import (
    ExportedGraph,
    FrontendGraph,
    FrontendType,
    FXInterpreter,
    Hook,
    PrecisionMode,
    QuantGraph,
    QuantizerState,
    disable_quant,
    enable_aligned_precision_mode,
    enable_fast_precision_mode,
    enable_quant,
    export_onnx,
    get_precision_mode,
    get_quant_state,
    is_quant_fixed,
    to_export_graph,
    to_frontend_graph,
    to_quant_graph,
)

@MODELS.register_module()
class MSEOnnxModel(ONNXModel):
    def __init__(
        self,
        onnx_path: str = "",
        onnx_cache_dir="data/models/onnx",
        external_data_file: Optional[Union[str, List[str]]] = None,
        data_preprocessor: Optional[Union[dict, BaseDataPreprocessor]] = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: Optional[dict] = None,
        num_classes=80,
        without_postprocess=False,
        **kwargs,
    ):
        """
        without_postprocess: onnx模型的输出,是否包含了后处理逻辑
        """
        super().__init__(
            onnx_path=onnx_path,
            onnx_cache_dir=onnx_cache_dir,
            external_data_file=external_data_file,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            **kwargs,
        )
        self.nc = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        name, input_shape = self.onnx_inputs[0]
        self.bs = input_shape[0]
        self.input_shape = input_shape[2:]
        assert self.bs == 1, "Only support batch size 1 for now."
        self.without_postprocess = without_postprocess

    def forward(
        self, inputs: Tensor, data_samples: Optional[list] = None, mode: str = "tensor"
    ) -> Union[Dict[str, Tensor], list]:
        if mode == "tensor":
            # feats = self.extract_feat(inputs)
            # return self.head(feats) if self.with_head else feats
            raise NotImplementedError(f"Not implemented for mode={mode} yet.")
        elif mode == "loss":
            # return self.loss(inputs, data_samples)
            raise NotImplementedError(f"Not implemented for mode={mode} yet.")
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def predict(
        self,
        inputs: Tensor,
        batch_data_samples: Optional[DetSampleList] = None,
        **kwargs,
    ) -> DetSampleList:  # type: ignore
        # assert len(inputs) == 1, "Only support batch size 1 for now."
        batch_nn_out = self.extract_feat(inputs)
        return batch_nn_out

    def convert_to_export_graph(self, data: Union[dict, tuple, list], include_none_exported_attrs=False, device="cuda") -> Optional[ExportedGraph]:
        # logger = get_root_logger()
        # if not self.allow_quant:
        #     logger.warning(f"{type(self).__name__} does not allow quantization, skip convert to exported graph.")
        #     return None

        if self.exported_model is not None:
            return self.exported_model
        assert self.quanted_model is not None, "Quantized model is not available, Please call `to_quant_graph` first."
        if not self.quanted_model.is_fixed():
            none_fixed_modules = self.quanted_model.get_none_fixed_modules()
            raise ValueError(
                f"Quantized model is not fixed, Please call `fixed` first. None fixed modules: {none_fixed_modules}"
            )

        if self.data_preprocessor is not None:
            data = self.data_preprocessor(data, False)
        inputs = self.prepare_inputs(data, device)
        # inputs[0] = inputs[0].repeat(30,1,1,1)
        exported_model = to_export_graph(self.quanted_model.to(device), inputs, include_none_exported_attrs=include_none_exported_attrs)
        # logger.debug(f"************* Start Exported model *************")
        # logger.debug(f"{exported_model.graph}")
        # logger.debug(f"************* End Exported model *************")

        self.exported_model = exported_model
        return self.exported_model
