# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from transformers import AutoModel, BertModel
from xhquant.api import convert_onnx_to_hmonnx, get_root_logger

from ..base_converter import HFTransfromersConverter
from .stella_mrl_convert_config import StellaMRLConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    ConfigDict,
    DeviceType,
    HMONNXGoldenInference,
    create_quant_config,
    is_ssfp_quant_config,
)


class Net(nn.Module):
    def __init__(self, bert_model: BertModel, dense_linear: nn.Linear, output_normalized: bool = False):
        super().__init__()
        self.bert_model = bert_model
        # Use MatMul + Add in forward to avoid exporting Gemm without attrs.
        self.register_buffer("dense_weight_t", dense_linear.weight.transpose(0, 1).contiguous())
        if dense_linear.bias is not None:
            self.register_buffer("dense_bias", dense_linear.bias.contiguous())
        else:
            self.dense_bias = None  # type: ignore[assignment]
        self.output_normalized = output_normalized

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor):
        last_hidden_state = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )[0]

        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        masked_hidden = last_hidden_state * mask
        token_count = torch.clamp(mask.sum(dim=1), min=1e-6)
        pooled = masked_hidden.sum(dim=1) / token_count

        sentence_embedding = torch.matmul(pooled, self.dense_weight_t)
        if self.dense_bias is not None:
            sentence_embedding = sentence_embedding + self.dense_bias
        if self.output_normalized:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        return sentence_embedding


class StellaMRLConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: StellaMRLConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs) -> Any:
        model: BertModel = AutoModel.from_pretrained(hf_model_dir, **kwargs)
        assert isinstance(model, BertModel), f"Expected BertModel, but got {type(model)}"
        model.eval()  # type: ignore
        self.hf_model_path = hf_model_dir
        return model

    def _build_dense_layer(self, hf_model_path: str, device: str) -> nn.Linear:
        dense_cfg = json.load(open(Path(hf_model_path) / "2_Dense" / "config.json", "r", encoding="utf-8"))
        dense_weight = load_safetensors_file(str(Path(hf_model_path) / "2_Dense" / "model.safetensors"))

        dense = nn.Linear(
            in_features=dense_cfg["in_features"],
            out_features=dense_cfg["out_features"],
            bias=dense_cfg.get("bias", True),
        )
        dense.weight.data.copy_(dense_weight["linear.weight"])
        if dense.bias is not None and "linear.bias" in dense_weight:
            dense.bias.data.copy_(dense_weight["linear.bias"])
        dense.eval()
        dense.to(device=device, dtype=torch.float16)
        return dense

    def _convert(self, hf_model_path: str, output_dir: str):
        model_name = Path(hf_model_path).name
        target_device = self.config.quant_scheme.target_device
        work_dir = Path(output_dir)

        logger = get_root_logger()
        config = self.config

        native_model = self.load_hf_model(
            hf_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        device = "cuda"
        native_model.to(device)
        dense_linear = self._build_dense_layer(hf_model_path, device)
        wrapped_model = Net(native_model, dense_linear, output_normalized=config.output_normalized).to(device)

        bz = config.batch_size
        context_length = config.context_length
        input_ids = torch.randint(0, 100, (bz, context_length), dtype=torch.int32).to(device)
        token_type_ids = torch.zeros((bz, context_length), dtype=torch.int32).to(device)
        attention_mask = torch.ones((bz, context_length), dtype=torch.int16).to(device)

        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"
        quant_type = config.quant_scheme.quant_type
        quant_config = create_quant_config(config.quant_scheme)
        quant_config = ConfigDict(quant_config)

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        meta_info["hf_model_path"] = hf_model_path
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["output_normalized"] = config.output_normalized
        meta_info["output_dim"] = int(dense_linear.out_features)

        onnx_file = str(Path(work_dir) / "onnx" / f"{model_name}.onnx")
        Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
        if not Path(onnx_file).exists():
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                logger.info(f"export onnx model to {temp_onnx_file}")
                torch.onnx.export(
                    wrapped_model,
                    (input_ids, token_type_ids, attention_mask),
                    temp_onnx_file,
                    input_names=["input_ids", "token_type_ids", "attention_mask"],
                    output_names=["sentence_embedding"],
                )
                onnx_model = onnx.load(temp_onnx_file)

                logger.info("simplify onnx model")
                onnx_model_sim, checked = onnxsim.simplify(onnx_model)
                if checked:
                    onnx_model = onnx_model_sim
        else:
            logger.info(f"load onnx model from {onnx_file}")
            onnx_model = onnx.load(onnx_file)

        logger.info(f"save onnx model to {onnx_file}")
        if not os.path.exists(onnx_file):
            onnx.save(
                onnx_model,
                onnx_file,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{Path(onnx_file).stem}_external_data",
            )

        prefix = f"{model_name}-{target_device}-{quant_type}-{bz}x{context_length}"
        hmonnx_file = work_dir / "hmonnx" / f"{prefix}.onnx"
        hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["hmonnx_file"] = str(hmonnx_file.relative_to(work_dir))

        if not Path(hmonnx_file).exists():
            logger.info("convert onnx model to hmonnx model........")
            convert_onnx_to_hmonnx(
                onnx_file,
                (input_ids.cpu(), token_type_ids.cpu(), attention_mask.cpu()),
                DeviceType.XH2a,
                hmonnx_file,
                input_names=["input_ids", "token_type_ids", "attention_mask"],
                output_names=["sentence_embedding"],
            )
        else:
            logger.warning(f"hmonnx model {hmonnx_file} exists, skip.")

        session = HMONNXGoldenInference(hmonnx_file)
        session.to(device)
        session.save_golden = True
        session.golden_dir = work_dir / "hmonnx/golden"
        session.step = 0
        session(
            input_ids.to(torch.int32).to(device),
            token_type_ids.to(torch.int32).to(device),
            attention_mask.to(torch.int16).to(device),
        )

        logger.info(f"Export model to {hmonnx_file}")
        json.dump(meta_info, open(work_dir / "meta.json", "w", encoding="utf-8"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: StellaMRLConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        assert not is_ssfp, "stella_mrl converter does not support SSFP quantization"
        StellaMRLConverterXH2a(config)._convert(hf_model_path, output_dir)
