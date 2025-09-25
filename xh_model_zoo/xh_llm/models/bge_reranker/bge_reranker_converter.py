import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import onnx
import onnxsim
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel
from transformers import BertModel
from xhquant.api import convert_onnx_to_hmonnx
from xhquant.api import get_root_logger
from xhquant.api import xhquant_init

from ..base_converter import HFTransfromersConverter
from .bge_reranker_convert_config import BGERerankerConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    DeviceType,
    ConfigDict,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    HMONNXGoldenInference,
)


class Net(nn.Module):
    def __init__(self, bge_model: BertModel):
        super().__init__()
        self._bge_model = bge_model

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor):
        return self._bge_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]


class BGERerankerConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: BGERerankerConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs) -> Any:
        model: BertModel = AutoModel.from_pretrained(hf_model_dir, device_map="cpu", torch_dtype=torch.float16)
        model.eval()  # type:ignore
        return model

    def _convert(self, hf_model_path: str, output_dir: str):
        model_name = Path(hf_model_path).name
        target_device = self.config.quant_scheme.target_device
        work_dir = Path(output_dir)

        logger = get_root_logger()
        config = self.config

        native_model = self.load_hf_model(
            hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
        )
        device = "cuda"
        wraped_model = Net(native_model)
        wraped_model.to(device)

        bz = config.batch_size
        context_length = config.context_length
        input_ids = torch.randint(0, 100, (bz, context_length)).to(device)
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

        onnx_file = str(Path(work_dir) / "onnx" / f"{model_name}.onnx")
        Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
        if not Path(onnx_file).exists():
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                logger.info(f"export onnx model to {temp_onnx_file}")
                torch.onnx.export(
                    wraped_model,
                    (input_ids, token_type_ids, attention_mask),
                    temp_onnx_file,
                    input_names=["input_ids", "token_type_ids", "attention_mask"],
                    output_names=[
                        "hidden_state",
                    ],
                )
                onnx_model = onnx.load(temp_onnx_file)

                logger.info(f"simplify onnx model")
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
            logger.info(f"convert onnx model to hmonnx model........")
            convert_onnx_to_hmonnx(
                onnx_file,
                (input_ids.cpu(), token_type_ids.cpu(), attention_mask.cpu()),
                DeviceType.XH2a,
                hmonnx_file,
                input_names=["input_ids", "token_type_ids", "attention_mask"],
                output_names=[
                    "hidden_state",
                ],
            )
        else:
            logger.warning(f"hmonnx model {hmonnx_file} is exists, skip.")

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
        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: BGERerankerConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        assert not is_ssfp, f"不支持SSFP量化"
        BGERerankerConverterXH2a(config)._convert(hf_model_path, output_dir)
