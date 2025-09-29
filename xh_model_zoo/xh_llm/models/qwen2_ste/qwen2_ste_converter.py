import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from shapely import length
from torch import Tensor
from tqdm.autonotebook import trange
from transformers import AutoModel, BertModel
from xhquant.api import convert_onnx_to_hmonnx, get_root_logger, xhquant_init

from ..base_converter import HFTransfromersConverter
from .qwen2_ste_convert_config import SteQwen2ConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    DeviceType,
    ConfigDict,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    HMONNXGoldenInference,
)


class Net(nn.Module):
    def __init__(self, ste_model: BertModel, config=None):
        super().__init__()
        self.config = config
        self.ste_model = ste_model
        self.prompts = ste_model.prompts

    def preprocess(self, sentences, prompt_name=None):
        if prompt_name is not None:
            prompt = self.prompts[prompt_name]
        else:
            prompt = None

        if prompt is not None and len(prompt) > 0:
            extra_features = {}
            sentences = [prompt + sentence for sentence in sentences]
            length = self.ste_model._get_prompt_length(prompt)
            extra_features["prompt_length"] = length

        all_embeddings = []
        length_sorted_idx = np.argsort([-self.ste_model._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        feats = []
        for start_index in trange(0, len(sentences), self.config.batch_size, desc="Batches"):
            sentences_batch = sentences_sorted[start_index : start_index + self.config.batch_size]
            features = self.ste_model.tokenize(sentences_batch)
            feats.append(features)
        return feats

    def sen_pool(self, attention_mask, token_embeddings):
        output_vectors = []
        bs, seq_len, hidden_dim = token_embeddings.shape

        values, indices = attention_mask.flip(1).max(1)
        indices = torch.where(values == 0, seq_len - 1, indices)
        gather_indices = seq_len - indices - 1

        # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
        gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
        gather_indices = gather_indices.unsqueeze(1)
        assert gather_indices.shape == (bs, 1, hidden_dim)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        output_vectors.append(embedding)

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def forward(self, input_ids, attention_mask):
        trans_out = self.ste_model[0].auto_model(input_ids, attention_mask)[0]
        pool_out = self.sen_pool(attention_mask, trans_out)
        out = F.normalize(pool_out, p=2, dim=1)
        return out


class SteQwen2ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: SteQwen2ConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs) -> Any:
        # model: BertModel = AutoModel.from_pretrained(hf_model_dir, device_map="cpu", torch_dtype=torch.float16)
        model = SentenceTransformer(hf_model_dir, trust_remote_code=True)
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
        wraped_model = Net(native_model, self.config)
        wraped_model.to(device)

        bz = config.batch_size
        context_length = config.context_length
        input_ids = torch.randint(0, 100, (bz, context_length)).to(device)
        attention_mask = torch.ones((bz, context_length), dtype=torch.int16).to(device)

        # queries = [
        #     "how much protein should a female eat",
        #     "summit define",
        # ]

        # inputs = wraped_model.preprocess(queries)

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
                    (input_ids, attention_mask),  # inputs[0], #
                    temp_onnx_file,
                    input_names=["input_ids", "attention_mask"],
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
                (input_ids.cpu(), attention_mask.cpu()),
                DeviceType.XH2a,
                hmonnx_file,
                input_names=["input_ids", "attention_mask"],
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
        session(input_ids.to(torch.int32).to(device), attention_mask.to(torch.int16).to(device))

        logger.info(f"Export model to {hmonnx_file}")
        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: SteQwen2ConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        assert not is_ssfp, f"不支持SSFP量化"
        SteQwen2ConverterXH2a(config)._convert(hf_model_path, output_dir)
