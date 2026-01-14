import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import yaml
from modelscope import AutoTokenizer, AutoModelForMaskedLM
from xhquant.api import CacheTensor
from ....datasets.preprocess.mix_search_preprocess import ms_data_preprocess
from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model

from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    PrecisionMode,
    CacheTensor,
)


class BertConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
        model = AutoModelForMaskedLM.from_pretrained(hf_model_dir)

        self.hf_model_path = hf_model_dir
        return model

    def _convert(self, native_model, output_dir: str, tokenizer=None):
        logger = get_root_logger()
        config = self.config
        device = "cuda"

        # native_model = self.load_hf_model(
        #     hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
        # )

        # 融合GPTQ权重
        resume_from = self.config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, native_model)
        # lm_head = native_model.lm_head
        # if not hasattr(lm_head, "quant_weight"):
        #     config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        model_name = "bert_ch"
        target_device = config.quant_scheme.target_device
        # batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"
        quant_type = config.quant_scheme.quant_type
        quant_config = create_quant_config(config.quant_scheme)
        quant_config = ConfigDict(quant_config)

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        # meta_info["hf_model_path"] = hf_model_path
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["quant_weight"] = resume_from

        work_dir = Path(output_dir)

        token_embedding = native_model.bert.embeddings.word_embeddings
        
        token_type_ids = torch.zeros((1,context_length), dtype=torch.long, device=device)
        token_type_embeddings = native_model.bert.embeddings.token_type_embeddings(token_type_ids)

        position_ids = torch.arange(context_length, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = native_model.bert.embeddings.position_embeddings(position_ids)

        atten_mask = torch.zeros((1,context_length), device=device)
        # attention_mask_padded = torch.ones( (1, target_length - current_length), device=device)* -torch.inf
        # attention_mask_padded = torch.concat([attention_mask, attention_mask_padded], dim=1).unsqueeze(0).unsqueeze(0)


        token_embedding_file = Path(work_dir) / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        input_txt = "你好"
        input_ids = tokenizer(
            input_txt, return_tensors="pt", padding="max_length", max_length=context_length
        ).input_ids.cuda()
        output = native_model(input_ids)

        from ._model import register_wrap_modules as bert_register_wrap_modules  # noqa: F403, F401
        bert_register_wrap_modules(native_model)

        # 改写原模型, 以适合torch.fx导出
        wrap_cfg = Config(
            dict(
                # batch_size=batch_size,
                max_sequence_length=context_length,  # 最大上下文长度
                input_sequence_length=input_sequence_length,  # prefill时输入的序列长度
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()
        wraped_qwen_model = wrap_llm_model(native_model, wrap_cfg)

        input_emb = token_embedding(input_ids)

        warp_out = wraped_qwen_model(
            input_emb, token_type_embeddings, position_embeddings, atten_mask
        )

        inputs = [
            input_emb, 
            token_type_embeddings, 
            position_embeddings, 
            atten_mask
        ]

        input_names = [
            "input_emb",
            "token_type_embeddings",
            "position_embeddings",
            "atten_mask",
        ]
        output_names = ["logits"]

        prefix = f"{model_name}-{target_device}-{context_length//1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info(f"********************* start export prefill model *********************")

        quanted_model = convert_fx_model_to_quanted_model(
            wraped_qwen_model,
            inputs,
            target_device,
            quant_config=quant_config,
        )

        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(quanted_model, inputs, str(prefill_onnx_file), input_names, output_names)
        logger.info(f"Export Prefill model to {prefill_onnx_file}")

        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        BertConverterXH2a(config)._convert(hf_model_path, output_dir)
