from modelscope import AutoTokenizer, AutoModelForMaskedLM
import torch
from xh_model_zoo.xh_llm.models.bert.bert_converter import BertConverterXH2a
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger 
import argparse
from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLConvertConfig, VisualConfig

def main(args):

    tokenizer = AutoTokenizer.from_pretrained("/data02/datasets/bert_chinese")

    model = AutoModelForMaskedLM.from_pretrained("/data02/datasets/bert_chinese")
    model = model.to("cuda")

    input_txt = "你好"
    input_ids = tokenizer(
        input_txt, return_tensors="pt", padding="max_length", max_length=args.context_length
    ).input_ids

    output = model(input_ids.cuda())
    target_device = DeviceType.XH2a

    quant_type = args.quant_type
    # ops=dict(MatMul=dict(
    #             act_scheme=dict(
    #                 bits=8,
    #                 fp_mode="sefp",
    #             ),
    #             act_schema_2=dict(
    #                 bits=16,
    #                 fp_mode="sefp",
    #             ),))

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type) # , ops=ops

    config = Qwen2_5_VLConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        gptqmodel_cfg=args.use_gptqmodel,
        max_pe_length=args.max_pe_length,
    )

    BertConverterXH2a(config)._convert(
        model,
        "work_dirs/bert",
        tokenizer,
    )
    # current_logits = output.logits[:, -1, :]
    # probs = torch.softmax(current_logits, dim=-1)
    # next_token = torch.argmax(probs, dim=-1).item()
    # reply = tokenizer.decode(next_token)
    # print(reply)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="weights/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=256, help="max sequence length")
    parser.add_argument("--max_pe_length", type=int, default=32768, help="max pe length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--image_max_size_h", type=int, default=448, help="image max size height")
    parser.add_argument("--image_max_size_w", type=int, default=448, help="image max size width")
    parser.add_argument("--image_max_size_t", type=int, default=2, help="if image, temporal max size is 2, if video, temporal max size is fps")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument("--sample_image_path", type=str, default="data/images/qwen2_vl_demo.jpeg", help="sample image path for generate golden")
    parser.add_argument("--use_gptqmodel", action="store_true", help="use gptqmodel quanted model")
    parser.add_argument(
        "--quant_weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    main(args)
