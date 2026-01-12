import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from xhquant.api import (
    HMONNXInference,
)

from xh_model_zoo.xh_llm.models.whisper._model_opt import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=str, default="./data/models/whisper-medium")
    parser.add_argument(
        "--hmonnx-model", type=str, default="./work_dirs/whisper-medium_XH2a"
    )
    args = parser.parse_args()
    main(args)

    device = torch.device("cuda")
    # load model and processor
    hmonnx_dir = args.hmonnx_model
    hf_model = args.hf_model
    processor = WhisperProcessor.from_pretrained(hf_model)
    model = WhisperForConditionalGeneration.from_pretrained(hf_model)
    model.config.forced_decoder_ids = None

    model.model.encoder.decoder_m = model.model.decoder

    # load dummy dataset and read audio files
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = ds[1]["audio"]
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features
    # [1,80,3000]

    # input_features = torch.load("work_dirs/whisper/input.pt")

    encoder = HMONNXInference(
        str(
            Path(hmonnx_dir)
            / "encoder/hmonnx/whisper_meduim_encoder_xh2a_w8a8_sefp.onnx"
        )
    )
    prefill = HMONNXInference(
        str(
            Path(hmonnx_dir)
            / "prefill/hmonnx/whisper_meduim_prefill_xh2a_w8a8_sefp.onnx"
        )
    )
    decoder = HMONNXInference(
        str(
            Path(hmonnx_dir)
            / "decoder/hmonnx/whisper_meduim_decoder_xh2a_w8a8_sefp.onnx"
        )
    )
    encoder.to(device)
    prefill.to(device)
    decoder.to(device)
    encoder.exec_device = device
    prefill.exec_device = device
    decoder.exec_device = device

    detect_ids = torch.tensor([[50258]])  # [1,1]
    default_decoder_ids = torch.tensor([[50258, 0, 50359, 50363]])  # [1,1]
    cache_position = torch.tensor([[0]])
    cache_position_prefill = torch.tensor([[0, 1, 2, 3]])

    # detect language  input_features  detect_ids => [1,51865]
    detect_encoder_out = encoder(input_features.to(device).half())

    mask_atten = torch.ones(([1, 16, 1, 1024])).half()
    mask_atten[:, :, :, 0 + 1 :] *= -65504

    decoder_input_names = decoder.get_input_names()
    decoder_detext_inputs = {
        decoder_input_names[0]: detect_ids.to(device).to(torch.int32),
        decoder_input_names[1]: cache_position.to(device).to(torch.int32),
        decoder_input_names[2]: torch.tensor([0]).to(device).to(torch.int32),
        decoder_input_names[3]: torch.tensor([1]).to(device).to(torch.int32),
        decoder_input_names[4]: mask_atten,
    }

    k_cache = [
        torch.ones([1, 16, 1024, 64], dtype=torch.float16) * (-65504) for i in range(24)
    ]
    v_cache = [
        torch.ones([1, 16, 1024, 64], dtype=torch.float16) * (-65504) for i in range(24)
    ]

    for data_detect, k_data_cache in zip(decoder_input_names[5:29], k_cache):
        decoder_detext_inputs[data_detect] = k_data_cache

    # logits = decoder.run(decoder_detext_inputs)

    for data_detect, v_data_cache in zip(decoder_input_names[29:53], v_cache):
        decoder_detext_inputs[data_detect] = v_data_cache

    k_list = []
    for i in range(24):
        k_list.append(detect_encoder_out[2 * i])

    v_list = []
    for i in range(24):
        v_list.append(detect_encoder_out[2 * i + 1])

    for data_detect, k_data in zip(decoder_input_names[53:77], k_list):
        decoder_detext_inputs[data_detect] = k_data

    for data_detect, v_data in zip(decoder_input_names[77:101], v_list):
        decoder_detext_inputs[data_detect] = v_data

    output = decoder.run(decoder_detext_inputs)

    logits, _, _ = output[0], output[1:25], output[25:49]

    # postprocess  50259
    lang_to_id = [
        50327,
        50334,
        50272,
        50350,
        50304,
        50355,
        50330,
        50292,
        50302,
        50347,
        50309,
        50315,
        50270,
        50283,
        50297,
        50285,
        50261,
        50281,
        50259,
        50262,
        50307,
        50310,
        50300,
        50277,
        50338,
        50265,
        50319,
        50333,
        50352,
        50354,
        50279,
        50276,
        50291,
        50339,
        50286,
        50312,
        50275,
        50311,
        50274,
        50266,
        50356,
        50329,
        50316,
        50323,
        50306,
        50264,
        50294,
        50345,
        50353,
        50336,
        50293,
        50301,
        50349,
        50295,
        50308,
        50296,
        50314,
        50320,
        50282,
        50343,
        50346,
        50313,
        50271,
        50342,
        50288,
        50328,
        50321,
        50269,
        50340,
        50267,
        50284,
        50263,
        50344,
        50332,
        50322,
        50298,
        50305,
        50324,
        50326,
        50317,
        50303,
        50357,
        50273,
        50318,
        50287,
        50299,
        50331,
        50289,
        50341,
        50348,
        50268,
        50351,
        50280,
        50290,
        50337,
        50278,
        50335,
        50325,
        50260,
    ]

    non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
    non_lang_mask[0, list(lang_to_id)] = False
    logits[:, :, non_lang_mask[0]] = -np.inf
    lang_ids = logits.argmax(-1)

    non_lang_mask = torch.ones_like(logits, dtype=torch.bool)
    non_lang_mask[0, 0, list(lang_to_id)] = False
    logits[:, :, non_lang_mask[0][0]] = -np.inf
    lang_ids = logits.argmax(-1)

    # prefill 2221
    default_decoder_ids[0, 1] = lang_ids  # [[50258, 50259, 50359, 50363]] # 34.5197

    mask_atten = torch.ones(([1, 16, 1, 1024])).half()
    mask_atten[:, :, :, 0 + 4 :] *= -65504

    prefill_input_names = prefill.get_input_names()
    prefill_inputs = {
        prefill_input_names[0]: default_decoder_ids.to(device).to(torch.int32),
        prefill_input_names[1]: cache_position_prefill.to(device).to(torch.int32),
        prefill_input_names[2]: torch.tensor([0]).to(device).to(torch.int32),
        prefill_input_names[3]: torch.tensor([4]).to(device).to(torch.int32),
        prefill_input_names[4]: mask_atten,
    }

    for data_detect, k_data_cache in zip(prefill_input_names[5:29], k_cache):
        prefill_inputs[data_detect] = k_data_cache

    for data_detect, v_data_cache in zip(prefill_input_names[29:53], v_cache):
        prefill_inputs[data_detect] = v_data_cache

    for data_detect, k_data in zip(prefill_input_names[53:77], k_list):
        prefill_inputs[data_detect] = k_data

    for data_detect, v_data in zip(prefill_input_names[77:101], v_list):
        prefill_inputs[data_detect] = v_data

    output = prefill.run(prefill_inputs)
    logits, new_k_cache, new_v_cache = output[0], output[1:25], output[25:49]
    next_token_logits = logits[:, -1, :].to(
        copy=True, dtype=torch.float32, device=device
    )
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    default_decoder_ids = torch.cat(
        [default_decoder_ids.to(device), next_tokens[:, None]], dim=-1
    )
    # decoder 2221 <=>   6966

    cnt = 3
    while default_decoder_ids.shape[1] < 448 and next_tokens.item() != 50257:
        cnt += 1

        mask_atten = torch.ones(([1, 16, 1, 1024])).half()
        mask_atten[:, :, :, cnt + 1 :] *= -65504

        prefill_inputs[prefill_input_names[0]] = next_tokens.unsqueeze(0).to(
            torch.int32
        )
        prefill_inputs[prefill_input_names[1]] = (
            torch.tensor([[cnt]]).to(torch.int32).to(device)
        )
        prefill_inputs[prefill_input_names[2]] = (
            torch.tensor([cnt]).to(device).to(torch.int32)
        )
        prefill_inputs[prefill_input_names[3]] = (
            torch.tensor([1]).to(device).to(torch.int32)
        )
        prefill_inputs[prefill_input_names[4]] = mask_atten

        for data_detect, k_data_cache in zip(prefill_input_names[5:29], new_k_cache):
            prefill_inputs[data_detect] = k_data_cache

        for data_detect, v_data_cache in zip(prefill_input_names[29:53], new_v_cache):
            prefill_inputs[data_detect] = v_data_cache

        output = decoder.run(prefill_inputs)
        logits, new_k_cache, new_v_cache = output[0], output[1:25], output[25:49]
        next_token_logits = logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=device
        )
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        default_decoder_ids = torch.cat(
            [default_decoder_ids.to(device), next_tokens[:, None]], dim=-1
        )

    # [50257] 448

    transcription = processor.batch_decode(
        default_decoder_ids, skip_special_tokens=True
    )
    print(transcription)

    # onnx =========================================
    # input_names = [inp.name for inp in encoder_session.get_inputs()]
    # detect_encoder_out = encoder_session.run( None, {input_names[0]:input_features.cpu().numpy()} )
    # decoder_input_names = decoder.get_input_names()
    # decoder_detext_inputs = {
    #     decoder_input_names[0]: detect_ids.cpu().numpy() ,
    #     decoder_input_names[1]: cache_position.cpu().numpy() ,
    # }

    # for data_detect, kv_data in zip(decoder_input_names[2:], detect_encoder_out):
    #     decoder_detext_inputs[data_detect] = kv_data

    # logits = decoder_session.run(None,decoder_detext_inputs)

    # # postprocess  50259
    # lang_to_id = [50327, 50334, 50272, 50350, 50304,
    #     50355, 50330, 50292, 50302, 50347, 50309, 50315, 50270,
    #     50283, 50297, 50285, 50261, 50281, 50259, 50262, 50307,
    #     50310, 50300, 50277, 50338, 50265, 50319, 50333, 50352,
    #     50354, 50279, 50276, 50291, 50339, 50286, 50312, 50275,
    #     50311, 50274, 50266, 50356, 50329, 50316, 50323, 50306,
    #     50264, 50294, 50345, 50353, 50336, 50293, 50301, 50349,
    #     50295, 50308, 50296, 50314, 50320, 50282, 50343, 50346,
    #     50313, 50271, 50342, 50288, 50328, 50321, 50269, 50340,
    #     50267, 50284, 50263, 50344, 50332, 50322, 50298, 50305,
    #     50324, 50326, 50317, 50303, 50357, 50273, 50318, 50287,
    #     50299, 50331, 50289, 50341, 50348, 50268, 50351, 50280,
    #     50290, 50337, 50278, 50335, 50325, 50260]

    # logits = torch.from_numpy(logits[0])

    # non_lang_mask = torch.ones_like(logits, dtype=torch.bool)
    # non_lang_mask[0, 0, list(lang_to_id)] = False
    # logits[:, :, non_lang_mask[0][0]] = -np.inf
    # lang_ids = logits.argmax(-1)

    # # prefill 2221
    # default_decoder_ids[0,1] = 50259 # lang_ids # [[50258, 50259, 50359, 50363]] # 34.5197

    # prefill_input_names = prefill.get_input_names()
    # prefill_inputs = {
    #     prefill_input_names[0]: default_decoder_ids.cpu().numpy(),
    #     prefill_input_names[1]: cache_position_prefill.cpu().numpy(),
    # }

    # for data_prefill, kv_data in zip(prefill_input_names[2:], detect_encoder_out):
    #     prefill_inputs[data_prefill] = kv_data

    # logits = prefill_session.run(None, prefill_inputs)
    # logits = torch.from_numpy(logits[0])
    # next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32, device=device)
    # next_tokens = torch.argmax(next_token_logits, dim=-1)
    # default_decoder_ids = torch.cat([default_decoder_ids.to(device), next_tokens[:, None]], dim=-1)
    # # decoder 2221

    # cnt = 3
    # while default_decoder_ids.shape[1] < 448 and next_tokens.item() != 50257:
    #     cnt += 1
    #     prefill_inputs[prefill_input_names[0]] = next_tokens.unsqueeze(0).cpu().numpy()
    #     prefill_inputs[prefill_input_names[1]] = torch.tensor([[cnt]]).cpu().numpy()

    #     logits = decoder_session.run(None,prefill_inputs)[0]
    #     logits = torch.from_numpy(logits)
    #     next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32, device=device)

    #     scores = next_token_logits
    #     input_ids = default_decoder_ids

    #     # post process 1  88913.0938
    #     begin_suppress_tokens = torch.tensor([220, 50257], device=device)
    #     vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
    #     suppress_token_mask = isin_mps_friendly(vocab_tensor, begin_suppress_tokens)
    #     scores_processed = scores
    #     if input_ids.shape[-1] == 4:
    #         scores_processed = torch.where(suppress_token_mask, -float("inf"), scores)

    #     scores = scores_processed
    #     # post process 2
    #     suppress_token = torch.tensor([
    #         1,     2,     7,     8,     9,    10,    14,    25,    26,    27,
    #         28,    29,    31,    58,    59,    60,    61,    62,    63,    90,
    #         91,    92,    93,   359,   503,   522,   542,   873,   893,   902,
    #         918,   922,   931,  1350,  1853,  1982,  2460,  2627,  3246,  3253,
    #         3268,  3536,  3846,  3961,  4183,  4667,  6585,  6647,  7273,  9061,
    #         9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635,
    #         15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130,
    #         26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425,
    #         49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362
    #     ], device=device)

    #     vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
    #     suppress_token_mask = isin_mps_friendly(vocab_tensor, suppress_token)
    #     scores = torch.where(suppress_token_mask, -float("inf"), scores)

    #     next_tokens = torch.argmax(scores, dim=-1)
    #     default_decoder_ids = torch.cat([default_decoder_ids.to(device), next_tokens[:, None]], dim=-1)

    # # [[50258, 50259, 50359, 50363,  2221,    13,  2326,   388,   391,   307,
    # #       264, 50244,   295,   264,  2808,  5359,   293,   321,   366,  5404,
    # #       281,  2928,   702, 14943,    13, 50257]]

    # transcription = processor.batch_decode(default_decoder_ids, skip_special_tokens=True)


if __name__ == "__main__":
    main()
