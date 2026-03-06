# FireRedASR Migration Validation Report (2026-03-05)

## 1. Baseline

- HF baseline command:

```bash
CUDA_VISIBLE_DEVICES=2 python examples/audio/fireredasr/fireredasr_hf_forward.py \
  --mode hf \
  --model_dir weights/FireRedASR-LLM-L \
  --wav_path data/wav \
  --ref_file data/wav/text \
  --use_gpu --beam_size 1 --decode_max_len 32 --temperature 0
```

- Result: `cer=0.011235955056179775` on 4 utts.

## 2. Flow A: no common quant (`w8a8`)

- LLM merge export dir: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_merged_w8a8`
- LLM keep export dir: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_lora_w8a8`
- Audio export dir: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_audio_encoder_w8a8`

From `export_meta_info.json`:

- merge:
  - `valid_wrap_vs_hf.cosine_sim=0.999998927116394`
  - `valid_quant_vs_hf.cosine_sim=0.9953804612159729`
- keep:
  - `valid_wrap_vs_hf.cosine_sim=0.9999943971633911`
  - `valid_quant_vs_hf.cosine_sim=0.9933393001556396`

Demo (single utt) example output:

- `/tmp/fireredasr_hf_forward_hmonnx_w8a8_merge_single_dbg2.json`
- CER: `2.076923076923077` (`decode_max_len=8`)

## 3. Flow B: common quant first, then keep/merge export

- Common quant ckpt dirs:
  - merge: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora`
  - keep: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_keep_lora`
- Audio export dirs:
  - merge: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_audio_encoder_merge_common_quant`
  - keep: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_audio_encoder_keep_common_quant`
- LLM export dirs:
  - merge: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_merged`
  - keep: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_lora`

From `export_meta_info.json`:

- merge:
  - `valid_wrap_vs_hf.cosine_sim=0.9999973773956299`
  - `valid_quant_vs_hf.cosine_sim=0.9927502870559692`
  - `valid_asr.match_rate=0.0`
- keep:
  - `valid_wrap_vs_hf.cosine_sim=0.9999970197677612`
  - `valid_quant_vs_hf.cosine_sim=0.9405761957168579`

Demo (single utt) example output:

- merge llm-only: `/tmp/fireredasr_hf_forward_hmonnx_common_merge_single_llm_only.json`
  - CER: `1.5384615384615385` (`decode_max_len=8`)
- keep llm-only: `/tmp/fireredasr_hf_forward_hmonnx_common_keep_single_llm_only.json`
  - CER: `1.2307692307692308` (`decode_max_len=8`)

## 4. Current status

- Migration and both export chains are runnable.
- Stage validation data and artifacts are complete.
- End-to-end demo text consistency with HF is not yet closed (HMONNX decode still drifts).
