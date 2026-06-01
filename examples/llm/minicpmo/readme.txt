MiniCPM-o example notes
=======================

Environment
-----------

- Recommended dependency: `transformers==4.44.2`

Export commands
---------------

1. 导出 audio 部分：
   `python examples/llm/minicpmo/minicpmo_audio_export_hmonnx.py`
   输出目录：
   `work_dirs/MiniCPM-o-2_6-XH2a-audio-w8a8h0_sefp/hmonnx/audio`

2. 导出 vision 部分：
   `python examples/llm/minicpmo/minicpmo_vision_export_hmonnx.py`
   输出目录：
   `work_dirs/MiniCPM-o-2_6-XH2a-vision-w8a8h0_sefp/hmonnx/vision`

3. 导出 llm 部分：
   `python examples/llm/minicpmo/minicpmo_llm_export_hmonnx.py`
   输出目录：
   `work_dirs/MiniCPM-o-2_6-XH2a-llm-w8a8h0_sefp/hmonnx/llm`

4. 导出 tts 部分：
   `python examples/llm/minicpmo/minicpmo_tts_export_hmonnx.py`
   输出目录：
   `work_dirs/MiniCPM-o-2_6-XH2a-tts-w8a8h0_sefp/hmonnx/tts`

5. 导出 tts-dvae 部分：
   `python examples/llm/minicpmo/minicpmo_tts_dvae_export_hmonnx.py`
   输出目录：
   `work_dirs/MiniCPM-o-2_6-XH2a-tts-dvae-w8a8h0_sefp/hmonnx/tts-dvae`

6. 导出 tts-vocos 部分：
   `python examples/llm/minicpmo/minicpmo_tts_vocos_export_hmonnx.py`
   输出目录：
   `work_dirs/MiniCPM-o-2_6-XH2a-tts-vocos-w8a8h0_sefp/hmonnx/tts-vocos`

License and redistribution notes
--------------------------------

- The local example scripts in this directory are intended to be distributed
  under the repository's Apache-2.0 licensing model.
- `patch/modeling_minicpmo.py` contains a local patch derived from the
  upstream OpenBMB MiniCPM / MiniCPM-V project. Its file header preserves
  Apache-2.0 licensing and adaptation notice information.
- The upstream MiniCPM-O-2.6 / MiniCPM series referenced by these examples is
  listed in the repository `THIRD_PARTY_NOTICES` as Apache License 2.0.
- Model checkpoints, tokenizer assets, downloaded files, and bundled media
  samples may have additional attribution or redistribution requirements.
  Before publishing this directory outside the repository, verify the
  provenance of files under `assets/` and any downloaded model weights.
- Asset-by-asset verification should be tracked in
  `examples/llm/minicpmo/assets/PROVENANCE.md`.
