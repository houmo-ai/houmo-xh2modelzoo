# MiniCPM-o model integration notes

This directory contains the local MiniCPM-o integration used by
`xh_model_zoo` for wrapping, conversion, and HMONNX export flows.

## Code license

- The local integration code in this directory is intended to be distributed
  under **Apache-2.0**.
- Several implementation files adapt or wrap upstream MiniCPM / MiniCPM-V
  model behavior and therefore carry additional provenance notes in their file
  headers.

## Upstream relationship

- Upstream project: <https://github.com/OpenBMB/MiniCPM-V>
- Repository notice entry: see `THIRD_PARTY_NOTICES` under
  `MiniCPM-O-2.6 / MiniCPM Series`

Important distinction:

- local wrapper / converter / compatibility code can be reviewed as repository
  source code;
- upstream checkpoints, tokenizer files, media samples, and other bundled
  assets remain separate compliance objects and may have additional
  redistribution requirements.

## Reviewer notes

The following files are the primary upstream-adaptation surfaces and should
retain provenance comments when modified:

- `minicpmo_hf_compatible.py`
- `_audio_model_impl.py`
- `_llm_model_impl.py`
- `_vision_model_impl.py`
- `_tts_model_impl.py`
- `_tts_dvae_model_impl.py`
- `_tts_vocos_model_impl.py`

## Remaining cleanup items

These items are not changed automatically here because they may affect imports
or release packaging, but they should be reviewed before an external release:

- `minicpmo_tts_convert copy.py` — suspicious duplicate/backup-style file name
- ` utils.py` — path appears to contain a leading space in the file name
- `__pycache__/` artifacts — should not be part of source release payloads
