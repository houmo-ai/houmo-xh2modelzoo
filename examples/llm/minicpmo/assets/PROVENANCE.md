# MiniCPM-o asset provenance register

Purpose: provide a strict audit surface for binary assets bundled under
`examples/llm/minicpmo/assets/`.

## Interpretation

- `verified` means the source and license are known and recorded.
- `unverified` means the file must **not** be assumed safe for external
  redistribution until provenance is confirmed.

## Current release gate

Until the `source`, `license/terms`, and `redistribution status` columns are
filled in with verified values, treat the listed assets as:

- internal/example-use only
- blocked for external redistribution

## Asset register

| Path | Asset type | Source | License / terms | Redistribution status | Current status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `assets/Skiing.mp4` | video sample | unverified | unverified | blocked pending verification | unverified | likely demo media; confirm original source |
| `assets/demo.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm whether generated locally or sourced externally |
| `assets/mimick.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm source and permission |
| `assets/qa.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm source and permission |
| `assets/radar.jpg` | image sample | unverified | unverified | blocked pending verification | unverified | confirm source and permission |
| `assets/chattts_tokenizer/special_tokens_map.json` | tokenizer metadata | likely upstream tokenizer artifact | unverified | blocked pending verification | unverified | likely tied to upstream tokenizer package |
| `assets/chattts_tokenizer/tokenizer.json` | tokenizer artifact | likely upstream tokenizer artifact | unverified | blocked pending verification | unverified | verify exact package/repo and license |
| `assets/chattts_tokenizer/tokenizer_config.json` | tokenizer metadata | likely upstream tokenizer artifact | unverified | blocked pending verification | unverified | verify exact package/repo and license |
| `assets/input_examples/Trump_WEF_2018_10s.mp3` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm clip source and rights |
| `assets/input_examples/assistant_default_female_voice.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm whether synthetic and reproducible |
| `assets/input_examples/assistant_male_voice.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm whether synthetic and reproducible |
| `assets/input_examples/audio_understanding.mp3` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm source and rights |
| `assets/input_examples/chi-english-1.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm source and rights |
| `assets/input_examples/cxk_original.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm source and rights |
| `assets/input_examples/exciting-emotion.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm source and rights |
| `assets/input_examples/fast-pace.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm source and rights |
| `assets/input_examples/icl_20.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm source and rights |
| `assets/input_examples/indian-accent.wav` | audio sample | unverified | unverified | blocked pending verification | unverified | confirm source and rights |

## Completion checklist

- [ ] every asset source URL or generation path is recorded
- [ ] every asset license / terms field is filled
- [ ] every asset redistribution status is approved
- [ ] any non-redistributable asset is removed or replaced before release
