# Calibration data

AutoRound quantization for Qwen3.5/Qwen3.6 and Gemma4 reads the local
`NeelNanda/pile-10k` JSONL from this path:

```text
data/calib_data/NeelNanda-pile-10k.jsonl
```

The JSONL is intentionally committed because customer and CI environments may
not have public network access. CI should use this checked-in file directly and
should not download calibration data at runtime.
