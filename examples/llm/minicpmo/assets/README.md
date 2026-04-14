# MiniCPM-o example assets

This directory stores tokenizer files, media samples, and local example inputs
used by `examples/llm/minicpmo`.

## Compliance guidance

- Do **not** assume every binary asset in this directory is automatically
  covered for arbitrary redistribution just because the surrounding example
  code is Apache-2.0.
- Upstream tokenizer files, checkpoints, downloaded media, and sample audio or
  video should be reviewed for provenance before external release.
- If additional asset sources are identified later, record them in a local
  provenance note and, if needed, in the repository-level
  `THIRD_PARTY_NOTICES`.
- The current asset-by-asset audit register lives in `PROVENANCE.md`.

## Practical rule

Local code changes can usually be relicensed or annotated under the
repository's normal process, but third-party data and media assets should be
treated as separate compliance objects.
