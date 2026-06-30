# GPTQModel env setup for transformer 5.5 CI

The transformer 5.5 CI lane depends on the GPTQModel Gerrit checkout because
Qwen3.5 and Gemma4 quantization call GPTQModel recipe APIs:

- `gptqmodel.recipes.qwen35.quantize_qwen35`
- `gptqmodel.recipes.qwen35_autoround.quantize_qwen35_autoround`
- `gptqmodel.recipes.gemma4.quantize_gemma4`

Do **not** install GPTQModel in the xh2modelzoo CI job. Point Python at the
Gerrit checkout with environment variables before running the CI lane.

## Checkout location

Use the GPTQModel checkout prepared by CI/IT. The path is environment-specific,
so this document intentionally uses a placeholder instead of an absolute
machine path.

## Environment setup

```bash
export GPTQMODEL_SOURCE_DIR=/path/to/gptqmodel
export PYTHONPATH="${GPTQMODEL_SOURCE_DIR}:${GPTQMODEL_SOURCE_DIR}/third_party/auto-round:${PYTHONPATH:-}"
```

`GPTQMODEL_SOURCE_DIR` is read by
`benchmark_test/ci_test_transformer5.5.0/conftest.py`, which prepends both the
GPTQModel checkout root and its `third_party/auto-round` subtree to `sys.path`.
The explicit `PYTHONPATH` export is still recommended for non-pytest checks and
for early imports in CI wrappers.

The transformer5.5 CI image must already provide two dependency groups:

1. GPTQModel runtime dependencies. Installing GPTQModel as a package is not
   required, but its Python dependencies are still required when using the
   Gerrit checkout through `PYTHONPATH`. Use the checkout-owned requirements
   file during image preparation, for example:

   ```bash
   python -m pip install -r /path/to/gptqmodel/requirements.txt
   ```

   This provides recipe dependencies such as `torch`, `transformers`,
   `datasets`, `logbar`, `tokenicer`, `torchao`, and `kernels`.

2. `run.sh` installs xh2modelzoo-missing GPTQModel runtime dependencies
   before quantization starts:

   ```bash
   python -m pip install --disable-pip-version-check \
     qwen-vl-utils==0.0.14 \
     compressed-tensors==0.15.0.1 \
     'threadpoolctl>=3.6.0' \
     'device-smi>=0.5.2' \
     'hf_transfer>=0.1.9' \
     'huggingface_hub>=0.34.4' \
     'tokenicer>=0.0.8' \
     'logbar>=0.2.1' \
     'maturin>=1.9.4' \
     'pyarrow>=21.0' \
     'torchao>=0.14.1' \
     'kernels>=0.12.2' \
     'defuser>=0.0.6' \
     py-cpuinfo \
     tqdm \
     pydantic
   ```

   This list is the subset of `/path/to/gptqmodel/requirements.txt` that is
   not declared by xh2modelzoo, plus `qwen-vl-utils` and
   `compressed-tensors` for model import. Without these packages, CI can fail
   before quant/export with missing imports such as `qwen_vl_utils`,
   `compressed_tensors`, `logbar`, `cpuinfo`, or `tokenicer`.

   Do **not** install `compressed-tensors==0.14.0.1` through normal pip
   dependency resolution in the transformer5.5 image: its package metadata
   requires `transformers<5.0.0` and conflicts with this CI lane's
   transformer5.5 runtime. `compressed-tensors==0.15.0.1` keeps the
   top-level symbols used by `xh_model_zoo.xh_llm.models.base_model` and its
   package metadata only requires `transformers>=4.45.0`. Avoid `0.16.x` and
   newer here unless `base_model.py` is updated, because they remove the
   `has_offloaded_params` top-level export used by the current code.

This lane does not compile or install GPTQModel CUDA extensions.

## Verify checkout imports

Run this in the same Python environment used by xh2modelzoo CI:

```bash
python - <<'PY'
import gptqmodel
import auto_round
import qwen_vl_utils
import compressed_tensors
from compressed_tensors import has_offloaded_params
import tqdm
import pydantic
import tokenicer
import logbar
import cpuinfo
import threadpoolctl
import device_smi
import hf_transfer
import huggingface_hub
import pyarrow
import torchao
import kernels
import defuser
from gptqmodel.recipes.gemma4 import quantize_gemma4
from gptqmodel.recipes.qwen35 import quantize_qwen35
from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

print("gptqmodel", getattr(gptqmodel, "__version__", "unknown"), gptqmodel.__file__)
print("auto_round", auto_round.__file__)
print("qwen_vl_utils", qwen_vl_utils.__file__)
print("compressed_tensors", compressed_tensors.__version__)
print("GPTQModel recipe imports OK")
PY
```

Expected result: the command exits with code 0 and prints
`GPTQModel recipe imports OK`. The `gptqmodel` and `auto_round` paths should
point under `${GPTQMODEL_SOURCE_DIR}`; `qwen_vl_utils` and
`compressed_tensors` should come from the CI Python environment.

## Run xh2modelzoo CI

```bash
cd /path/to/xh2modelzoo
./benchmark_test/ci_test_transformer5.5.0/run.sh
```

For a quick setup check without running quant/export:

```bash
./benchmark_test/ci_test_transformer5.5.0/run.sh --collect-only -q
```
