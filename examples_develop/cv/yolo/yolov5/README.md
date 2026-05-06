# yolov5 ONNX Export Example (Removed)

The original Python source files in this directory were derived from
[`ailia-models`](https://github.com/axinc-ai/ailia-models) wrappers around
Ultralytics YOLOv5. The upstream YOLOv5 weights and inference scripts inherit
**AGPL-3.0-only** from Ultralytics, which is incompatible with the
redistribution terms of `xh2modelzoo`. Those files have been removed.

## How to use yolov5 with our quantization toolchain

If you accept the AGPL-3.0 obligations, install the upstream packages yourself:

```bash
pip install ultralytics yolov5    # AGPL-3.0 — accept the license terms
```

Then export YOLOv5 to ONNX following the upstream docs and feed the resulting
ONNX into our quantization toolchain.

> Note: this project does **not** redistribute Ultralytics / ailia-models
> derived code. Users who opt into AGPL-3.0 do so under their own
> responsibility.
