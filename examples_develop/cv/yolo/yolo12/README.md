# yolo12 ONNX Export Example (Removed)

The original Python source files in this directory were derived from
[`ultralytics`](https://github.com/ultralytics/ultralytics), which is licensed under
**AGPL-3.0-only**. Because that license is incompatible with the redistribution
terms of `xh2modelzoo`, those files have been removed.

## How to use yolo12 with our quantization toolchain

If you accept the AGPL-3.0 obligations of upstream Ultralytics, you can still
reproduce the example yourself:

```bash
pip install ultralytics      # AGPL-3.0 — accept the license terms
yolo export model=yolo12.pt format=onnx
```

Then point our quantization toolchain at the resulting ONNX file. Refer to the
upstream documentation for export options:
<https://docs.ultralytics.com/>.

> Note: this project does **not** redistribute Ultralytics code. Users who opt
> into AGPL-3.0 do so under their own responsibility.
