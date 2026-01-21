# DEIM: DETR with Improved Matching for Fast Convergence

## 拉取源码

```bash
bash examples/cv/deim/download_source.sh 
```

## 下载权重(<https://drive.google.com/file/d/18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8/view?usp=drive_link>)

```text
Save to data/models/deim/deim_dfine_hgnetv2_m_coco_90e.pth
```

## 导出HMONNX模型

```bash
python examples/cv/deim/export_hmonnx.py -r data/models/deim/deim_dfine_hgnetv2_m_coco_90e.pth
```
