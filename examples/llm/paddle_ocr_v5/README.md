# Paddle OCR V5 示例

本目录包含基于 PaddlePaddle 的 OCR 模型使用示例，包括模型量化、自定义推理和基本测试。

## 环境要求

安装方法：
```
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/  
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install triton==3.4.0                                                                               
pip install onnx==1.19.1
安装 xhquant
```

### 基础环境
- Python 3.12
- CUDA 11.8
- PaddlePaddle 3.3.0
- PyTorch 2.7.0+cu118


### 主要依赖包
- paddlepaddle-gpu==3.3.0
- paddleocr
- torch==2.7.0+cu118
- torchvision==0.22.0+cu118
- torchaudio==2.7.0+cu118
- onnxruntime-gpu
- xhquant (自定义量化库)

## 文件说明

### 1. test.py
**功能**: Paddle GPU 环境测试脚本

**用途**: 验证 PaddlePaddle 的 CUDA 环境是否正常工作

**使用方法**:
```bash
python test.py
```

**输出**: 
- CUDA 可用性检查
- GPU 张量创建和运算测试

### 2. hm_export.py
**功能**: PaddleOCR ONNX 模型量化导出

**主要特性**:
- 将 PaddleOCR 的文本检测和识别模型转换为量化格式
- 支持多种量化方案 (w8a8, w8a8h1_sefp 等)
- 生成用于 XH2a 设备的量化模型
- 支持模型精度验证

**使用方法**:
```bash
python hm_export.py \
    --onnx_model_path data/models/ocr_onnx \
    --quant-type w8a8h1_sefp \
    --output_path work_dirs/paddle_312 \
    --device cuda
```

**参数说明**:
- `--onnx_model_path`: ONNX 模型目录路径（包含 det.onnx 和 rec.onnx）
- `--quant-type`: 量化类型，默认为 w8a8h1_sefp
- `--output_path`: 输出目录，默认为 work_dirs/paddle_312
- `--device`: 设备类型，默认为 cuda

**输出文件**:
- `hmquant_xh2_paddleocr_det.onnx`: 量化后的文本检测模型
- `hmquant_xh2_paddleocr_rec.onnx`: 量化后的文本识别模型
- `hmonnx/golden/`: 金标准数据目录

### 3. fp_demo.py
**功能**: 使用自定义 OCR 模型进行推理演示

**主要特性**:
- 使用 PaddleOCR 进行标准 OCR 推理
- 集成自定义的文本检测和识别模型
- 支持量化模型推理
- 结果可视化和 JSON 导出

**使用方法**:
```bash
python fp_demo.py
```

**工作流程**:
1. 初始化 PaddleOCR 实例
2. 运行标准 OCR 推理（使用原始模型）
3. 加载量化模型
4. 替换为自定义的文本检测和识别模型
5. 运行自定义 OCR 推理（使用量化模型）
6. 保存和可视化结果

**自定义模型**:
- `Cus_TextDetPredictor`: 自定义文本检测模型
- `Cus_TextRecPredictor`: 自定义文本识别模型

**输出**:
- `output/`: 原始模型推理结果
- `output_hm/`: 量化模型推理结果
- JSON 格式的文本识别结果

## 模型文件

### 输入模型
- `data/models/ocr_onnx/det.onnx`: 文本检测 ONNX 模型
- `data/models/ocr_onnx/rec.onnx`: 文本识别 ONNX 模型
- `data/models/ocr_onnx/general_ocr_002.png`: 测试图片

### 输出模型
- `work_dirs/paddle_312/hmquant_xh2_paddleocr_det.onnx`: 量化检测模型
- `work_dirs/paddle_312/hmquant_xh2_paddleocr_rec.onnx`: 量化识别模型

## 完整使用流程

### 1. 环境测试
```bash
conda activate paddle
python test.py
```

### 2. 模型量化
```bash
python hm_export.py \
    --onnx_model_path data/models/ocr_onnx \
    --quant-type w8a8h1_sefp \
    --output_path work_dirs/paddle_312 \
    --device cuda
```

### 3. 推理演示
```bash
python fp_demo.py
```

## 自定义模型说明

### Cus_TextDetPredictor
**继承自**: `TextDetPredictor`

**主要改进**:
- 支持量化模型推理
- 优化了内存使用
- 兼容 HMONNX 格式

**使用方法**:
```python
from xh_model_zoo.xh_llm.models.paddle_ocrv5.cus_textdet import Cus_TextDetPredictor

# 替换文本检测模型
ocr.paddlex_pipeline.text_det_model = Cus_TextDetPredictor.to_hf_compatible(
    ocr.paddlex_pipeline.text_det_model, det_model
)
```

### Cus_TextRecPredictor
**继承自**: `TextRecPredictor`

**主要改进**:
- 支持量化模型推理
- 动态尺寸调整（基于宽高比）
- 优化了长文本处理
- 兼容 HMONNX 格式

**使用方法**:
```python
from xh_model_zoo.xh_llm.models.paddle_ocrv5.cus_textrec import Cus_TextRecPredictor

# 替换文本识别模型
ocr.paddlex_pipeline.text_rec_model = Cus_TextRecPredictor.to_hf_compatible(
    ocr.paddlex_pipeline.text_rec_model, rec_model
)
```

## 尺寸调整方案

### 方案 1: 基于宽高比的动态调整（推荐）

**原理**: 根据图片的宽高比动态选择合适的输入尺寸，避免强制变形导致的精度损失。

**尺寸映射规则**:
```python
max_ratio <= 3:    → 320   (短文本)
max_ratio <= 6:    → 640   (中等长度)
max_ratio <= 10:   → 960   (长文本)
max_ratio > 10:     → 1280  (超长文本)
```

**优点**:
- ✅ 保持文字比例，不变形
- ✅ 适应不同长度的文本
- ✅ 减少 padding 浪费
- ✅ 相对固定的输入尺寸

**缺点**:
- ⚠️ 需要导出多个尺寸的模型
- ⚠️ 需要对文本检测模型做相应修改

### 方案 1 的实现步骤

#### 1. 导出多个尺寸的识别模型

需要为文本识别模型导出 4 种不同尺寸的量化模型：

```bash
# 尺寸 1: 320x48 (短文本)
python hm_export.py \
    --onnx_model_path data/models/ocr_onnx \
    --quant-type w8a8h1_sefp \
    --output_path work_dirs/paddle_320 \
    --device cuda \
    --rec_shape [6, 3, 48, 320]

# 尺寸 2: 640x48 (中等长度)
python hm_export.py \
    --onnx_model_path data/models/ocr_onnx \
    --quant-type w8a8h1_sefp \
    --output_path work_dirs/paddle_640 \
    --device cuda \
    --rec_shape [6, 3, 48, 640]

# 尺寸 3: 960x48 (长文本)
python hm_export.py \
    --onnx_model_path data/models/ocr_onnx \
    --quant-type w8a8h1_sefp \
    --output_path work_dirs/paddle_960 \
    --device cuda \
    --rec_shape [6, 3, 48, 960]

# 尺寸 4: 1280x48 (超长文本)
python hm_export.py \
    --onnx_model_path data/models/ocr_onnx \
    --quant-type w8a8h1_sefp \
    --output_path work_dirs/paddle_1280 \
    --device cuda \
    --rec_shape [6, 3, 48, 1280]
```

#### 2. 修改文本识别模型 (cus_textrec.py)

需要修改 `Cus_TextRecPredictor` 以支持多尺寸模型：

```python
class Cus_TextRecPredictor(TextRecPredictor):
    entities = []

    def __init__(self, *args, **kwargs):
        pass

    def __setup__(self, rec_models):
        """
        初始化多个尺寸的模型
        
        Args:
            rec_models: 字典，包含不同尺寸的模型
                {
                    320: model_320,
                    640: model_640,
                    960: model_960,
                    1280: model_1280
                }
        """
        self.rec_models = rec_models
        return self

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,
        rec_models = None,
        fixed_shape = None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        cls.fixed_shape = fixed_shape
        hf_model.__class__ = cls
        hf_model.__setup__(rec_models)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return hf_model

    def process(self, batch_data, return_word_box=False):
        batch_raw_imgs = self.pre_tfs["Read"](imgs=batch_data.instances)
        
        # 计算每张图片的宽高比
        width_list = []
        for img in batch_raw_imgs:
            width_list.append(img.shape[1] / float(img.shape[0]))
        
        # 根据宽高比选择目标宽度和对应模型
        if self.fixed_shape is not None:
            # 使用固定尺寸
            target_width = self.fixed_shape[2]
            self.rec_model = self.rec_models[target_width]
        else:
            # 根据宽高比动态调整尺寸
            max_ratio = max(width_list)
            if max_ratio <= 3:
                target_width = 320
            elif max_ratio <= 6:
                target_width = 640
            elif max_ratio <= 10:
                target_width = 960
            else:
                target_width = 1280
            
            # 选择对应尺寸的模型
            self.rec_model = self.rec_models[target_width]
        
        # 临时修改 rec_image_shape
        original_rec_image_shape = self.pre_tfs["ReisizeNorm"].rec_image_shape
        self.pre_tfs["ReisizeNorm"].rec_image_shape = [3, 48, target_width]
        
        indices = np.argsort(np.array(width_list))
        batch_imgs = self.pre_tfs["ReisizeNorm"](imgs=batch_raw_imgs)
        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)
        
        # 恢复原始配置
        self.pre_tfs["ReisizeNorm"].rec_image_shape = original_rec_image_shape
        
        # 使用对应尺寸的模型进行推理
        if self._use_static_model:
            if x[0].shape[-1] != 320:
                batch_preds = self.infer(x=x)
            else:
                inp = torch.from_numpy(x[0]).half().cuda()
                batch_preds = self.rec_model(inp)
                batch_preds = batch_preds.cpu().numpy().astype(np.float32)
                batch_preds = [batch_preds]
        else:
            with TemporaryDeviceChanger(self.device):
                batch_preds = self.infer(x=x)
        
        # 后续处理...
```



#### 4. 使用示例

```python
from paddleocr import PaddleOCR
from xhquant.api import HMONNXGoldenInference
from xh_model_zoo.xh_llm.models.paddle_ocrv5.cus_textrec import Cus_TextRecPredictor
from xh_model_zoo.xh_llm.models.paddle_ocrv5.cus_textdet import Cus_TextDetPredictor

# 初始化 OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu",
)

# 加载多个尺寸的识别模型
rec_models = {
    320: HMONNXGoldenInference("work_dirs/paddle_320/hmquant_xh2_paddleocr_rec.onnx"),
    640: HMONNXGoldenInference("work_dirs/paddle_640/hmquant_xh2_paddleocr_rec.onnx"),
    960: HMONNXGoldenInference("work_dirs/paddle_960/hmquant_xh2_paddleocr_rec.onnx"),
    1280: HMONNXGoldenInference("work_dirs/paddle_1280/hmquant_xh2_paddleocr_rec.onnx"),
}

# 设置设备
for model in rec_models.values():
    model.to("cuda:0")
    model.save_golden = False
    model.golden_dir = "work_dirs/hmonnx/golden"
    model.step = 0

# 加载检测模型
det_model = HMONNXGoldenInference("work_dirs/paddle_312/hmquant_xh2_paddleocr_det.onnx")
det_model.to("cuda:0")
det_model.save_golden = False
det_model.golden_dir = "work_dirs/hmonnx/golden"
det_model.step = 0

# 替换模型
ocr.paddlex_pipeline.text_det_model = Cus_TextDetPredictor.to_hf_compatible(
    ocr.paddlex_pipeline.text_det_model, det_model
)

ocr.paddlex_pipeline.text_rec_model = Cus_TextRecPredictor.to_hf_compatible(
    ocr.paddlex_pipeline.text_rec_model, rec_models
)

# 运行推理
result = ocr.predict(input="data/models/ocr_onnx/general_ocr_002.png")
```

### 方案 2: 固定尺寸（简单）

**原理**: 使用固定的输入尺寸，简单但可能影响精度。

**使用方法**:
```python
ocr.paddlex_pipeline.text_rec_model = Cus_TextRecPredictor.to_hf_compatible(
    ocr.paddlex_pipeline.text_rec_model, 
    rec_model,
    fixed_shape=[3, 48, 960]  # 使用较大的固定尺寸
)
```

**优点**:
- ✅ 实现简单
- ✅ 只需要一个模型
- ✅ 推理速度快

**缺点**:
- ⚠️ 可能影响精度
- ⚠️ 不适合所有长度的文本

### 方案对比

| 方案 | 精度 | 速度 | 内存 | 复杂度 |
|------|------|------|------|--------|
| 固定尺寸 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| 动态调整（方案1） | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

**推荐**: 对于精度要求较高的场景，使用方案 1（动态调整）。

## 相关资源

- [PaddlePaddle 官方文档](https://www.paddlepaddle.org.cn/)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [ONNX 文档](https://onnx.ai/)
- [量化技术说明](./docs/quantization.md)

## 许可证

本示例代码遵循 Apache 2.0 许可证。