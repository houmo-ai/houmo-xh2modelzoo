# OCR 模型尺寸统一方案

## 问题分析

强制使用固定尺寸会影响精度，因为：
1. **文字变形**：强制 resize 会导致文字拉伸或压缩
2. **信息丢失**：过小的尺寸会丢失细节信息
3. **padding 过多**：过大的尺寸会产生大量无效 padding

## 解决方案

### 方案 1: 基于宽高比的动态调整（推荐）

根据 batch 中图片的宽高比动态选择合适的尺寸：

```python
# 宽高比范围 → 目标宽度
max_ratio <= 3:    → 320   (短文本)
max_ratio <= 6:    → 640   (中等长度)
max_ratio <= 10:   → 960   (长文本)
max_ratio > 10:     → 1280  (超长文本)
```

**优点：**
- ✅ 保持文字比例，不变形
- ✅ 适应不同长度的文本
- ✅ 减少 padding 浪费
- ✅ 相对固定的输入尺寸

**缺点：**
- ⚠️ 仍有 4 种不同的尺寸

### 方案 2: 分组处理

将图片按尺寸分组，每组使用不同的模型或尺寸：

```python
def group_images_by_ratio(images, thresholds=[3, 6, 10]):
    """根据宽高比分组"""
    groups = {0: [], 1: [], 2: [], 3: []}
    
    for img in images:
        h, w = img.shape[:2]
        ratio = w / h
        
        if ratio <= 3:
            groups[0].append(img)  # 短文本
        elif ratio <= 6:
            groups[1].append(img)  # 中等长度
        elif ratio <= 10:
            groups[2].append(img)  # 长文本
        else:
            groups[3].append(img)  # 超长文本
    
    return groups

# 处理每个组
for group_id, images in groups.items():
    if not images:
        continue
    
    target_width = [320, 640, 960, 1280][group_id]
    # 处理该组的图片
```

**优点：**
- ✅ 每组内部尺寸完全统一
- ✅ 可以针对每组优化模型
- ✅ 便于并行处理

**缺点：**
- ⚠️ 需要维护多个模型或配置
- ⚠️ 小组可能影响 batch 效率

### 方案 3: 自适应 Padding

使用动态 padding，但限制最大尺寸：

```python
def adaptive_padding(images, max_width=1280, step=64):
    """自适应 padding 到最近的 step 倍数"""
    
    # 计算每张图片的目标宽度
    target_widths = []
    for img in images:
        h, w = img.shape[:2]
        ratio = w / h
        target_w = int(np.ceil(48 * ratio / step) * step)
        target_w = min(target_w, max_width)
        target_widths.append(target_w)
    
    # 找到最大宽度
    max_width = max(target_widths)
    
    # Padding 到统一宽度
    padded_images = []
    for img, target_w in zip(images, target_widths):
        # Resize 到目标宽度
        resized = cv2.resize(img, (target_w, 48))
        # Padding 到最大宽度
        pad_width = max_width - target_w
        padded = np.pad(resized, ((0, 0), (0, pad_width)), mode='constant')
        padded_images.append(padded)
    
    return np.stack(padded_images, axis=0)
```

**优点：**
- ✅ 减少不必要的 padding
- ✅ 保持文字比例
- ✅ 相对灵活的尺寸选择

**缺点：**
- ⚠️ 仍有多种可能的尺寸
- ⚠️ 需要处理不同的尺寸组合

### 方案 4: 混合策略（最优）

结合多种策略的优点：

```python
class SmartBatchProcessor:
    def __init__(self, fixed_shapes=None, max_width=1280):
        """
        Args:
            fixed_shapes: 固定尺寸列表，如 [[3, 48, 320], [3, 48, 640], [3, 48, 960]]
            max_width: 最大宽度限制
        """
        self.fixed_shapes = fixed_shapes or [[3, 48, 320], [3, 48, 640], [3, 48, 960]]
        self.max_width = max_width
    
    def process_batch(self, images):
        """智能处理 batch"""
        # 1. 计算宽高比
        ratios = [img.shape[1] / img.shape[0] for img in images]
        max_ratio = max(ratios)
        
        # 2. 选择最合适的固定尺寸
        target_shape = self._select_shape(max_ratio)
        
        # 3. 处理图片
        processed_images = []
        for img in images:
            processed = self._process_single(img, target_shape)
            processed_images.append(processed)
        
        # 4. Stack 成 batch
        return np.stack(processed_images, axis=0)
    
    def _select_shape(self, ratio):
        """根据宽高比选择最合适的尺寸"""
        if ratio <= 3:
            return self.fixed_shapes[0]  # [3, 48, 320]
        elif ratio <= 6:
            return self.fixed_shapes[1]  # [3, 48, 640]
        elif ratio <= 10:
            return self.fixed_shapes[2]  # [3, 48, 960]
        else:
            # 超长文本：计算合适的宽度
            target_w = min(int(np.ceil(48 * ratio / 64) * 64), self.max_width)
            return [3, 48, target_w]
    
    def _process_single(self, img, target_shape):
        """处理单张图片"""
        c, h, w = target_shape
        
        # Resize 到目标高度，保持宽高比
        orig_h, orig_w = img.shape[:2]
        ratio = orig_w / orig_h
        target_w_keep_ratio = int(np.ceil(h * ratio))
        
        # 限制到目标宽度
        if target_w_keep_ratio > w:
            target_w_keep_ratio = w
        
        # Resize
        resized = cv2.resize(img, (target_w_keep_ratio, h))
        
        # Padding 到目标宽度
        pad_width = w - target_w_keep_ratio
        padded = np.pad(resized, ((0, 0), (0, pad_width)), mode='constant')
        
        # 转换为 CHW 格式并归一化
        padded = padded.transpose((2, 0, 1)) / 255.0
        padded = (padded - 0.5) / 0.5
        
        return padded
```

## 推荐使用方式

### 1. 对于文本识别（Cus_TextRecPredictor）

```python
# 使用方案 1：基于宽高比的动态调整
rec_predictor = Cus_TextRecPredictor.to_hf_compatible(
    hf_model=base_model,
    rec_model=your_rec_model,
    fixed_shape=None  # 不使用固定尺寸，使用动态调整
)
```

### 2. 对于文本检测（Cus_TextDetPredictor）

```python
# 使用固定尺寸（检测模型通常对尺寸不敏感）
det_predictor = Cus_TextDetPredictor.to_hf_compatible(
    hf_model=base_model,
    det_model=your_det_model,
    fixed_shape=[3, 512, 896]  # 固定尺寸
)
```

### 3. 完全固定尺寸（如果需要）

```python
# 使用固定尺寸，但选择较大的尺寸以减少精度损失
rec_predictor = Cus_TextRecPredictor.to_hf_compatible(
    hf_model=base_model,
    rec_model=your_rec_model,
    fixed_shape=[3, 48, 960]  # 使用较大的固定尺寸
)
```

## 性能对比

| 方案 | 精度 | 速度 | 内存 | 复杂度 |
|------|------|------|------|--------|
| 完全固定尺寸 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| 基于宽高比动态调整 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 分组处理 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 自适应 Padding | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 混合策略 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 总结

- **追求最高精度**：使用方案 4（混合策略）
- **平衡精度和速度**：使用方案 1（基于宽高比动态调整）
- **追求最快速度**：使用完全固定尺寸，但选择较大的尺寸
- **处理超长文本**：使用方案 2（分组处理）或方案 3（自适应 Padding）
