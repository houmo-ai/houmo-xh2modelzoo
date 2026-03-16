#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT分类模型推理API服务
使用FastAPI构建RESTful API
支持NPU、GPU、CPU多设备推理
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pickle
import os
import logging
import re
from typing import Dict, List, Optional
from transformers import BertTokenizer, BertForSequenceClassification
import uvicorn

# 尝试导入NPU支持
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BERT分类模型推理API",
    description="基于BERT的文本分类推理服务，支持NPU/GPU/CPU多设备",
    version="1.0.0"
)

# 请求模型
class TextRequest(BaseModel):
    text: str
    max_length: Optional[int] = 512

class BatchTextRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 512

# 响应模型
class PredictionResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    device_type: str
    device_count: int

# 全局变量
model = None
tokenizer = None
label_encoder = None
device = None
device_type = None
page_pattern = None  # 页码正则表达式

def detect_and_set_device():
    """检测并设置设备"""
    global device, device_type
    
    # 从环境变量获取设备号，默认为0
    device_id = int(os.getenv("DEVICE", "0"))
    
    # 根据torch_npu导入情况和CUDA可用性选择设备
    if NPU_AVAILABLE:
        device = torch.device(f"npu:{device_id}")
        device_type = "NPU"
        logger.info(f"使用NPU设备: {device}")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        device_type = "GPU"
        logger.info(f"使用GPU设备: {device}")
    else:
        device = torch.device("cpu")
        device_type = "CPU"
        logger.info(f"使用CPU设备: {device}")
    
    return device, device_type

def get_device_count():
    """获取当前设备类型的设备数量"""
    if device_type == "NPU" and NPU_AVAILABLE:
        return torch.npu.device_count()
    elif device_type == "GPU":
        return torch.cuda.device_count()
    else:
        return 1  # CPU

def load_model():
    """加载模型和相关组件"""
    global model, tokenizer, label_encoder, device, device_type, page_pattern
    
    try:
        # 初始化页码正则表达式
        page_pattern = re.compile(r'cur_page:\d+', re.IGNORECASE)
        
        # 检测并设置设备
        device, device_type = detect_and_set_device()
        
        # 模型路径
        model_path = "/data02/users/cc_work/model312/BERT"
        label_encoder_path = "/data02/users/cc_work/model312/BERT/label_encoder.pkl"
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"标签编码器路径不存在: {label_encoder_path}")
        
        # 加载模型和tokenizer
        logger.info("正在加载BERT模型...")
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # 移动到设备并设置为评估模式
        model.to(device)
        model.eval()
        
        tokenizer = BertTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # 加载标签编码器
        logger.info("正在加载标签编码器...")
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info(f"模型加载完成! 设备: {device} ({device_type})")
        return True
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return False

def process_text_content(content: str) -> str:
    """处理文本内容：移除页码字符串，移除所有空白字符，截取前100个字符"""
    global page_pattern
    # 移除页码字符串
    content = page_pattern.sub('', content)
    # 移除所有空白字符（换行符、空格、制表符等）
    content = re.sub(r'\s+', '', content)
    # 只截取前100个字符
    content = content[:100]
    return content

def predict_single_text(text: str, max_length: int = 512) -> Dict:
    """对单个文本进行预测"""
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 预处理文本内容
        text = process_text_content(text)
        
        # 文本预处理
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 移动到设备
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        # 预测
        with torch.no_grad():
            outputs = model(**encoding)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 获取所有类别的概率
        all_probs = probabilities[0].cpu().numpy()
        prob_dict = {}
        for i, prob in enumerate(all_probs):
            label = label_encoder.inverse_transform([i])[0]
            prob_dict[label] = float(prob)
        
        # 解码预测标签
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "probabilities": prob_dict
        }
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    logger.info("正在启动BERT推理服务...")
    success = load_model()
    if not success:
        logger.error("模型加载失败，服务可能无法正常工作")

@app.get("/", response_model=HealthResponse)
async def root():
    """健康检查接口"""
    return HealthResponse(
        status="running",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        device_type=device_type if device_type else "unknown",
        device_count=get_device_count()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """详细健康检查"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        device_type=device_type if device_type else "unknown",
        device_count=get_device_count()
    )

@app.get("/device/info")
async def get_device_info():
    """获取设备详细信息"""
    info = {
        "current_device": str(device) if device else "unknown",
        "device_type": device_type if device_type else "unknown",
        "npu_available": NPU_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "environment_device": os.getenv("DEVICE", "0")
    }
    
    return info

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(request: TextRequest):
    """单文本预测接口"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    result = predict_single_text(request.text, request.max_length)
    return PredictionResponse(**result)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchTextRequest):
    """批量文本预测接口"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="文本列表不能为空")
    
    if len(request.texts) > 100:  # 限制批量大小
        raise HTTPException(status_code=400, detail="批量预测最多支持100个文本")
    
    predictions = []
    for text in request.texts:
        if text.strip():  # 跳过空文本
            result = predict_single_text(text, request.max_length)
            predictions.append(PredictionResponse(**result))
    
    return BatchPredictionResponse(predictions=predictions)

@app.get("/labels")
async def get_labels():
    """获取所有可能的标签"""
    if label_encoder is None:
        raise HTTPException(status_code=500, detail="标签编码器未加载")
    
    try:
        labels = label_encoder.classes_.tolist()
        return {"labels": labels, "count": len(labels)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取标签失败: {str(e)}")

@app.post("/reload")
async def reload_model():
    """重新加载模型"""
    logger.info("正在重新加载模型...")
    success = load_model()
    if success:
        return {"message": "模型重新加载成功"}
    else:
        raise HTTPException(status_code=500, detail="模型重新加载失败")

if __name__ == "__main__":
    # 从环境变量获取 workers 数量，默认为 1
    workers = int(os.getenv("WORKERS", "1"))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=workers
    )