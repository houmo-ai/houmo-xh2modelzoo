import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
import numpy as np
import os
from pathlib import Path
import torch.nn as nn
import argparse
from xhquant.api import HMONNXGoldenInference


def preprocess_image(image_path, target_w, target_h, device):
    print(f"[-] Preprocessing Image: Target ({target_w}x{target_h})")
    
    original_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_image.size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    image_resized = original_image.resize((new_w, new_h), Image.BILINEAR)
    image_padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    image_padded.paste(image_resized, (0, 0))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image_padded).to(device)
    
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    img_tensor = img_tensor.half() if device == "cuda" else img_tensor.float()
    
    ratio_info = {
        "scale": scale,
        "orig_w": orig_w,
        "orig_h": orig_h
    }
    
    return img_tensor, original_image, ratio_info

def preprocess_text(prompt, max_len=256,device='cuda'):
    print(f"[-] Preprocessing Text: '{prompt}'")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    tokenized = tokenizer(
        prompt, 
        padding="max_length", 
        max_length=max_len, 
        truncation=True, 
        return_tensors="pt"
    )
    
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    token_type_ids = tokenized["token_type_ids"].to(device)
    position_ids = torch.arange(max_len, device=device).unsqueeze(0)
    text_token_mask = attention_mask.bool()

    if len(input_ids.shape) == 1: input_ids = input_ids.unsqueeze(0)
    if len(attention_mask.shape) == 1: attention_mask = attention_mask.unsqueeze(0)
    if len(position_ids.shape) == 1: position_ids = position_ids.unsqueeze(0)
    if len(token_type_ids.shape) == 1: token_type_ids = token_type_ids.unsqueeze(0)
    if len(text_token_mask.shape) == 1: text_token_mask = text_token_mask.unsqueeze(0)

    # 这里的 int32 转换视你的具体 runtime 要求而定
    return (
        input_ids.to(torch.int32), 
        attention_mask.to(torch.int32), 
        position_ids, 
        token_type_ids.to(torch.int32), 
        text_token_mask
    )


def post_process(pred_logits, pred_boxes, ratio_info, img_w_model, img_h_model, box_threshold):
    probs = torch.sigmoid(pred_logits)[0]
    boxes = pred_boxes[0]
    
    scores, _ = probs.max(dim=-1)
    
    keep = scores > box_threshold
    scores = scores[keep]
    boxes = boxes[keep]
    
    if len(scores) == 0:
        return [], []
    
    cx = boxes[:, 0] * img_w_model
    cy = boxes[:, 1] * img_h_model
    bw = boxes[:, 2] * img_w_model
    bh = boxes[:, 3] * img_h_model
    
    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh
    
    scale = ratio_info['scale']
    
    real_x1 = x1 / scale
    real_y1 = y1 / scale
    real_x2 = x2 / scale
    real_y2 = y2 / scale
    
    orig_w = ratio_info['orig_w']
    orig_h = ratio_info['orig_h']
    
    final_boxes = torch.stack([
        torch.clamp(real_x1, 0, orig_w),
        torch.clamp(real_y1, 0, orig_h),
        torch.clamp(real_x2, 0, orig_w),
        torch.clamp(real_y2, 0, orig_h)
    ], dim=1)
    
    return final_boxes, scores

def visualize(image_pil, boxes, scores):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{score:.2f}"
        text_w = draw.textlength(text, font=font)
        draw.rectangle([x1, y1, x1 + text_w, y1 + 25], fill="red", outline="red")
        draw.text((x1, y1), text, fill="white", font=font)
    
    return image_pil

# ==========================================
# 5. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # 默认路径指向你刚才导出的静态 ONNX
    parser.add_argument("--hmonnx", type=str, default="./work_dirs/groundingdino/hmonnx/groundingdino_XH2a.onnx")
    parser.add_argument("--image_path",type=str,default='data/images/qwen2_vl_demo.jpeg')
    parser.add_argument('--text_prompt',type=str,default='person.')
    parser.add_argument('--output_dir',type=str,default='results')
    parser.add_argument('--target_w',type=int,default=1200)
    parser.add_argument('--target_h',type=int,default=800)
    parser.add_argument('--text_len',type=int,default=256)
    parser.add_argument('--box_threshold',type=float,default=0.35)
    parser.add_argument('--device',type=str,default='cuda')

    args = parser.parse_args()

    print(f">>>  初始化 HMONNX Session: {args.hmonnx}")
    # 3. 初始化 HMONNX 工具
    session = HMONNXGoldenInference(args.hmonnx)
    session.to(args.device)
    
    session.save_golden = True
    session.golden_dir = Path(args.output_dir) / "golden_debug"
    session.step = 0
    
    print(">>>  数据预处理...")
    img_tensor, orig_img, ratio_info = preprocess_image(args.image_path, args.target_w, args.target_h, args.device)
    input_ids, attn_mask, pos_ids, token_type_ids, text_token_mask = preprocess_text(args.text_prompt, args.text_len, args.device)
    
    # 组装输入 (根据你导出的参数顺序)
    # 如果导出时只有 input_ids 和 mask，请注释掉其他的
    inference_inputs = [
        img_tensor,      # (1, 3, 800, 1200)
        input_ids,       # (1, 256)
        attn_mask,       # (1, 256)
        # pos_ids,         # (1, 256)
        token_type_ids,  # (1, 256)
        # text_token_mask  # (1, 256)
    ]
    
    print("\n[DEBUG] Input Shapes:")
    for i, t in enumerate(inference_inputs):
        print(f"  Input[{i}]: {t.shape}")

    print("\n>>> 执行推理 (Golden Inference)...")
    outputs = session(*inference_inputs)
    print(">>> 推理完成.")
        
    pred_logits = None
    pred_boxes = None

    if isinstance(outputs, dict):
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
    elif isinstance(outputs, (list, tuple)):
        pred_logits = outputs[0]
        pred_boxes = outputs[1]
    
    print(f"Output Shapes -> Logits: {pred_logits.shape}, Boxes: {pred_boxes.shape}")
    
    print(">>> 后处理与可视化...")
    final_boxes, final_scores = post_process(pred_logits, pred_boxes, ratio_info, args.target_w, args.target_h, args.box_threshold)
    
    print(f"\n检测到 {len(final_boxes)} 个目标:")
    if len(final_boxes) > 0:
        result_img = visualize(orig_img, final_boxes, final_scores)
        save_path = Path(args.output_dir) / "result_wrapper_fixed.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result_img.save(save_path)
        print(f"结果已保存至: {save_path}")
        for box, score in zip(final_boxes, final_scores):
            print(f"  - Score: {score:.4f}, Box: {[int(x) for x in box.tolist()]}")
    else:
        print("未检测到目标 (Try lowering BOX_THRESHOLD).")

if __name__ == "__main__":
    main()