import torch
import sys
sys.path.insert(0, '/data01/home/xuchen/xh2/xh2_model_zoo/examples/cv/superglue')
import superglue as orig
from export_superglue_onnx import SuperGlueExport

config = {
    'descriptor_dim': 256,
    'weights': '/data01/home/xuchen/xh2/xh2_model_zoo/examples/cv/superglue/superglue_outdoor.pth',
    'keypoint_encoder': [32, 64, 128, 256],
    'GNN_layers': ['self', 'cross'] * 9,
    'sinkhorn_iterations': 100,
    'match_threshold': 0.2,
}

model_orig = orig.SuperGlue(config)
model_orig.eval()

model_new = SuperGlueExport(config)
model_new.load_pretrained('/data01/home/xuchen/xh2/xh2_model_zoo/examples/cv/superglue/superglue_outdoor.pth')
model_new.eval()

import cv2
import numpy as np

img1 = cv2.imread('/data02/datasets/hpatches/raw/hpatches_seq/hpatches/hpatches-sequences-release/i_pool/1.ppm', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/data02/datasets/hpatches/raw/hpatches_seq/hpatches/hpatches-sequences-release/i_pool/2.ppm', cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

img1_t = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0) / 255.0
img2_t = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0) / 255.0

kpts0 = torch.randn(1, 800, 2)
kpts1 = torch.randn(1, 800, 2)
scores0 = torch.rand(1, 800)
scores1 = torch.rand(1, 800)
desc0 = torch.randn(1, 256, 800)
desc1 = torch.randn(1, 256, 800)

with torch.no_grad():
    data = {
        'image0': img1_t, 'image1': img2_t,
        'keypoints0': kpts0, 'keypoints1': kpts1,
        'scores0': scores0, 'scores1': scores1,
        'descriptors0': desc0, 'descriptors1': desc1,
    }
    out_orig = model_orig(data)
    m0_o = out_orig['matches0']
    m1_o = out_orig['matches1']
    ms0_o = out_orig['matching_scores0']
    ms1_o = out_orig['matching_scores1']

    m0_n, m1_n, ms0_n, ms1_n = model_new(img1_t, img2_t, kpts0, scores0, desc0, kpts1, scores1, desc1)

print("=== matches0 ===")
# print(f"  diff: {(m0_o - m0_n).abs().max().item():.2e}")
print(f"  orig: {m0_o[0, :8].tolist()}")
print(f"  new:  {m0_n[0, :8].tolist()}")

print("=== matches1 ===")
# print(f"  diff: {(m1_o - m1_n).abs().max().item():.2e}")
print(f"  orig: {m1_o[0, :8].tolist()}")
print(f"  new:  {m1_n[0, :8].tolist()}")

print("=== mscores0 ===")
valid = ~(torch.isinf(ms0_o) | torch.isinf(ms0_n))
# diff = (ms0_o[valid] - ms0_n[valid]).abs().max().item() if valid.any() else 0
# print(f"  diff (no inf): {diff:.2e}")
print(f"  orig: {ms0_o[0, :8].tolist()}")
print(f"  new:  {ms0_n[0, :8].tolist()}")
# print(f"  orig has inf: {torch.isinf(ms0_o).any().item()}")
# print(f"  new has inf: {torch.isinf(ms0_n).any().item()}")

print("=== mscores1 ===")
valid = ~(torch.isinf(ms1_o) | torch.isinf(ms1_n))
diff = (ms1_o[valid] - ms1_n[valid]).abs().max().item() if valid.any() else 0
print(f"  diff (no inf): {diff:.2e}")
print(f"  orig has inf: {torch.isinf(ms1_o).any().item()}")
print(f"  new has inf: {torch.isinf(ms1_n).any().item()}")

max_diff = max(
    (m0_o - m0_n).abs().max().item(),
    (m1_o - m1_n).abs().max().item(),
)
print(f"\nmatches max diff: {max_diff:.2e}")
print("PASS" if max_diff < 1e-4 else "FAIL")
