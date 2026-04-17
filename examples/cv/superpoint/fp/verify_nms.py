import torch
import torch.nn.functional as F

def batched_nms_original(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def batched_nms_replaced(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = (scores == max_pool(scores)).float()
    for _ in range(2):
        supp_mask = (max_pool(max_mask) > 0).float()
        supp_scores = supp_mask * zeros + (1 - supp_mask) * scores
        new_max_mask = (supp_scores == max_pool(supp_scores)).float()
        not_supp_mask = 1 - supp_mask
        max_mask = max_mask + new_max_mask * not_supp_mask - max_mask * new_max_mask * not_supp_mask
    return max_mask * scores + (1 - max_mask) * zeros


torch.manual_seed(42)
scores = torch.rand(1, 1, 100, 100)
nms_radius = 4

result_original = batched_nms_original(scores.clone(), nms_radius)
result_replaced = batched_nms_replaced(scores.clone(), nms_radius)

print(f"Max absolute difference: {(result_original - result_replaced).abs().max().item()}")
print(f"Results are close: {torch.allclose(result_original, result_replaced)}")
print(f"Results are identical: {(result_original == result_replaced).all()}")