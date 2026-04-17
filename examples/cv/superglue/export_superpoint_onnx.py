import torch
import torch.nn as nn
import onnx
import onnxsim

def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class SuperPointDense(nn.Module):
    """SuperPoint for ONNX export - outputs dense scores and descriptors"""

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {
                'descriptor_dim': 256,
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'remove_borders': 4,
            }
        self.config = config

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, config['descriptor_dim'], kernel_size=1, stride=1, padding=0)

        self.nms_radius = config['nms_radius']

    def load_pretrained(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location='cpu'))
        print(f"Loaded weights from {weight_path}")

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        # scores = simple_nms(scores, self.nms_radius)

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)

        return scores, descriptors


def export_superpoint_onnx(
    weight_path: str,
    output_path: str = "superpoint_dense.onnx",
    input_shape: tuple = (1, 1, 480, 640),
    simplify: bool = True,
):
    device = torch.device('cpu')

    model = SuperPointDense()
    model.load_pretrained(weight_path)
    model.eval()
    model.to(device)

    dummy_input = torch.randn(input_shape, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['scores', 'descriptors'],
        # dynamic_axes={
        #     'input': {0: 'batch', 2: 'height', 3: 'width'},
        #     'scores': {0: 'batch', 1: 'score_h', 2: 'score_w'},
        #     'descriptors': {0: 'batch', 1: 'desc_dim', 2: 'desc_h', 3: 'desc_w'},
        # },
    )
    print(f"Exported ONNX to {output_path}")

    if simplify:
        print("Running onnxsim optimization...")
        model_onnx = onnx.load(output_path)
        model_simp, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simp, output_path)
            print(f"Simplified ONNX saved to {output_path}")
        else:
            print("Simplification check failed, keeping original ONNX.")

    return output_path


if __name__ == "__main__":
    weight_path = "superpoint_v1.pth"
    output_path = "superpoint_dense.onnx"
    input_shape = (1, 1, 480, 640)

    export_superpoint_onnx(
        weight_path=weight_path,
        output_path=output_path,
        input_shape=input_shape,
        simplify=True,
    )