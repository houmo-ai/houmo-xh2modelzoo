import torch
import torch.nn as nn
from typing import List
from copy import deepcopy


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    dim = query.shape[1]
    query_perm = query.permute(0, 2, 3, 1)
    key_perm = key.permute(0, 2, 1, 3)
    scores = torch.matmul(query_perm, key_perm) # / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    value_perm = value.permute(0, 2, 3, 1)
    return torch.matmul(prob, value_perm).permute(0, 3, 1, 2), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim: int, layers: List[int]):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


class SuperGlueExport(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = self.default_config
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def load_pretrained(self, weight_path: str):
        self.load_state_dict(torch.load(weight_path, map_location='cpu'))
        print(f'Loaded SuperGlue weights from {weight_path}')
        for layer in self.gnn.layers:
             d_model = 8
             inv_sqrt_d = 1/ d_model  # ** -0.5
            #  print(inv_sqrt_d)
             layer.attn.proj[0].weight.data *= inv_sqrt_d
             if layer.attn.proj[0].bias is not None:
                 layer.attn.proj[0].bias.data *= inv_sqrt_d

        # inv_sqrt_d = (1 / 16)** -0.5
        # self.final_proj.weight.data *= inv_sqrt_d
        # if self.final_proj.bias is not None:
        #     self.final_proj.bias.data *= inv_sqrt_d
            #  layer.attn.proj[1].weight.data *= inv_sqrt_d
            #  if layer.attn.proj[1].bias is not None:
            #      layer.attn.proj[1].bias.data *= inv_sqrt_d

    @staticmethod
    def normalize_keypoints(kpts, image_shape):
        _, _, height, width = image_shape
        one = kpts.new_tensor(1)
        size = torch.stack([one * width, one * height])[None]
        center = size / 2
        scaling = size.max(1, keepdim=True).values * 0.7
        return (kpts - center[:, None, :]) / scaling[:, None, :]

    @staticmethod
    def arange_like(x, dim):
        return x.new_ones(x.shape[dim]).cumsum(0) - 1

    def forward(self, image0, image1, keypoints0, scores0, descriptors0,
                keypoints1, scores1, descriptors1):
        desc0, desc1 = descriptors0, descriptors1
        kpts0, kpts1 = keypoints0, keypoints1

        kpts0 = self.normalize_keypoints(kpts0, image0.shape)
        kpts1 = self.normalize_keypoints(kpts1, image1.shape)

        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores1)

        desc0, desc1 = self.gnn(desc0, desc1)

        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.matmul(mdesc0.transpose(1, 2), mdesc1)  / self.config['descriptor_dim']**.5

        return scores

        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        bins0 = self.bin_score.expand(b, m, 1)
        bins1 = self.bin_score.expand(b, 1, n)
        alpha = self.bin_score.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        norm = -(ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
            u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
            for _ in range(iters):
                u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
                v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
            return Z + u.unsqueeze(2) + v.unsqueeze(1)

        scores = log_sinkhorn_iterations(couplings, log_mu, log_nu,
                                    self.config['sinkhorn_iterations'])
        scores = scores - norm

        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = self.arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = self.arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return indices0, indices1, mscores0, mscores1


def export_superglue_onnx(
    weight_path: str = "superglue_outdoor.pth",
    output_path: str = "superglue.onnx",
    max_keypoints: int = 800,
    descriptor_dim: int = 256,
    simplify: bool = True,
):
    try:
        import onnx
        import onnxsim
    except ImportError:
        print("ERROR: onnx or onnxsim not installed. Run: pip install onnx onnxsim")
        return None

    device = torch.device('cpu')

    config = {
        'descriptor_dim': descriptor_dim,
        'weights': weight_path,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    model = SuperGlueExport(config)
    model.load_pretrained(weight_path)
    model.eval()
    model.to(device)

    dummy_image0 = torch.randn(1, 1, 480, 640)
    dummy_image1 = torch.randn(1, 1, 480, 640)
    dummy_kpts0 = torch.randn(1, max_keypoints, 2)
    dummy_kpts1 = torch.randn(1, max_keypoints, 2)
    dummy_scores0 = torch.randn(1, max_keypoints)
    dummy_scores1 = torch.randn(1, max_keypoints)
    dummy_desc0 = torch.randn(1, descriptor_dim, max_keypoints)
    dummy_desc1 = torch.randn(1, descriptor_dim, max_keypoints)

    torch.onnx.export(
        model,
        (dummy_image0, dummy_image1, dummy_kpts0, dummy_scores0, dummy_desc0,
         dummy_kpts1, dummy_scores1, dummy_desc1),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['image0', 'image1', 'keypoints0', 'scores0', 'descriptors0',
                     'keypoints1', 'scores1', 'descriptors1'],
        output_names=['score'],
        # output_names=['matches0', 'matches1', 'matching_scores0', 'matching_scores1'],
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
    export_superglue_onnx(
        weight_path="superglue_outdoor.pth",
        output_path="superglue.onnx",
        max_keypoints=800,
        descriptor_dim=256,
        simplify=True,
    )