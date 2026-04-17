import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/data01/home/xuchen/xh2/xh2_model_zoo/examples/cv/superglue')
sys.path.insert(0, '/data01/home/xuchen/xh2/xh2_model_zoo/examples/cv/superpoint/fp')

class SuperPointOri(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

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
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        print('SuperPointOri initialized (random weights)')

    def forward(self, data):
        x = self.relu(self.conv1a(data['image']))
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
        scores = F.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)

        def simple_nms(scores, nms_radius):
            def max_pool(x):
                return F.max_pool2d(x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)
            zeros = torch.zeros_like(scores)
            max_mask = scores == max_pool(scores)
            for _ in range(2):
                supp_mask = max_pool(max_mask.float()) > 0
                supp_scores = torch.where(supp_mask, zeros, scores)
                new_max_mask = supp_scores == max_pool(supp_scores)
                max_mask = max_mask | (new_max_mask & (~supp_mask))
            return torch.where(max_mask, scores, zeros)

        scores = simple_nms(scores, self.config['nms_radius'])

        keypoints = [torch.nonzero(s > self.config['keypoint_threshold']) for s in scores]
        scores_list = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        def remove_borders(kpts, sc, border, height, width):
            keep = (kpts[:, 0] >= border) & (kpts[:, 0] < width - border) & \
                   (kpts[:, 1] >= border) & (kpts[:, 1] < height - border)
            return kpts[keep], sc[keep]

        keypoints, scores_list = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores_list)]))

        def top_k_keypoints(kpts, sc, k):
            if k >= len(kpts) or k == -1:
                return kpts, sc
            sc, indices = torch.topk(sc, k, dim=0)
            return kpts[indices], sc

        if self.config['max_keypoints'] >= 0:
            keypoints, scores_list = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores_list)]))

        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = F.normalize(descriptors, p=2, dim=1)

        def sample_descriptors_ori(kpts, desc, s=8):
            b, c, h, w = desc.shape
            kpts = kpts - s / 2 + 0.5
            kpts /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)]).to(kpts)[None]
            kpts = kpts * 2 - 1
            desc_out = F.grid_sample(desc, kpts.view(b, 1, -1, 2), mode='bilinear', align_corners=True)
            return F.normalize(desc_out.reshape(b, c, -1), p=2, dim=1)

        descriptors = [sample_descriptors_ori(k[None], d[None], 8)[0] for k, d in zip(keypoints, descriptors)]

        keypoints = list(keypoints)
        scores_list = list(scores_list)

        return {
            'keypoints': keypoints,
            'scores': scores_list,
            'descriptors': descriptors,
        }


from fp_superpoint import SuperPoint as SuperPointFP


def main():
    torch.manual_seed(42)

    img = torch.rand(1, 1, 480, 640)

    model_fp = SuperPointFP()
    model_fp.eval()

    model_ori = SuperPointOri({})
    model_ori.eval()

    with torch.no_grad():
        out_fp = model_fp({"image": img})
        out_ori = model_ori({"image": img})

    print("=" * 60)
    print("fp_superpoint.py output:")
    for k, v in out_fp.items():
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)} items")
            for i, t in enumerate(v):
                print(f"    [{i}]: {t.shape}")
        else:
            print(f"  {k}: {v.shape}")

    print("=" * 60)
    print("ori_sp.py output:")
    for k, v in out_ori.items():
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)} items")
            for i, t in enumerate(v):
                print(f"    [{i}]: {t.shape}")
        else:
            print(f"  {k}: {v.shape}")

    print("=" * 60)
    print("Key differences:")
    print("  1. fp uses 'keypoint_scores', ori uses 'scores'")
    print(f"  2. fp keypoints: {out_fp['keypoints'][0].shape[0]}, ori keypoints: {out_ori['keypoints'][0].shape[0]}")
    print("  3. Both return list of [Tensor(N, 2)], [Tensor(N,)], [Tensor(N, 256)]")
    print("=" * 60)


if __name__ == "__main__":
    main()