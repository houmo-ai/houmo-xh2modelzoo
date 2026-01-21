import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
import math
from .utils import *
from xhquant_llm.singlequant.const import CLIPMAX, CLIPMIN
import random
import os
import scipy
from .hadamard_utils import is_pow2, matmul_hadU


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        swc=None,
        lac=None,
        act_group_size=None,
        quant_method=None,
        rotate=True,
        mse=None,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.rotate = rotate
        self.quant_method = quant_method
        init_value = 4.0
        if lwc:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric  # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        self.sigmoid = nn.Sigmoid()
        self.enable = True
        self.group_size = group_size
        self.is_weight = shape != None
        self.recorded_x_max = None
        self.let_s = None
        self.act_group_size = act_group_size
        self.lac = lac
        self.swc = swc
        self.init_singlequant_params = torch.tensor(1)
        self.block_size = 128
        if self.rotate is None:
            # self.H = self.load_hadamard(self.block_size)
            self.H = self.get_hadamard_matrix(self.block_size)
        elif self.quant_method == "singlequant":
            self.R, self.R_ = None, None
            self.R1_l, self.R1_r = None, None
            if self.rotate is not False:
                self.init_singlequant_params = torch.tensor(0)
        # mse quantization
        self.mse = mse

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x.float() / scale).half()  # avoid overflow

        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)

        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]
        return x_dequant

    def rotation_r(self, weight):
        self.hidden_dim = weight.shape[-1]
        original_weight = weight.detach().clone()
        device = weight.device

        def givens_rotation_matrix(
            d: int,
            i: int,
            j: int,
            theta: torch.Tensor,
            device: torch.device = None,
            dtype: torch.dtype = None,
        ) -> torch.Tensor:
            R = torch.eye(d, device=device, dtype=dtype)
            c = torch.cos(theta)
            s = torch.sin(theta)
            R[i, i] = c
            R[j, j] = c
            R[i, j] = -s
            R[j, i] = s
            return R

        def build_orthogonal_matrix(
            d: int,
            rotations: list,
            device: torch.device = None,
            dtype: torch.dtype = None,
        ) -> torch.Tensor:
            R = torch.eye(d, device=device, dtype=dtype)
            for i, j, theta in rotations:
                G = givens_rotation_matrix(d, i, j, theta, device=device, dtype=dtype)
                R = G @ R
            return R

        def decompose_givens_to_e1(x: torch.Tensor) -> list:
            x = x.clone().to(torch.float64)
            d = x.numel()
            rotations = []
            for k in range(1, d):
                i = d - k - 1
                j = d - k
                a, b = x[i], x[j]
                theta = -torch.atan2(b, a)
                rotations.append((i, j, theta))
                # 更新向量
                c, s = torch.cos(theta), torch.sin(theta)
                xi, xj = x[i], x[j]
                x[i] = c * xi - s * xj
                x[j] = s * xi + c * xj
            return rotations

        def orthogonal_from_vector(x: torch.Tensor) -> torch.Tensor:
            device, dtype = x.device, x.dtype
            rotations = decompose_givens_to_e1(x)
            d = x.numel()
            return build_orthogonal_matrix(d, rotations, device=device, dtype=dtype)

        def orthogonal_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            Rx = orthogonal_from_vector(x)
            Ry = orthogonal_from_vector(y)
            return Ry.t() @ Rx

        dim = weight.shape[-1]
        weight = weight.reshape(
            -1, dim
        )  # (-1,dim) 在低维做,右乘旋转矩阵. 很多个64一组，找方差最大的这一组,让它均匀分布
        row_variances = torch.var(weight, dim=1)
        max_row_idx = torch.argmax(row_variances).item()
        w = weight[max_row_idx]  # 最大值所在的行,(-1)维度找最大的
        r = w.norm(p="fro")
        while True:
            v = torch.empty(len(w), device=weight.device).uniform_(-1, 1)
            # v = torch.ones(len(w),device=weight.device)
            # v = torch.linspace(-1,1,len(w),device=weight.device)
            # v = torch.linspace(-1,1,16,device=weight.device)
            # v = torch.cat([v]*(len(w)//16),dim=-1)
            v_norm = v.norm(p="fro")
            if v_norm != 0:
                break
        b = v / v_norm * r
        R_col = orthogonal_map(w, b).T  # Rw@w = Rb@b -> R_col = Rb.t() @ Rw
        rotated_weight = torch.matmul(original_weight, R_col)
        return (rotated_weight, R_col)

    def rotation_l(self, weight):
        self.hidden_dim = weight.shape[-1]
        original_weight = weight.detach().clone()

        def givens_rotation_matrix(
            d: int,
            i: int,
            j: int,
            theta: torch.Tensor,
            device: torch.device = None,
            dtype: torch.dtype = None,
        ) -> torch.Tensor:
            R = torch.eye(d, device=device, dtype=dtype)
            c = torch.cos(theta)
            s = torch.sin(theta)
            R[i, i] = c
            R[j, j] = c
            R[i, j] = -s
            R[j, i] = s
            return R

        def build_orthogonal_matrix(
            d: int,
            rotations: list,
            device: torch.device = None,
            dtype: torch.dtype = None,
        ) -> torch.Tensor:
            R = torch.eye(d, device=device, dtype=dtype)
            for i, j, theta in rotations:
                G = givens_rotation_matrix(d, i, j, theta, device=device, dtype=dtype)
                R = G @ R
            return R

        def decompose_givens_to_e1(
            x: torch.Tensor,
        ) -> list:  # 向量通过givens移动到一个onehot向量(非单位)
            x = x.clone().to(torch.float64)
            d = x.numel()
            rotations = []
            for k in range(1, d):
                i = d - k - 1
                j = d - k
                a, b = x[i], x[j]
                theta = -torch.atan2(b, a)
                rotations.append((i, j, theta))
                c, s = torch.cos(theta), torch.sin(theta)
                xi, xj = x[i], x[j]
                x[i] = c * xi - s * xj
                x[j] = s * xi + c * xj
            return rotations

        def orthogonal_from_vector(x: torch.Tensor) -> torch.Tensor:
            device, dtype = x.device, x.dtype
            rotations = decompose_givens_to_e1(x)
            d = x.numel()
            return build_orthogonal_matrix(d, rotations, device=device, dtype=dtype)

        def orthogonal_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            Rx = orthogonal_from_vector(x)
            Ry = orthogonal_from_vector(y)
            return Ry.t() @ Rx  # R @ x = Ry @ y -> R = Ry.t() @ Rx

        dim = weight.shape[1]  # 高维64

        weight = weight.reshape(dim, -1)  # (64,-1)
        row_variances = torch.var(weight, dim=0)  # (-1)
        max_row_idx = torch.argmax(row_variances).item()
        w = weight[:, max_row_idx]  # w为最大的值所对应的64维向量
        r = w.norm(p="fro")
        while True:
            v = torch.empty(len(w), device=weight.device).uniform_(-1, 1)  # 均匀分布
            ranks_w = torch.argsort(torch.argsort(w))
            v_sorted = torch.sort(v).values
            v = v_sorted[ranks_w]  # 均匀分布从打到小排列
            v_norm = v.norm(p="fro")
            if v_norm != 0:
                break
        b = v / v_norm * r  # 标准化,减少v的范数
        R_col = orthogonal_map(w, b)  # Rw@w = Rb@b -> R_col = Rb.t() @ Rw

        rotated_weight = torch.matmul(
            R_col, original_weight
        )  # (64,64) @(n,64,64) # 因此是左
        return (rotated_weight, R_col)

    def load_hadamard(self, n, device="cuda"):
        file_path = os.path.join("./hadamard_matrices", f"hadamard_{n}.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Hadamard matrix file {file_path} does not exist, please generate it"
            )
        tensor = torch.load(file_path, map_location=device).to(torch.float16)
        return tensor
    
    def get_hadamard_matrix(self, n, device="cuda"):
        if not is_pow2(n):
            return matmul_hadU(torch.eye(n))
        hadamard_matrix = scipy.linalg.hadamard(n)
        tensor = torch.from_numpy(hadamard_matrix)
        tensor = (tensor / torch.sqrt(torch.tensor(n, dtype=torch.float32))).float()
        return tensor.to(device)

    def online_singlequant_cali(self, weight):

        def rand_perm_rotation_left(
            weight: torch.Tensor,
        ):  # reshape为64,-1, 在宏观上做一次旋转(ART).
            device = weight.device
            dtype = weight.dtype
            dim = weight.shape[1]  # (n,64,64)
            weight_ = weight.reshape(dim, -1).abs()  # (64,-1)
            m, n = weight_.shape
            flat_idx = torch.argmax(weight_)
            r = int(flat_idx // n)  # r: 看在第一个64维度的第几个(行索引)
            c1 = int(flat_idx % n)  # c1: 看在第二个64维度的第几个(列索引)
            weight_2 = weight_.clone().detach()  # (64,-1)
            weight_2[r, :] = weight_[
                r, c1
            ]  # 将第r行第c1列的值(最大值)赋值给第r行,避免除了这一行的干扰
            min_idx = torch.argmin(weight_2)  # 找最小值所在的行
            r2 = int(min_idx // n)
            perm_rows = [r, r2] + [i for i in range(m) if i not in (r, r2)]
            Q = torch.eye(m, device=device, dtype=dtype)[
                perm_rows, :
            ]  # shape [m/64, m]  置换矩阵

            rot_small = torch.empty(m - 2, m - 2, device=device)  # 随机生成随机正交矩阵
            torch.nn.init.orthogonal_(rot_small)
            top_pad = torch.zeros(m - 2, 2, device=device, dtype=dtype)
            rot_bottom = torch.cat([top_pad, rot_small], dim=1)
            zero_top = torch.zeros(2, m, device=device, dtype=dtype)
            rot = torch.cat([zero_top, rot_bottom], dim=0)
            a = weight_[r, c1]
            b = weight_[r2, c1]
            c = (a + b) / torch.sqrt(2 * (a**2 + b**2))
            s = c * (a - b) / (a + b)
            rot[0, 0] = c
            rot[0, 1] = -s
            rot[1, 0] = s
            rot[1, 1] = c
            M = rot @ Q  # (Givens+O) @ (Q) # Q为置换矩阵
            return M

        init_shape = weight.shape
        weight = weight.squeeze()
        hidden_dim = weight.shape[-1]
        l_dim, r_dim = self.get_decompose_dim(hidden_dim)
        weight = weight.reshape(-1, l_dim, r_dim)  # (oc,l_dim,r_dim)
        # had_r = self.load_hadamard(r_dim, device=weight.device)
        had_r = self.get_hadamard_matrix(r_dim, device=weight.device)

        had_l = rand_perm_rotation_left(
            weight
        )  # (Givens,O) @ Q, 权重reshape为(64,-1)后,
        weight = torch.matmul(had_l, weight)
        weight = torch.matmul(weight, had_r)  # 权重右乘随机哈达玛变换
        self.R1_l = had_l.half()
        self.R1_r = had_r.half()
        weight, self.R_ = self.rotation_l(weight)  # 左乘URT
        iterations = 0 # 好像没啥用
        if True:
            for i in range(iterations):
                weight,R_ = self.rotation_l(weight)
                self.R_ = R_ @ self.R_
        weight, self.R = self.rotation_r(weight)  # 右乘RHT
        if True:
            for i in range(iterations):
                weight,R = self.rotation_r(weight)
                self.R = self.R @ R
        weight = weight.reshape(init_shape)
        return weight

    def get_decompose_dim(self, dim: int):
        sqrt_dim = math.sqrt(dim)
        n_2 = 1
        max_k = int(math.log2(dim))
        for k in range(max_k + 1):
            a = 1 << k  # 2^k
            if dim % a != 0:
                continue
            if abs(a - sqrt_dim) < abs(n_2 - sqrt_dim):
                n_2 = a
        n_1 = dim // n_2
        return n_1, n_2

    def init_singlequant(self, x: torch.Tensor):
        if self.quant_method is None:
            return x
        if self.rotate is None:
            x_shape = x.shape
            hadamard = self.H.to(x)
            x = x.reshape(-1, self.block_size)
            x = x.matmul(hadamard).view(x_shape)
        elif self.quant_method == "singlequant":
            if self.rotate:
                if not self.init_singlequant_params:
                    x = self.online_singlequant_cali(x)
                    self.init_singlequant_params = torch.tensor(1)
                else:
                    x = x.squeeze()
                    if self.R1_r != None:
                        had_l = self.R1_l.to(x.device)
                        had_r = self.R1_r.to(x.device)
                        init_shape = x.shape
                        x = x.reshape(-1, had_l.shape[0], had_r.shape[0])
                        x = torch.matmul(had_l, x)
                        x = torch.matmul(x, had_r)
                        if self.R_ != None:
                            R_ = self.R_.to(x.device)
                            x = torch.matmul(R_, x)
                        if self.R != None:
                            R = self.R.to(x.device)
                            x = torch.matmul(x, R)
                        x = x.reshape(init_shape)
        else:
            raise NotImplementedError
        return x

    def forward(self, x: torch.Tensor, return_no_quant=False):
        if hasattr(self, "smooth_scales"):
            x /= self.smooth_scales.to(x.device)
        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            x = self.init_singlequant(x)
        if return_no_quant:
            reduce_shape = [-1]
            xmin = x.amin(reduce_shape, keepdim=True)
            xmax = x.amax(reduce_shape, keepdim=True)
            if self.swc:
                xmax = self.swc * xmax
                xmin = self.swc * xmin
            elif self.lwc:
                xmax = self.sigmoid(self.upbound_factor) * xmax
                xmin = self.sigmoid(self.lowbound_factor) * xmin
            if self.lac:
                xmax = self.lac * xmax
                xmin = self.lac * xmin
            return x

        if self.recorded_x_max is None:
            self.recorded_x_max = (
                x.abs().reshape(-1, x.shape[-1]).max(axis=0).values
            )  # 记录每个通道的最大值
        if self.let_s is not None:
            x /= self.let_s

        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros(
                    (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
                )
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True).to(x.device)
        xmax = x.amax(reduce_shape, keepdim=True).to(x.device)
        if self.swc:  # 进行shrink缩放
            xmax = self.swc * xmax
            xmin = self.swc * xmin
        elif self.lwc:
            xmax = self.sigmoid(self.upbound_factor.to(x.device)) * xmax
            xmin = self.sigmoid(self.lowbound_factor.to(x.device)) * xmin
        elif self.lac:
            xmax = self.lac * xmax
            xmin = self.lac * xmin

        if self.mse is None:
            if self.symmetric:
                abs_max = torch.max(xmax.abs(), xmin.abs())
                scale = abs_max / (2 ** (self.n_bits - 1) - 1)
                self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
                zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
            else:
                range = xmax - xmin
                scale = range / (2**self.n_bits - 1)
                self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
                zero_point = -(xmin) / (self.scale)
        else:
            zero_point = torch.zeros_like(xmax)
            self.scale = torch.ones_like(xmax)
            best_err = torch.full_like(xmax, float("inf"), dtype=torch.float32)
            for p in np.linspace(1, 0.4, 100):
                pxmax = xmax * p
                pxmin = xmin * p
                if self.symmetric:
                    abs_max = torch.max(pxmax.abs(), pxmin.abs())
                    s = abs_max / (2 ** (self.n_bits - 1) - 1)
                    s = s.clamp(min=CLIPMIN, max=CLIPMAX)
                    zp = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(s)
                else:
                    range = pxmax - pxmin
                    s = range / (2**self.n_bits - 1)
                    s = s.clamp(min=CLIPMIN, max=CLIPMAX)
                    zp = (-(pxmin) / (s)).round()
                zp = zp.clamp(min=-CLIPMAX, max=CLIPMAX)
                err = (
                    (self.fake_quant(x, s, zp) - x)
                    .abs()
                    .pow(self.mse)
                    .mean(dim=-1, keepdim=True)
                )
                mask = err < best_err
                if mask.any():
                    best_err[mask] = err[mask].to(best_err.dtype)
                    self.scale[mask] = s[mask]
                    zero_point[mask] = zp[mask]
        self.round_zero_point = zero_point.clamp(min=-CLIPMAX, max=CLIPMAX).round()

    def register_scales_and_zeros(self):
        self.register_buffer("scales", self.scale)
        self.register_buffer("zeros", self.round_zero_point)
        del self.scale
        del self.round_zero_point

    def register_singlequant_params(self):
        if self.rotate is not True:
            return

        R1_l, R1_r = self.R1_l, self.R1_r
        R, R_ = self.R, self.R_
        delattr(self, "R1_l")
        delattr(self, "R1_r")
        delattr(self, "R")
        delattr(self, "R_")
        delattr(self, "init_singlequant_params")
        self.register_buffer("R1_r", R1_r)
        self.register_buffer("R1_l", R1_l)
        self.register_buffer("R_", R_)
        self.register_buffer("R", R)
        self.register_buffer("init_singlequant_params", torch.tensor(1))

    def copy_singlequant_params(self, quantizer_ref):
        if quantizer_ref.rotate is True:
            assert quantizer_ref.init_singlequant_params == True
            if quantizer_ref.R != None:
                self.R = quantizer_ref.R.clone().detach()
                self.R_ = quantizer_ref.R_.clone().detach()
            if quantizer_ref.R1_l != None:
                self.R1_l = quantizer_ref.R1_l.clone().detach()
                self.R1_r = quantizer_ref.R1_r.clone().detach()
            self.init_singlequant_params = torch.tensor(1)
