# ruff: noqa: E501

import math
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhquant import nn as xhnn

from ...register import DynamicRegister, XHLLM_TRACEABLE_MODULES


"""
minicpmo的类是动态加载的, 所以需要动态注册, 不能使用装饰器
"""
DType = torch.dtype


def _resolve_nn_cls(nn_cls_str: str):
    def _canon(name: str) -> str:
        return name.replace("-", "").replace("_", "").lower()

    nn_cls = None
    try:
        nn_cls = eval(nn_cls_str)
    except Exception:
        class_name = nn_cls_str.split(".")[-1]
        target = _canon(nn_cls_str)
        for name in list(sys.modules.keys()):
            obj = sys.modules[name]
            if hasattr(obj, class_name):
                candidate = getattr(obj, class_name)
                full_cls_name = f"{candidate.__module__}.{candidate.__name__}"
                if full_cls_name == nn_cls_str or _canon(full_cls_name) == target:
                    nn_cls = candidate
                    break
    if isinstance(nn_cls, type) and issubclass(nn_cls, nn.Module):
        return nn_cls
    return None


def _register_traceable_if_available(cls_to_key: Dict):
    resolved_cls_to_key = {}
    for nn_cls, key in cls_to_key.items():
        if isinstance(nn_cls, str):
            resolved_cls = _resolve_nn_cls(nn_cls)
            if resolved_cls is not None:
                resolved_cls_to_key[resolved_cls] = key
        elif isinstance(nn_cls, type) and issubclass(nn_cls, nn.Module):
            resolved_cls_to_key[nn_cls] = key

    if not resolved_cls_to_key:

        def _identity(dm_class):
            return dm_class

        return _identity

    return XHLLM_TRACEABLE_MODULES.register_module(resolved_cls_to_key)


@_register_traceable_if_available(
    {
        "transformers_modules.MiniCPM-o-4_5.modeling_navit_siglip.SiglipAttention": "minicpmo.vision.SiglipAttention",
    }
)
class _SiglipAttention(DynamicRegister):
    def _setup(self, *args, **kwargs):
        self.softmax = xhnn.SoftmaxPlus(-1, dtype=torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # batch_size, q_len, _ = hidden_states.shape if hasattr(hidden_states, 'shape') else hidden_states.size()
        batch_size, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        if attention_mask is not None:
            # if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
            #     )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = self.softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

        # if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


@_register_traceable_if_available(
    {
        "transformers_modules.MiniCPM-o-4_5.modeling_navit_siglip.SiglipVisionEmbeddings": "minicpmo.vision.SiglipVisionEmbeddings",
    }
)
class _SiglipVisionEmbeddings(DynamicRegister):
    def _setup(self, cfg: Optional[Dict] = None):
        # self.max_im_h = cfg.all_pixel_shape[2]
        # self.max_im_w = cfg.all_pixel_shape[3]
        if isinstance(self.patch_embedding.padding, str):
            if self.patch_embedding.padding == "valid":
                self.patch_embedding.padding = (0, 0)
            else:
                raise AssertionError(f"unsupported patch embedding padding: {self.patch_embedding.padding!r}")

    def get_position_ids(
        self,
        pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor,
        tgt_sizes: Optional[torch.IntTensor] = None,
    ):
        batch_size = pixel_values.shape[0]
        # batch_size = pixel_values.shape[0] if hasattr(pixel_values, 'shape') else pixel_values.size(0)
        assert batch_size == 1
        max_im_h, max_im_w = pixel_values.shape[2], pixel_values.shape[3]
        # max_im_h, max_im_w = pixel_values.shape[2] if hasattr(pixel_values, 'shape') else pixel_values.size(2), pixel_values.shape[3] if hasattr(pixel_values, 'shape') else pixel_values.size(3)
        # max_im_h = self.max_im_h
        # max_im_w = self.max_im_w
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        return position_ids

    def forward(self, pixel_values: torch.FloatTensor, position_ids: torch.IntTensor) -> torch.Tensor:
        # batch_size = pixel_values.size(0)
        # assert batch_size == 1

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # # # max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        # # max_im_h = self.max_im_h
        # # max_im_w = self.max_im_w
        # # max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        # # boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        # # position_ids = torch.full(
        # #     size=(
        # #         batch_size,
        # #         max_nb_patches_h * max_nb_patches_w,
        # #     ),
        # #     fill_value=0,
        # # )

        # # for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        # #     if tgt_sizes is not None:
        # #         nb_patches_h = tgt_sizes[batch_idx][0]
        # #         nb_patches_w = tgt_sizes[batch_idx][1]
        # #     else:
        # #         nb_patches_h = p_attn_mask[:, 0].sum()
        # #         nb_patches_w = p_attn_mask[0].sum()

        # #     fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        # #     fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

        # #     bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        # #     bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        # #     pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
        # #     position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        # position_ids = self.get_position_ids(pixel_values, patch_attention_mask, tgt_sizes)

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


@_register_traceable_if_available(
    {
        "transformers_modules.MiniCPM-o-4_5.modeling_minicpmo.Resampler": "minicpmo.vision.Resampler",
    }
)
class _Resampler(DynamicRegister):
    def _setup(self, cfg: Optional[Dict] = None):
        self.image_slice_max_size = cfg.image_slice_max_size
        self._adjust_pos_cache(torch.tensor([self.image_slice_max_size]), "cpu")

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def forward(self, x: torch.Tensor, tgt_sizes: torch.Tensor):
        bs = 1
        dtype = x.dtype

        max_h, max_w = self.image_slice_max_size
        max_patch_len = int(max_h * max_w)

        base_pos_embed = self.pos_embed[:max_h, :max_w, :].reshape((max_patch_len, -1)).to(dtype)
        pos_embed = base_pos_embed.unsqueeze(1)

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        # q = self.ln_q(self.query)  # Q * D

        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, bs),
            x + pos_embed,  # L * B * D +  L * B * D
            x,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


def _create_nn_linear(w, b):
    out_features, in_feature = w.shape[:2]
    m = nn.Linear(in_feature, out_features, b is not None)
    m.weight.data = w
    if b is not None:
        m.bias.data = b
    return m


@_register_traceable_if_available(
    {
        nn.MultiheadAttention: "minicpmo.vision.MultiheadAttention",
    }
)
class _MultiheadAttention(DynamicRegister):
    def _setup(self, cfg: Optional[Dict] = None):
        self.scaled = 1 / math.sqrt(128)
        self._head_dim_const = self.embed_dim // self.num_heads
        self.dropout = 0.0

        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)

        # q, k, v = F.linear(query, w_q, b_q), F.linear(key, w_k, b_k), F.linear(value, w_v, b_v)
        out_features, in_features = w_q.shape[:2]
        self.in_proj_q = _create_nn_linear(w_q, b_q)
        self.in_proj_k = _create_nn_linear(w_k, b_k)
        self.in_proj_v = _create_nn_linear(w_v, b_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # why_not_fast_path = ""
        # if (
        #     (attn_mask is not None and torch.is_floating_point(attn_mask))
        #     or (key_padding_mask is not None)
        #     and torch.is_floating_point(key_padding_mask)
        # ):
        #     why_not_fast_path = "floating-point masks are not supported for fast path."

        # is_batched = query.dim() == 3
        is_batched = True

        # key_padding_mask = _canonical_mask(
        #     mask=key_padding_mask,
        #     mask_name="key_padding_mask",
        #     other_type=F._none_or_dtype(attn_mask),
        #     other_name="attn_mask",
        #     target_type=query.dtype,
        # )

        # attn_mask = _canonical_mask(
        #     mask=attn_mask,
        #     mask_name="attn_mask",
        #     other_type=None,
        #     other_name="",
        #     target_type=query.dtype,
        #     check_other=False,
        # )

        # if not is_batched:
        #     why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        # elif query is not key or key is not value:
        #     # When lifting this restriction, don't forget to either
        #     # enforce that the dtypes all match or test cases where
        #     # they don't!
        #     why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        # elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
        #     why_not_fast_path = (
        #         f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        #     )
        # elif self.in_proj_weight is None:
        #     why_not_fast_path = "in_proj_weight was None"
        # elif query.dtype != self.in_proj_weight.dtype:
        #     # this case will fail anyway, but at least they'll get a useful error message.
        #     why_not_fast_path = (
        #         f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        #     )
        # elif self.training:
        #     why_not_fast_path = "training is enabled"
        # elif (self.num_heads % 2) != 0:
        #     why_not_fast_path = "self.num_heads is not even"
        # elif not self.batch_first:
        #     why_not_fast_path = "batch_first was not True"
        # elif self.bias_k is not None:
        #     why_not_fast_path = "self.bias_k was not None"
        # elif self.bias_v is not None:
        #     why_not_fast_path = "self.bias_v was not None"
        # elif self.add_zero_attn:
        #     why_not_fast_path = "add_zero_attn was enabled"
        # elif not self._qkv_same_embed_dim:
        #     why_not_fast_path = "_qkv_same_embed_dim was not True"
        # elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
        #     why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
        #                          is not supported with NestedTensor input"
        # elif torch.is_autocast_enabled():
        #     why_not_fast_path = "autocast is enabled"

        # if not why_not_fast_path:
        #     tensor_args = (
        #         query,
        #         key,
        #         value,
        #         self.in_proj_weight,
        #         self.in_proj_bias,
        #         self.out_proj.weight,
        #         self.out_proj.bias,
        #     )
        #     # We have to use list comprehensions below because TorchScript does not support
        #     # generator expressions.
        #     if torch.overrides.has_torch_function(tensor_args):
        #         why_not_fast_path = "some Tensor argument has_torch_function"
        #     elif _is_make_fx_tracing():
        #         why_not_fast_path = "we are running make_fx tracing"
        #     elif not all(_check_arg_device(x) for x in tensor_args):
        #         why_not_fast_path = (
        #             "some Tensor argument's device is neither one of "
        #             f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
        #         )
        #     elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
        #         why_not_fast_path = (
        #             "grad is enabled and at least one of query or the "
        #             "input/output projection weights or biases requires_grad"
        #         )
        #     if not why_not_fast_path:
        #         merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

        #         if self.in_proj_bias is not None and self.in_proj_weight is not None:
        #             return torch._native_multi_head_attention(
        #                 query,
        #                 key,
        #                 value,
        #                 self.embed_dim,
        #                 self.num_heads,
        #                 self.in_proj_weight,
        #                 self.in_proj_bias,
        #                 self.out_proj.weight,
        #                 self.out_proj.bias,
        #                 merged_mask,
        #                 need_weights,
        #                 average_attn_weights,
        #                 mask_type,
        #             )

        # any_nested = query.is_nested or key.is_nested or value.is_nested
        # assert not any_nested, (
        #     "MultiheadAttention does not support NestedTensor outside of its fast path. "
        #     + f"The fast path was not hit because {why_not_fast_path}"
        # )

        # if self.batch_first and is_batched:
        #     # make sure that the transpose op does not affect the "is" property
        #     if key is value:
        #         if query is key:
        #             query = key = value = query.transpose(1, 0)
        #         else:
        #             query, key = (x.transpose(1, 0) for x in (query, key))
        #             value = key
        #     else:
        #         query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[Tensor],
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = True  # _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        # if not is_batched:
        #     # unsqueeze if the input is unbatched
        #     query = query.unsqueeze(1)
        #     key = key.unsqueeze(1)
        #     value = value.unsqueeze(1)
        #     if key_padding_mask is not None:
        #         key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
        tgt_len, _, embed_dim = query.shape
        bsz = 1
        src_len, _, _ = key.shape

        # key_padding_mask = _canonical_mask(
        #     mask=key_padding_mask,
        #     mask_name="key_padding_mask",
        #     other_type=_none_or_dtype(attn_mask),
        #     other_name="attn_mask",
        #     target_type=query.dtype,
        # )

        # if is_causal and attn_mask is None:
        #     raise RuntimeError(
        #         "Need attn_mask if specifying the is_causal hint. "
        #         "You may use the Transformer module method "
        #         "`generate_square_subsequent_mask` to create this mask."
        #     )

        # if is_causal and key_padding_mask is None and not need_weights:
        if False:
            # when we have a kpm or need weights, we need attn_mask
            # Otherwise, we use the is_causal hint go as is_causal
            # indicator to SDPA.
            attn_mask = None
        else:
            # attn_mask = _canonical_mask(
            #     mask=attn_mask,
            #     mask_name="attn_mask",
            #     other_type=None,
            #     other_name="",
            #     target_type=query.dtype,
            #     check_other=False,
            # )

            if key_padding_mask is not None:
                # We have the attn_mask, and use that to merge kpm into it.
                # Turn off use of is_causal hint, as the merged mask is no
                # longer causal.
                is_causal = False

        # assert (
        #     embed_dim == embed_dim_to_check
        # ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        # if isinstance(embed_dim, torch.Tensor):
        #     # embed_dim can be a tensor when JIT tracing
        #     head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        # else:
        #     head_dim = embed_dim // num_heads
        head_dim = self._head_dim_const
        # assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        # if use_separate_proj_weight:
        #     # allow MHA to have different embedding dimensions when separate projection weights are used
        #     assert (
        #         key.shape[:2] == value.shape[:2]
        #     ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        # else:
        #     assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
            # q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
            w_q, w_k, w_v = in_proj_weight.chunk(3)
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            # q, k, v = F.linear(query, w_q, b_q), F.linear(key, w_k, b_k), F.linear(value, w_v, b_v)
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        else:
            # assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            # assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            # assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            # if in_proj_bias is None:
            #     b_q = b_k = b_v = None
            # else:
            #     b_q, b_k, b_v = in_proj_bias.chunk(3)
            # q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
            raise AssertionError("separate projection weights are not supported")
        # prep attention mask

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * num_heads, (
                f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            )
            assert static_k.size(2) == head_dim, f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, (
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            )
            assert static_v.size(2) == head_dim, f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            # assert key_padding_mask.shape == (
            #     bsz,
            #     src_len,
            # ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, num_heads, -1, -1)
                .reshape(bsz * num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        if need_weights:
            B, Nt, E = q.shape
            # q_scaled = q / math.sqrt(E)
            q_scaled = q * self.scaled

            assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

            if attn_mask is not None:
                # attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
                attn_output_weights = q_scaled @ k.transpose(-2, -1) + attn_mask

            else:
                # attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1)) #要求输入必须是三维
                attn_output_weights = q_scaled @ k.transpose(-2, -1)
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)

            # attn_output = torch.bmm(attn_output_weights, v)
            attn_output = attn_output_weights @ v
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, embed_dim)
            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            # attn_mask can be either (L,S) or (N*num_heads, L, S)
            if attn_mask is not None:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

            q = q.view(bsz, num_heads, tgt_len, head_dim)
            k = k.view(bsz, num_heads, src_len, head_dim)
            v = v.view(bsz, num_heads, src_len, head_dim)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, embed_dim)

            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            return attn_output, None


@_register_traceable_if_available(
    {
        "transformers_modules.MiniCPM-o-4_5.modeling_navit_siglip.SiglipVisionTransformer": "minicpmo.vision.SiglipVisionTransformer",
    }
)
class _SiglipVisionTransformer(DynamicRegister):
    def _setup(self, cfg: Optional[Dict] = None):
        self.image_slice_max_size: Tuple[int, int] = cfg["image_slice_max_size"]
        # max_seq_len = self.image_slice_max_size[0] * self.image_slice_max_size[1]
        # tgt_w, tgt_h = self.image_slice_max_size
        # attention_mask = torch.zeros((1, 1, max_seq_len, max_seq_len), dtype=torch.float16)
        # resampler_pos_embed = (
        #     self.resampler.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, 1, -1)).to(torch.float16)
        # )
        # resampler_key_padding_mask = torch.zeros((1, max_seq_len), dtype=torch.float16)

        # self.register_buffer("attention_mask", attention_mask)
        # self.register_buffer("resampler_pos_embed", resampler_pos_embed)
        # self.register_buffer("resampler_key_padding_mask", resampler_key_padding_mask)

    def forward(
        self,
        pixel_values,
        position_ids: Optional[torch.IntTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        tgt_sizes: Optional[torch.IntTensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ):
        # attention_mask = self.attention_mask
        # resampler_pos_embed = self.resampler_pos_embed
        # resampler_key_padding_mask = self.resampler_key_padding_mask
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False
        output_attentions = False
        output_hidden_states = False
        # batch_size = pixel_values.size(0)
        # if patch_attention_mask is None:
        #     patch_attention_mask = torch.ones(
        #         size=(
        #             batch_size,
        #             pixel_values.size(2) // self.config.patch_size,
        #             pixel_values.size(3) // self.config.patch_size,
        #         ),
        #         dtype=torch.bool,
        #         device=pixel_values.device,
        #     )

        hidden_states = self.embeddings(pixel_values, position_ids)
        # patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        # if not torch.any(~patch_attention_mask):
        #     attention_mask = None
        # else:
        #     attention_mask = (
        #         _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
        #         if not self._use_flash_attention_2
        #         else patch_attention_mask
        #     )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        last_hidden_state = self.resampler(last_hidden_state, tgt_sizes)
        # if not return_dict:
        #     return (last_hidden_state, None) + encoder_outputs[1:]

        # return BaseModelOutputWithPooling(
        #     last_hidden_state=last_hidden_state,
        #     # pooler_output=None,
        #     # hidden_states=encoder_outputs.hidden_states,
        #     # attentions=encoder_outputs.attentions,
        # )
        return last_hidden_state


def register_wrap_cls(hf_model):
    vpm = hf_model.vpm
    _SiglipVisionTransformer.register(type(vpm))
    _SiglipVisionEmbeddings.register(type(vpm.embeddings))
    _SiglipAttention.register(type(vpm.encoder.layers[0].self_attn))
    _Resampler.register(type(hf_model.resampler))
    _MultiheadAttention.register(type(hf_model.resampler.attn))
