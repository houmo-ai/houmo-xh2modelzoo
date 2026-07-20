import torch

from xhmodel_merak.xh_llm.kv_cache_mixin import KVCacheWithLinearMixin
from xhmodel_merak.xh_llm.types import KVCacheWithLinearConfig


def test_meta_kv_cache_scope_keeps_linear_attention_caches_on_cpu():
    config = KVCacheWithLinearConfig(
        num_layers=1,
        kv_cache_shape=[1, 1, 4, 2],
        linear_kv_cache_config={
            "conv_dim": 6,
            "conv_kernel_size": 4,
            "num_v_heads": 1,
            "head_k_dim": 2,
            "head_v_dim": 2,
            "num_layers": 1,
        },
    )
    cache_mixin = KVCacheWithLinearMixin(config).to("meta")

    with cache_mixin.kv_cache_scope(device="meta"):
        assert cache_mixin.past_key_caches[0].device.type == "meta"
        assert cache_mixin.past_value_caches[0].device.type == "meta"
        assert cache_mixin.past_conv_caches[0].device.type == "cpu"
        assert cache_mixin.past_recurrent_states[0].device.type == "cpu"
        assert torch.device(cache_mixin._device).type == "meta"

    assert not cache_mixin.past_key_caches
    assert not cache_mixin.past_value_caches
    assert not cache_mixin.past_conv_caches
    assert not cache_mixin.past_recurrent_states
