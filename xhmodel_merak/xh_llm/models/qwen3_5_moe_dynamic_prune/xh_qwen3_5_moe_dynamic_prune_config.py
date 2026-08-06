from ..qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig


class XHQwen3_5MoeDynamicPruneModelConfig(XHQwen3_5MoeModelConfig):  # noqa: N801
    """Qwen3.5-MoE config with converter-independent prune controls."""

    def __init__(
        self,
        *,
        dynamic_prune_threshold: float = 0.0,
        dynamic_prune_s_scalar_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dynamic_prune_threshold = float(dynamic_prune_threshold)
        self.dynamic_prune_s_scalar_path = dynamic_prune_s_scalar_path
