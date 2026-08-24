from ...builder import register_other_model


@register_other_model("XHKokoroModel")
class XHKokoroModel:
    """Registry entry for the static Host/NPU Kokoro pipeline."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.kokoro.workflow:KokoroWorkflow"


__all__ = ["XHKokoroModel"]
