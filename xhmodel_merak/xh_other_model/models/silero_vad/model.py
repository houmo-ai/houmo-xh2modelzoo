from ...builder import register_other_model


@register_other_model("XHSileroVADModel")
class XHSileroVADModel:
    """Registry entry for the two-rate static Silero VAD workflow."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.silero_vad.workflow:SileroVADWorkflow"


__all__ = ["XHSileroVADModel"]
