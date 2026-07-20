from ...builder import register_other_model


@register_other_model("XHSenseVoiceSmallModel")
class XHSenseVoiceSmallModel:
    """Registry entry used by :class:`AutoWorkflow` for SenseVoiceSmall."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.sensevoice_small.workflow:SenseVoiceSmallWorkflow"


__all__ = ["XHSenseVoiceSmallModel"]
