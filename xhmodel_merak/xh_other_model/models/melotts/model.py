from ...builder import register_other_model


@register_other_model("XHMeloTTSModel")
class XHMeloTTSModel:
    """Registry entry for strict static MeloTTS encoder/decoder graphs."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.melotts.workflow:MeloTTSWorkflow"


__all__ = ["XHMeloTTSModel"]
