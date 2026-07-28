from ...builder import register_other_model
from . import _model_opt  # noqa: F401


@register_other_model("XHWhisperModel")
class XHWhisperModel:
    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.whisper.workflow:WhisperWorkflow"


__all__ = ["XHWhisperModel"]
