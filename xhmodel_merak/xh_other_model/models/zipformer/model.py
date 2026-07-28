from ...builder import register_other_model


@register_other_model("XHStreamingZipformerModel")
class XHStreamingZipformerModel:
    """Registry entry for the static streaming Zipformer encoder."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.zipformer.workflow:StreamingZipformerWorkflow"


__all__ = ["XHStreamingZipformerModel"]
