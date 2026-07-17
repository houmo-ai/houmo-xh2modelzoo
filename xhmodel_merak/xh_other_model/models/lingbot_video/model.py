from ...builder import register_other_model


@register_other_model("XHLingBotVideoModel")
class XHLingBotVideoModel:
    """Registry facade used to bind LingBot Video to its Merak workflow."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.lingbot_video.workflow:LingBotVideoWorkflow"
