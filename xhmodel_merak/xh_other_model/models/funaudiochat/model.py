from ...builder import register_other_model


@register_other_model("XHFunAudioChatModel")
class XHFunAudioChatModel:
    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.funaudiochat.workflow:FunAudioChatWorkflow"
