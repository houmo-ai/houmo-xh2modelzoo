from ...builder import register_other_model


@register_other_model("XHZImageModel")
class XHZImageModel:
    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.zimage.workflow:ZImageWorkflow"
