from ...builder import register_other_model


@register_other_model("XHHyMT2Model")
class XHHyMT2Model:
    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.hy_mt2.workflow:HyMT2Workflow"
