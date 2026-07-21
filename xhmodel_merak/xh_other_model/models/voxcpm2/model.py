from ...builder import register_other_model


@register_other_model("XHVoxCPM2Model")
class XHVoxCPM2Model:
    """Routing model for the multi-component VoxCPM2 workflow."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.voxcpm2.workflow:VoxCPM2Workflow"
