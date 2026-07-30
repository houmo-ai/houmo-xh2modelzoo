# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from xhmodel_merak.xh_other_model.builder import register_other_model


@register_other_model("XHWan22Model")
class XHWan22Model:
    """Wan2.2 non-LLM workflow binding model."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.wan2_2.workflow:Wan22Workflow"


__all__ = ["XHWan22Model"]
