"""Registered MiniCPM-V-4.5 model for the Merak LLM workflow."""

from xhmodel_merak.xh_llm.builder import register_llm_model

from .text_model import MiniCPMV45TextModel


@register_llm_model("MiniCPMV45ForConditionalGeneration")
class XHMiniCPMV45Model(MiniCPMV45TextModel):
    """MiniCPM Vision plus Qwen3-8B text adapter."""

    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.minicpm_v_4_5.workflow:MiniCPMV45Workflow"


__all__ = ["XHMiniCPMV45Model"]
