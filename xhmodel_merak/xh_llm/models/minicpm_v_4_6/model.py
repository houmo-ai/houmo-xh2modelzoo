"""Registered MiniCPM-V-4.6 model for the Merak LLM workflow."""

from xhmodel_merak.xh_llm.builder import register_llm_model

from .text_model import MiniCPMV46TextModel


@register_llm_model("MiniCPMV46ForConditionalGeneration")
class XHMiniCPMV46Model(MiniCPMV46TextModel):
    """MiniCPM Vision plus Qwen3.5 text adapter."""

    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.minicpm_v_4_6.workflow:MiniCPMV46Workflow"


__all__ = ["XHMiniCPMV46Model"]
