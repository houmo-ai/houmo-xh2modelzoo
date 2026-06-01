from typing import List, Union
from abc import abstractmethod
from xhquant.api import QuantGraph, FrontendGraph


class InferAdapter:
    # Adapter that builds a runnable inference object from either
    # - quantized graphs (QuantGraph / FrontendGraph)
    # - exported HMONNX files
    # Subclasses should return an instance that can run real inference.

    @classmethod
    def from_qmodel(cls, model: Union[QuantGraph, FrontendGraph], *args, **kwargs):
        """Build a demo instance from a quantized model or a frontend graph.
        Args:
            model (Union[QuantGraph,FrontendGraph]): The quantized model or the frontend graph.
        Not Support Native Model.
        """
        raise NotImplementedError

    @classmethod
    def from_hmonnx(cls, onnx_path: Union[str, List[str]], *args, **kwargs):
        """Build an inference adapter from exported HMONNX file(s).
        Args:
            onnx_path (Union[str, List[str]]): The path to the exported HMONNX file(s).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            Any: The result of the build.
        """
        raise NotImplementedError("from_hmonnx must be implemented in subclass")

    @abstractmethod
    def demo(self, *args, **kwargs):
        # This method is intended for simple demonstrations, for example, generating an answer given an input sentence.
        # For the evaluation of LLMs, we will integrate this into an HF-compatible model and perform evaluation.
        raise NotImplementedError("demo must be implemented in subclass")
    