import onnx_graphsurgeon as gs


def install_moeblock_shared_initializer_compat() -> None:
    """Keep shared MoeBlock constants alive until their last parser consumer."""

    from xhquant.xhonnxruntime.parsers import moeblock as parser_module

    if getattr(parser_module, "_laguna_shared_initializer_compat", False):
        return

    def _release_constant_inputs(inputs: list[gs.Tensor], current_node: gs.Node | None = None) -> None:
        for input_tensor in inputs:
            if not isinstance(input_tensor, gs.Constant):
                continue
            consumers = input_tensor.outputs
            if current_node is None or not consumers or consumers[-1] is current_node:
                input_tensor.values = None

    original_from_onnx_node = parser_module.MoeBlock.from_onnx_node.__func__

    def _from_onnx_node(cls, node, context):
        if not hasattr(parser_module, "_release_constant_inputs"):
            module = original_from_onnx_node(cls, node, context)
            _release_constant_inputs(node.inputs, node)
            return module

        original_release = parser_module._release_constant_inputs
        parser_module._release_constant_inputs = lambda inputs: _release_constant_inputs(inputs, node)
        try:
            return original_from_onnx_node(cls, node, context)
        finally:
            parser_module._release_constant_inputs = original_release

    parser_module.MoeBlock.from_onnx_node = classmethod(_from_onnx_node)
    parser_module._laguna_shared_initializer_compat = True
