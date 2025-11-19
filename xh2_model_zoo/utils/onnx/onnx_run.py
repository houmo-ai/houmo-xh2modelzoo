import os
from typing import Union

import onnx
import onnxruntime as ort


class OnnxRuntimeDetector:
    def __init__(self, weights: str, device: str):
        """
            read onnx model path and runtime
        Args:
            weights (str): onnx model path
            device_id (int): device id, -1 cpu, >0 gpu
        """
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(weights, providers=providers)
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.in_names = [x.name for x in self.session.get_inputs()]
        self.in_shape = [x.shape for x in self.session.get_inputs()]

    def forward(self, input_dict: Union[list, dict]) -> dict:
        """
            onnxruntime inference, return
        Args:
            input_dict (dict): {name: np.array, ...}

        Returns:
            dict: {name: np.array, ...}
        """
        # 支持输入列表自动按shape匹配输入
        if isinstance(input_dict, list):
            my_input_dict = {}
            my_set = set(tuple(x) for x in self.in_shape)
            assert len(self.in_shape) == len(my_set), "inputs have same shape please input dict with name"

            for i, (shape, name) in enumerate(zip(self.in_shape, self.in_names)):
                # pop dynamic shape
                new_shape = set([e for e in shape if e != -1 or not isinstance(e, str)])
                max_sim = -1
                idx = 0
                for j, inp in enumerate(input_dict):
                    if len(inp.shape) == len(shape):
                        c = new_shape & set(inp.shape)
                        if len(c) > max_sim:
                            max_sim = len(c)
                            idx = j

                my_input_dict[name] = input_dict[idx]

            assert len(my_input_dict) == len(self.in_shape), "inputs with wrong shape"
            input_dict = my_input_dict

        outs = dict()
        y = self.session.run(self.output_names, input_dict)
        for name, out in zip(self.output_names, y):
            outs[name] = out
        return outs

    def forward_single_input(self, data):
        input_dict = {self.in_names[0]: data}
        return self.session.run(self.output_names, input_dict)

    def forward_single_input_multi_output(self, data):
        y = self.forward_single_input(data)
        out_dict = dict()
        for name, out in zip(self.output_names, y):
            out_dict[name] = out
        return out_dict


class dummy_cls:
    def forward(self, input):
        return input


def remove_initializer_from_input(onnx_path: str):
    onnx_model = onnx.load(onnx_path)
    inputs = onnx_model.graph.input
    name_to_input = {}

    for input in inputs:
        name_to_input[input.name] = input
    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(onnx_model, onnx_path)
