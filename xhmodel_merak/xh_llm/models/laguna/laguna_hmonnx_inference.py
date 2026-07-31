from ...hmonnx import TextLLMHMONNXModel


class XHLagunaHMONNXModel(TextLLMHMONNXModel):
    def get_tokenizer(self, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("fix_mistral_regex", True)
        return super().get_tokenizer(**kwargs)
