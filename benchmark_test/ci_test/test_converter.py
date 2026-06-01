import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from xh_model_zoo.xh_llm.llm_converter import LLMConverter
    from xh_model_zoo.xh_llm.models.base_converter import BaseConverter, HFTransfromersConverter
    from xh_model_zoo.xh_llm.models._qwen2 import Qwen2ConverterXH2a
    from xh_model_zoo.xh_llm.models._qwen3 import Qwen3ConverterXH2a
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    pytest.skip(f"Skipping tests due to missing dependencies: {e}", allow_module_level=True)


class TestConverterRegistry:
    """测试转换器注册机制"""

    def test_base_converter_class_exists(self):
        """测试 BaseConverter 类存在"""
        assert BaseConverter is not None
        assert hasattr(BaseConverter, 'xh1_hmonnx_compatible')

    def test_hf_converter_class_exists(self):
        """测试 HFTransfromersConverter 类存在"""
        assert HFTransfromersConverter is not None

    def test_qwen2_converter_exists(self):
        """测试 Qwen2ConverterXH2a 类存在"""
        assert Qwen2ConverterXH2a is not None

    def test_qwen3_converter_exists(self):
        """测试 Qwen3ConverterXH2a 类存在"""
        assert Qwen3ConverterXH2a is not None

    def test_converter_inheritance(self):
        """测试转换器继承关系"""
        assert issubclass(Qwen2ConverterXH2a, HFTransfromersConverter)
        assert issubclass(Qwen3ConverterXH2a, HFTransfromersConverter)


class TestConverterCompatibility:
    """测试转换器兼容性方法"""

    def test_xh1_hmonnx_compatible_inputs(self):
        """测试输入名称映射"""
        input_names = ["inputs_embeds", "past_seq_length", "current_input_length"]
        result = BaseConverter.xh1_hmonnx_compatible(input_names)
        
        assert "input_1" in result
        assert "valid_length" in result
        assert "current_length" in result

    def test_xh1_hmonnx_compatible_kvcache(self):
        """测试 KV Cache 名称映射"""
        input_names = ["past_key_cache_0", "past_value_cache_0"]
        result = BaseConverter.xh1_hmonnx_compatible(input_names)
        
        assert any("kcache_input" in name for name in result)
        assert any("vcache_input" in name for name in result)


class TestLLMConverter:
    """测试 LLMConverter 工厂类"""

    def test_llm_converter_exists(self):
        """测试 LLMConverter 类存在"""
        assert LLMConverter is not None
        assert hasattr(LLMConverter, 'from_pretrained')

    def test_from_pretrained_method_exists(self):
        """测试 from_pretrained 方法存在"""
        assert callable(getattr(LLMConverter, 'from_pretrained', None))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
