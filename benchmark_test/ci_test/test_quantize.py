import pytest
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from xh_model_zoo.xh_llm.quarot import quantizer_utils, rotation_utils, quant_utils
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    pytest.skip(f"Skipping tests due to missing dependencies: {e}", allow_module_level=True)


class TestQuarotQuantization:
    """测试 QuARot 量化功能"""

    def test_quarot_function_exists(self):
        """测试 quarot 函数存在"""
        assert hasattr(quantizer_utils, 'quarot')
        assert callable(quantizer_utils.quarot)

    def test_gptq_function_exists(self):
        """测试 gptq 函数存在"""
        assert hasattr(quantizer_utils, 'gptq')
        assert callable(quantizer_utils.gptq)


class TestRotationUtils:
    """测试旋转量化工具"""

    def test_random_orthogonal_matrix_exists(self):
        """测试随机正交矩阵生成函数存在"""
        assert hasattr(rotation_utils, 'random_orthogonal_matrix')
        assert callable(rotation_utils.random_orthogonal_matrix)

    def test_random_orthogonal_matrix_shape(self):
        """测试随机正交矩阵形状"""
        size = 64
        device = torch.device('cpu')
        Q = rotation_utils.random_orthogonal_matrix(size, device)
        
        assert Q.shape == (size, size)
        assert torch.allclose(torch.eye(size), Q @ Q.T, atol=1e-5)

    def test_get_orthogonal_matrix_exists(self):
        """测试获取正交矩阵函数存在"""
        assert hasattr(rotation_utils, 'get_orthogonal_matrix')
        assert callable(rotation_utils.get_orthogonal_matrix)

    def test_rotate_model_exists(self):
        """测试 rotate_model 函数存在"""
        assert hasattr(rotation_utils, 'rotate_model')
        assert callable(rotation_utils.rotate_model)

    def test_fuse_layer_norms_exists(self):
        """测试 fuse_layer_norms 函数存在"""
        assert hasattr(rotation_utils, 'fuse_layer_norms')
        assert callable(rotation_utils.fuse_layer_norms)


class TestQuantUtils:
    """测试量化工具"""

    def test_quant_utils_import(self):
        """测试 quant_utils 可导入"""
        assert quant_utils is not None


class TestQuantOps:
    """测试量化操作"""

    def test_quant_ops_import(self):
        """测试 quant_ops 可导入"""
        from xh_model_zoo.xh_llm.quarot import quant_ops
        assert quant_ops is not None

    def test_hadamard_utils_import(self):
        """测试 hadamard_utils 可导入"""
        from xh_model_zoo.xh_llm.quarot import hadamard_utils
        assert hadamard_utils is not None

    def test_get_hadK_exists(self):
        """测试 get_hadK 函数存在"""
        from xh_model_zoo.xh_llm.quarot.hadamard_utils import get_hadK
        assert callable(get_hadK)

    def test_matmul_hadU_exists(self):
        """测试 matmul_hadU 函数存在"""
        from xh_model_zoo.xh_llm.quarot.hadamard_utils import matmul_hadU
        assert callable(matmul_hadU)


class TestQuantizer:
    """测试量化器"""

    def test_quantizer_import(self):
        """测试 quantizer 可导入"""
        from xh_model_zoo.xh_llm.quarot import quantizer
        assert quantizer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
