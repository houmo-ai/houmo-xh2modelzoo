import pytest
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from xhquant.api import export_onnx, to_export_graph, to_export_hmonnx, to_export_hmonnx_v2
    from xhquant.api import ExportedGraph, QuantGraph, FrontendGraph
    from xh_model_zoo.xh_llm.models.base_model import BaseModel
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    pytest.skip(f"Skipping tests due to missing dependencies: {e}", allow_module_level=True)
class TestExportFunctions:
    """测试导出功能"""

    def test_export_onnx_import(self):
        """测试 export_onnx 可从 xhquant 导入"""
        try:
            from xhquant.api import export_onnx
            assert callable(export_onnx)
        except ImportError:
            pytest.skip("xhquant not available")

    def test_to_export_graph_import(self):
        """测试 to_export_graph 可导入"""
        try:
            from xhquant.api import to_export_graph
            assert callable(to_export_graph)
        except ImportError:
            pytest.skip("xhquant not available")

    def test_to_export_hmonnx_import(self):
        """测试 to_export_hmonnx 可导入"""
        try:
            from xhquant.api import to_export_hmonnx
            assert callable(to_export_hmonnx)
        except ImportError:
            pytest.skip("xhquant not available")

    def test_to_export_hmonnx_v2_import(self):
        """测试 to_export_hmonnx_v2 可导入"""
        try:
            from xhquant.api import to_export_hmonnx_v2
            assert callable(to_export_hmonnx_v2)
        except ImportError:
            pytest.skip("xhquant not available")


class TestBaseModelExport:
    """测试 BaseModel 导出方法"""

    def test_base_model_import(self):
        """测试 BaseModel 可导入"""
        from xh_model_zoo.xh_llm.models.base_model import BaseModel
        assert BaseModel is not None

    def test_convert_to_export_graph_method_exists(self):
        """测试 convert_to_export_graph 方法存在"""
        from xh_model_zoo.xh_llm.models.base_model import BaseModel
        assert hasattr(BaseModel, 'convert_to_export_graph')
        assert callable(getattr(BaseModel, 'convert_to_export_graph'))

    def test_to_export_onnx_method_exists(self):
        """测试 to_export_onnx 方法存在"""
        from xh_model_zoo.xh_llm.models.base_model import BaseModel
        assert hasattr(BaseModel, 'to_export_onnx')
        assert callable(getattr(BaseModel, 'to_export_onnx'))


class TestExportedGraph:
    """测试 ExportedGraph 类型"""

    def test_exported_graph_import(self):
        """测试 ExportedGraph 可导入"""
        try:
            from xhquant.api import ExportedGraph
            assert ExportedGraph is not None
        except ImportError:
            pytest.skip("xhquant not available")


class TestQuantGraph:
    """测试 QuantGraph 类型"""

    def test_quant_graph_import(self):
        """测试 QuantGraph 可导入"""
        try:
            from xhquant.api import QuantGraph
            assert QuantGraph is not None
        except ImportError:
            pytest.skip("xhquant not available")


class TestFrontendGraph:
    """测试 FrontendGraph 类型"""

    def test_frontend_graph_import(self):
        """测试 FrontendGraph 可导入"""
        try:
            from xhquant.api import FrontendGraph
            assert FrontendGraph is not None
        except ImportError:
            pytest.skip("xhquant not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
