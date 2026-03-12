import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
benchmark_test_path = current_dir.parent
xh2_model_zoo_path = benchmark_test_path.parent
xh_model_zoo_path = xh2_model_zoo_path / 'xh_model_zoo'
sys.path.insert(0, str(xh2_model_zoo_path))
sys.path.insert(0, str(xh_model_zoo_path))
print(f"conftest.py: Added to sys.path: {xh2_model_zoo_path}")
print(f"conftest.py: Added to sys.path: {xh_model_zoo_path}")
