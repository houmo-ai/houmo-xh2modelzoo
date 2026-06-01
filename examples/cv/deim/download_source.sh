#!/bin/bash
base_dir=$(dirname "$0")
git clone https://github.com/ShihuaHuang95/DEIM.git $base_dir/DEIM
cd $base_dir/DEIM
git checkout c7ed52d
git apply --check ../deim_c7ed52d.patch
# 应用补丁
git apply ../deim_c7ed52d.patch