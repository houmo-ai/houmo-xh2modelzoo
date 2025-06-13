#!/bin/bash
WORK_DIR=$(dirname $(readlink -f "$0"))
unset PYTHONPATH
export PYTHONPATH=$WORK_DIR:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate xhquant
