# Gr00t Model Setup Guide

## Overview
This directory contains the Gr00t model implementation for robotics and vision tasks.

## Setup Instructions

### 1. Clone the Repository

The Gr00t model repository is already cloned at:
```
/data01/home/xuchen/xh2/xh2_model_zoo/xh_model_zoo/xh_llm/models/groot/gr00t
```

If you need to re-clone or update the repository:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
```

### 2. Soft Link Setup

The Gr00t model is already soft-linked to the correct location:
```
ln -s Isaac-GR00T/gr00t xh_model_zoo/xh_llm/models/groot/gr00t
```

The current repository is already in the correct location, so no additional soft linking is needed.

### 3. Install Dependencies

Install the required version of transformers:

```bash
conda activate paddle
pip install transformers==4.51.3
```


## Usage

After setup, you can use the Gr00t model in your code:

```python
from xh_model_zoo.xh_llm.models.groot import Cus_Groot

# Initialize the model
model = Cus_Groot.from_pretrained("path/to/model")
```

## Troubleshooting

### Transformers Version Issues
Ensure you have the correct version of transformers installed:
```bash
pip install transformers==4.51.3
```

### Submodule Issues
If you encounter issues with external dependencies, try:
```bash
git submodule update --init --recursive
git submodule update --remote
```

## Additional Resources

- Check the main repository documentation for detailed usage instructions
- Refer to the example scripts in the `examples/` directory
- See `configs/` for model configuration options

## Evaluation
- refer: https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/robocasa-gr1-tabletop-tasks/README.md
