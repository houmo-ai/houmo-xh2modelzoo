from torch.utils.data import Dataset
import json
from pathlib import Path
from ..registry import DATASETS


def format_qwen3_vl_dataset_for_calibration(item):
    content = []
    for p_input in item["struct"]:
        content.append({"type": p_input["type"], p_input["type"]: p_input["value"]})
    format_item = [{"role": "user", "content": content}, {"role": "assistant", "content": item["response"]}]
    return format_item


@DATASETS.register_module()
class VLLMCustomDataset(Dataset):
    def __init__(self, data_files: list[str]):
        super().__init__()
        self.data_files = data_files

        self.data = []

        for data_file in data_files:
            with open(data_file, "r") as f:
                data = json.load(f)
            for item in data:
                formatted = format_qwen3_vl_dataset_for_calibration(item)
                for message in formatted:
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    for entry in content:
                        if entry.get("type") == "image" and isinstance(entry.get("image"), str):
                            entry["image"] = str(Path(entry["image"]).expanduser())
                self.data.append(formatted)

        self.data = sorted(self.data, key=lambda x: len(x[1]["content"]), reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
