import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from ..utils import get_root_logger
from .builder import DATASETS


@DATASETS.register_module()
class WikiTextDataset(Dataset):
    def __init__(self, split: str = "test", seq_len: int = 2048, data_dir: str = "./data/cache"):
        super().__init__()
        self._seq_len = seq_len
        self.logger = get_root_logger()
        self.logger.info("preparing dataset: wikitext")
        self.wiki_testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=split,
            cache_dir=data_dir,
            keep_in_memory=True,
        )
        self.input_ids = None

    @property
    def seq_len(self):
        return self._seq_len

    def init_dataset(self, tokenizer):
        self.logger.info("init dataset: wikitext")
        wiki_testdata = self.wiki_testdata

        full_input_ids = []
        for text in wiki_testdata["text"]:
            if len(text) == 0:
                continue
            input_ids = tokenizer(text + "\n\n", return_tensors="pt").input_ids  # [1, seq_len]
            full_input_ids.append(input_ids)
        self.input_ids = torch.cat(full_input_ids, dim=-1).squeeze(0)  # [seq_len]
        self.logger.info(f"dataset init finished, len: {len(self)}")

    def __len__(self):
        assert self.input_ids is not None, f"Please call init_dataset() first"
        return len(self.input_ids) // self.seq_len

    def __getitem__(self, idx):
        assert self.input_ids is not None, f"Please call init_dataset() first"
        return self.input_ids[idx * self.seq_len : (idx + 1) * self.seq_len]
