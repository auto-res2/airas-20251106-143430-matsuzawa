"""src/preprocess.py
Data loading & preprocessing utilities for GLUE-style classification tasks.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

CACHE_DIR = ".cache/"


class GLUEDataModule:
    """Lightweight data-module wrapper around HF *datasets* for GLUE tasks.
    Produces *train_loader* and *val_loader* attributes.
    """

    def __init__(self, cfg, tokenizer: PreTrainedTokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.batch_size = cfg.dataset.batch_size

        task_name = cfg.dataset.name
        if task_name.startswith("glue_"):
            self.hf_task = task_name.split("_", 1)[1]
            self.dataset = load_dataset("glue", self.hf_task, cache_dir=CACHE_DIR)
        else:
            # arbitrary HF dataset name – assume standard train / validation splits
            self.hf_task = task_name
            self.dataset = load_dataset(task_name, cache_dir=CACHE_DIR)

        # keys to tokenize
        sent1_key, sent2_key = TASK_TO_KEYS.get(self.hf_task, ("sentence1", "sentence2"))
        self.num_labels = self.dataset["train"].features["label"].num_classes

        def _tokenize(batch):
            if sent2_key is None:
                return self.tokenizer(
                    batch[sent1_key],
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.dataset.max_length,
                )
            else:
                return self.tokenizer(
                    batch[sent1_key],
                    batch[sent2_key],
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.dataset.max_length,
                )

        self.dataset = self.dataset.map(_tokenize, batched=True, remove_columns=[col for col in self.dataset["train"].column_names if col not in ["label"]])
        self.dataset.set_format(type="torch")

        # pick validation split key
        val_split = "validation_matched" if "validation_matched" in self.dataset else "validation"
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset[val_split]

        collator = DataCollatorWithPadding(self.tokenizer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collator)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator)