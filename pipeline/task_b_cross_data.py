"""
Task B Dual-Stream Dataset and Tokenizer Utilities.

Encodes inputs into two distinct branches:
1. TC Stream: [CLS] title: <yt_title> [SEP] comment: <yt_comment> [SEP]
   - Generates precise Role IDs: 1 for Title tokens, 2 for Comment tokens, 0 for Padding.
2. Desc Stream: [CLS] desc: <yt_description> [SEP]
   - Generates pure Description token IDs and attention mask.
   - Supports Context Dropout (Review #17): randomly replaces Description with empty string during training
     with probability p (default 0.20) to prevent the model from shortcutting on video descriptions.
"""

from typing import List, Tuple, Dict, Optional, Union, Any
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import HATE2IDX, PipelineConfig
from .data import safe_clean, DataPipeline

ROLE_PAD = 0
ROLE_TITLE = 1
ROLE_COMMENT = 2
ROLE_SPECIAL = 3


class TaskBDualStreamDataset(Dataset):
    """
    PyTorch Dataset providing paired Dual-Stream inputs for Task B:
    - tc_input_ids, tc_attention_mask, tc_role_ids
    - desc_input_ids, desc_attention_mask
    - hate_speech_labels
    """
    def __init__(
        self,
        titles: List[str],
        comments: List[str],
        descriptions: List[str],
        labels: Optional[List[int]],
        tokenizer: Any,
        tc_max_len: int = 128,
        desc_max_len: int = 128,
        context_dropout_prob: float = 0.0,
        is_train: bool = False
    ):
        self.titles = list(titles)
        self.comments = list(comments)
        self.descriptions = list(descriptions)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.tc_max_len = tc_max_len
        self.desc_max_len = desc_max_len
        self.context_dropout_prob = context_dropout_prob
        self.is_train = is_train

        self.cls_token_id = getattr(tokenizer, 'cls_token_id', None) or getattr(tokenizer, 'bos_token_id', 0)
        self.sep_token_id = getattr(tokenizer, 'sep_token_id', None) or getattr(tokenizer, 'eos_token_id', 2)
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 0) if getattr(tokenizer, 'pad_token_id', None) is not None else 0

    def __len__(self) -> int:
        return len(self.titles)

    def _build_tc_stream(self, title_str: str, comment_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tokenizes Title and Comment independently, assigns structural role IDs, and concatenates with [CLS] and [SEP].
        """
        title_text = "title: " + safe_clean(title_str)
        comment_text = "comment: " + safe_clean(comment_str)

        title_ids = self.tokenizer.encode(title_text, add_special_tokens=False)
        comment_ids = self.tokenizer.encode(comment_text, add_special_tokens=False)

        # Budget: Reserve 3 tokens for [CLS], [SEP], [SEP]
        max_content = self.tc_max_len - 3
        
        # Fair length budgeting: Allocate max 48 tokens to title, remainder to comment
        max_title_len = min(len(title_ids), 48)
        max_comm_len = min(len(comment_ids), max_content - max_title_len)
        
        # If comment still has leftover budget, title can expand if needed
        if len(comment_ids) < (max_content - max_title_len):
            max_title_len = min(len(title_ids), max_content - len(comment_ids))

        t_ids_trunc = title_ids[:max_title_len]
        c_ids_trunc = comment_ids[:max_comm_len]

        # Assemble token sequence: [CLS] + title_ids + [SEP] + comment_ids + [SEP]
        full_ids = [self.cls_token_id] + t_ids_trunc + [self.sep_token_id] + c_ids_trunc + [self.sep_token_id]
        role_seq = [ROLE_SPECIAL] + [ROLE_TITLE] * len(t_ids_trunc) + [ROLE_SPECIAL] + [ROLE_COMMENT] * len(c_ids_trunc) + [ROLE_SPECIAL]

        # Padding / Truncation
        seq_len = len(full_ids)
        if seq_len < self.tc_max_len:
            pad_len = self.tc_max_len - seq_len
            input_ids = full_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len
            role_ids = role_seq + [ROLE_PAD] * pad_len
        else:
            input_ids = full_ids[:self.tc_max_len]
            attention_mask = [1] * self.tc_max_len
            role_ids = role_seq[:self.tc_max_len]

        return (
            np.array(input_ids, dtype=np.int64),
            np.array(attention_mask, dtype=np.int64),
            np.array(role_ids, dtype=np.int64)
        )

    def _build_desc_stream(self, desc_str: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenizes Description in isolation: [CLS] desc: <desc> [SEP]
        """
        cleaned_desc = safe_clean(desc_str)
        desc_text = "desc: " + cleaned_desc if cleaned_desc else "desc: none"
        desc_ids = self.tokenizer.encode(desc_text, add_special_tokens=False)

        max_desc_content = self.desc_max_len - 2
        d_ids_trunc = desc_ids[:max_desc_content]

        full_ids = [self.cls_token_id] + d_ids_trunc + [self.sep_token_id]
        seq_len = len(full_ids)

        if seq_len < self.desc_max_len:
            pad_len = self.desc_max_len - seq_len
            input_ids = full_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len
        else:
            input_ids = full_ids[:self.desc_max_len]
            attention_mask = [1] * self.desc_max_len

        return (
            np.array(input_ids, dtype=np.int64),
            np.array(attention_mask, dtype=np.int64)
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        title = self.titles[idx]
        comment = self.comments[idx]
        desc = self.descriptions[idx]

        # Context Dropout (Review #17):
        # With probability p during training, replace Description with empty string
        # to ensure the model treats Description as auxiliary context, not a shortcut.
        if self.is_train and self.context_dropout_prob > 0.0:
            if random.random() < self.context_dropout_prob:
                desc = ""

        tc_ids, tc_mask, tc_roles = self._build_tc_stream(title, comment)
        desc_ids, desc_mask = self._build_desc_stream(desc)

        tensors = (
            torch.tensor(tc_ids, dtype=torch.long),
            torch.tensor(tc_mask, dtype=torch.long),
            torch.tensor(tc_roles, dtype=torch.long),
            torch.tensor(desc_ids, dtype=torch.long),
            torch.tensor(desc_mask, dtype=torch.long),
        )

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return (*tensors, label_tensor)
        return tensors


def create_task_b_dual_stream_loaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    tokenizer: Any,
    batch_size: int = 32,
    tc_max_len: int = 128,
    desc_max_len: int = 128,
    context_dropout_prob: float = 0.20,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function creating training and validation DataLoaders for the Dual-Stream Architecture.
    Applies Context Dropout with probability context_dropout_prob (default 0.20) during training.
    """
    train_ds = TaskBDualStreamDataset(
        titles=df_train['yt_title'].values,
        comments=df_train['yt_comment'].values,
        descriptions=df_train['yt_description'].values,
        labels=df_train['hs_y'].values,
        tokenizer=tokenizer,
        tc_max_len=tc_max_len,
        desc_max_len=desc_max_len,
        context_dropout_prob=context_dropout_prob,
        is_train=True
    )

    val_ds = TaskBDualStreamDataset(
        titles=df_val['yt_title'].values,
        comments=df_val['yt_comment'].values,
        descriptions=df_val['yt_description'].values,
        labels=df_val['hs_y'].values,
        tokenizer=tokenizer,
        tc_max_len=tc_max_len,
        desc_max_len=desc_max_len,
        context_dropout_prob=0.0,
        is_train=False
    )

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )

    return train_loader, val_loader
