import re
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PipelineConfig
from .data import safe_clean
from .models.task_b_class_aware import (
    ROLE_PAD, ROLE_TITLE, ROLE_DESC, ROLE_COMMENT, ROLE_HINT
)

class TaskBAdditiveLatentDataset(Dataset):
    """
    Production-grade Dataset for Task B with Additive Latent Privileged Guidance:
    
    Yields 2 parallel input streams per sample:
      1. Primary Stream (Unconditional / Evaluation Path):
         [CLS] comment: <C> [SEP] title: <T> [SEP] desc: <D> [SEP]
         -> Returns: (input_ids_u, attention_mask_u, role_ids_u)
         
      2. Privileged Hint Stream (Conditional / Teacher Path):
         [CLS] hint: <LLM Diagnostic Rationale> [SEP]
         -> Returns: (hint_ids, hint_mask)
         
      3. Ground Truth Labels:
         -> st_label [1], hs_label [1], tg_label [10]
         
    Key Advantages:
      - Primary stream is 100% stable: Comment, Title, Description keep 100% of the 256-token budget.
      - If hint is empty or None, (hint_ids, hint_mask) contains pure PAD tokens.
      - Training with hint=None is mathematically and physically IDENTICAL to baseline training.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = 256,
        hint_max_len: int = 64,
        hint_col: str = 'hint'
    ):
        self.df = df
        self.max_len = max_len
        self.hint_max_len = hint_max_len
        self.tokenizer = tokenizer
        self.hint_col = hint_col

        self.st_labels = df['st_y'].values.astype(np.float32) if 'st_y' in df.columns else np.zeros(len(df), dtype=np.float32)
        self.hs_labels = df['hs_y'].values.astype(np.int64) if 'hs_y' in df.columns else np.zeros(len(df), dtype=np.int64)
        self.tg_labels = np.stack(df['tg_y'].values).astype(np.float32) if 'tg_y' in df.columns else np.zeros((len(df), 10), dtype=np.float32)

        # 1. Tokenize Primary Stream (Unguided: Comment + Title + Description)
        self.input_ids_u, self.attention_mask_u, self.role_ids_u = self._tokenize_primary(
            titles=df['yt_title'].fillna('').tolist(),
            descriptions=df['yt_description'].fillna('').tolist(),
            comments=df['yt_comment'].fillna('').tolist(),
            tokenizer=tokenizer,
            max_len=max_len
        )

        # 2. Tokenize Teacher Hint Stream (Privileged Information)
        raw_hints = df[hint_col].fillna('').tolist() if hint_col in df.columns else [''] * len(df)
        self.hint_ids, self.hint_mask = self._tokenize_hints(
            hints=raw_hints,
            tokenizer=tokenizer,
            max_len=hint_max_len
        )

    @staticmethod
    def _tokenize_primary(
        titles: List[str],
        descriptions: List[str],
        comments: List[str],
        tokenizer,
        max_len: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_input_ids = []
        all_attention_masks = []
        all_role_ids = []

        cls_id = getattr(tokenizer, 'cls_token_id', None) or getattr(tokenizer, 'bos_token_id', 101) or 101
        sep_id = getattr(tokenizer, 'sep_token_id', None) or getattr(tokenizer, 'eos_token_id', 102) or 102
        pad_id = getattr(tokenizer, 'pad_token_id', 0) or 0

        for title, desc, comment in zip(titles, descriptions, comments):
            clean_c = safe_clean(comment)
            clean_t = safe_clean(title)
            clean_d = safe_clean(desc)

            c_ids = tokenizer.encode(f"comment: {clean_c}", add_special_tokens=False) if clean_c else []
            t_ids = tokenizer.encode(f"title: {clean_t}", add_special_tokens=False) if clean_t else []
            d_ids = tokenizer.encode(f"desc: {clean_d}", add_special_tokens=False) if clean_d else []

            overhead = 4  # [CLS] + 3x [SEP]
            available = max(10, max_len - overhead)

            c_budget = int(available * 0.70)
            t_budget = int(available * 0.18)
            d_budget = available - c_budget - t_budget

            c_ids = c_ids[:c_budget]
            t_ids = t_ids[:t_budget]
            d_ids = d_ids[:d_budget]

            seq_ids = [cls_id]
            seq_roles = [ROLE_PAD]

            # 1. Comment
            if c_ids:
                seq_ids.extend(c_ids)
                seq_roles.extend([ROLE_COMMENT] * len(c_ids))
            seq_ids.append(sep_id)
            seq_roles.append(ROLE_PAD)

            # 2. Title
            if t_ids:
                seq_ids.extend(t_ids)
                seq_roles.extend([ROLE_TITLE] * len(t_ids))
            seq_ids.append(sep_id)
            seq_roles.append(ROLE_PAD)

            # 3. Description
            if d_ids:
                seq_ids.extend(d_ids)
                seq_roles.extend([ROLE_DESC] * len(d_ids))
            seq_ids.append(sep_id)
            seq_roles.append(ROLE_PAD)

            if len(seq_ids) > max_len:
                seq_ids = seq_ids[:max_len]
                seq_roles = seq_roles[:max_len]
                mask = [1] * max_len
            else:
                pad_len = max_len - len(seq_ids)
                mask = [1] * len(seq_ids) + [0] * pad_len
                seq_ids = seq_ids + [pad_id] * pad_len
                seq_roles = seq_roles + [ROLE_PAD] * pad_len

            all_input_ids.append(seq_ids)
            all_attention_masks.append(mask)
            all_role_ids.append(seq_roles)

        return (
            torch.tensor(all_input_ids, dtype=torch.long),
            torch.tensor(all_attention_masks, dtype=torch.long),
            torch.tensor(all_role_ids, dtype=torch.long),
        )

    @staticmethod
    def _tokenize_hints(
        hints: List[str],
        tokenizer,
        max_len: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_hint_ids = []
        all_hint_masks = []

        cls_id = getattr(tokenizer, 'cls_token_id', None) or getattr(tokenizer, 'bos_token_id', 101) or 101
        sep_id = getattr(tokenizer, 'sep_token_id', None) or getattr(tokenizer, 'eos_token_id', 102) or 102
        pad_id = getattr(tokenizer, 'pad_token_id', 0) or 0

        for hint in hints:
            clean_h = safe_clean(hint)
            if not clean_h:
                all_hint_ids.append([pad_id] * max_len)
                all_hint_masks.append([0] * max_len)
                continue

            h_ids = tokenizer.encode(f"hint: {clean_h}", add_special_tokens=False)
            h_ids = h_ids[:max_len - 2]
            seq_ids = [cls_id] + h_ids + [sep_id]
            
            pad_len = max_len - len(seq_ids)
            mask = [1] * len(seq_ids) + [0] * pad_len
            seq_ids = seq_ids + [pad_id] * pad_len

            all_hint_ids.append(seq_ids)
            all_hint_masks.append(mask)

        return (
            torch.tensor(all_hint_ids, dtype=torch.long),
            torch.tensor(all_hint_masks, dtype=torch.long),
        )

    def __len__(self):
        return len(self.hs_labels)

    def __getitem__(self, idx):
        return (
            self.input_ids_u[idx],
            self.attention_mask_u[idx],
            self.role_ids_u[idx],
            self.hint_ids[idx],
            self.hint_mask[idx],
            torch.tensor(self.st_labels[idx], dtype=torch.float32),
            torch.tensor(self.hs_labels[idx], dtype=torch.long),
            torch.tensor(self.tg_labels[idx], dtype=torch.float32),
        )

# Backward compatibility alias
TaskBRoleDataset = TaskBAdditiveLatentDataset
