import re
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PipelineConfig
from .data import safe_clean
from .models.task_b_class_aware import ROLE_PAD, ROLE_TITLE, ROLE_DESC, ROLE_COMMENT

class TaskBRoleDataset(Dataset):
    """
    Dataset tailored for Task B Architecture with Explicit Role Injection:
      [INPUT SEQUENCE]
      [CLS] comment: <C> ... </C> [SEP] title: <T> ... </T> [SEP] desc: <D> ... </D> [SEP]
      
      Produces:
        input_ids: [S]
        attention_mask: [S]
        role_ids: [S] (0=PAD/Special, 1=Title, 2=Description, 3=Comment)
        hs_label: [1] (0='no', 1='yes_implicit', 2='yes_explicit')
        st_label: [1] (float)
        tg_label: [10] (float bitmask)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = 256
    ):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        self.st_labels = df['st_y'].values.astype(np.float32) if 'st_y' in df.columns else np.zeros(len(df), dtype=np.float32)
        self.hs_labels = df['hs_y'].values.astype(np.int64) if 'hs_y' in df.columns else np.zeros(len(df), dtype=np.int64)
        self.tg_labels = np.stack(df['tg_y'].values).astype(np.float32) if 'tg_y' in df.columns else np.zeros((len(df), 10), dtype=np.float32)
        
        # Tokenize and build aligned role IDs
        self.input_ids, self.attention_mask, self.role_ids = self._tokenize_with_roles(
            titles=df['yt_title'].fillna('').tolist(),
            descriptions=df['yt_description'].fillna('').tolist(),
            comments=df['yt_comment'].fillna('').tolist(),
            tokenizer=tokenizer,
            max_len=max_len
        )

    @staticmethod
    def _tokenize_with_roles(
        titles: List[str],
        descriptions: List[str],
        comments: List[str],
        tokenizer,
        max_len: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_input_ids = []
        all_attention_masks = []
        all_role_ids = []

        cls_id = getattr(tokenizer, 'cls_token_id', None)
        if cls_id is None:
            cls_id = getattr(tokenizer, 'bos_token_id', 101) or 101

        sep_id = getattr(tokenizer, 'sep_token_id', None)
        if sep_id is None:
            sep_id = getattr(tokenizer, 'eos_token_id', 102) or 102

        pad_id = getattr(tokenizer, 'pad_token_id', 0) or 0

        for title, desc, comment in zip(titles, descriptions, comments):
            clean_c = safe_clean(comment)
            clean_t = safe_clean(title)
            clean_d = safe_clean(desc)

            # Tokenize segments individually with prompt role tags without special tokens
            c_ids = tokenizer.encode(f"comment: {clean_c}", add_special_tokens=False) if clean_c else []
            t_ids = tokenizer.encode(f"title: {clean_t}", add_special_tokens=False) if clean_t else []
            d_ids = tokenizer.encode(f"desc: {clean_d}", add_special_tokens=False) if clean_d else []

            # Structure: [CLS] comment: <comment> [SEP] title: <title> [SEP] desc: <description> [SEP]
            # Budget tokens prioritizing comment (up to 70%), title (up to 18%), desc (remainder)
            overhead = 4  # [CLS], 3x [SEP]
            available = max(10, max_len - overhead)
            
            c_budget = int(available * 0.70)
            t_budget = int(available * 0.18)
            d_budget = available - c_budget - t_budget

            c_ids = c_ids[:c_budget]
            t_ids = t_ids[:t_budget]
            d_ids = d_ids[:d_budget]

            seq_ids = [cls_id]
            seq_roles = [ROLE_PAD]

            # 1. Comment (Primary Target)
            if c_ids:
                seq_ids.extend(c_ids)
                seq_roles.extend([ROLE_COMMENT] * len(c_ids))
            seq_ids.append(sep_id)
            seq_roles.append(ROLE_PAD)

            # 2. Title (Video Context)
            if t_ids:
                seq_ids.extend(t_ids)
                seq_roles.extend([ROLE_TITLE] * len(t_ids))
            seq_ids.append(sep_id)
            seq_roles.append(ROLE_PAD)

            # 3. Description (Background Context)
            if d_ids:
                seq_ids.extend(d_ids)
                seq_roles.extend([ROLE_DESC] * len(d_ids))
            seq_ids.append(sep_id)
            seq_roles.append(ROLE_PAD)

            # Pad or truncate
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

    def __len__(self):
        return len(self.hs_labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.role_ids[idx],
            torch.tensor(self.st_labels[idx], dtype=torch.float32),
            torch.tensor(self.hs_labels[idx], dtype=torch.long),
            torch.tensor(self.tg_labels[idx], dtype=torch.float32),
        )
