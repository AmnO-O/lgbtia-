import os
import re
import glob
from collections import Counter
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

from .config import ID_ORDER, SCOPE_DIM, TARGET_DIM, HATE2IDX, PipelineConfig


def safe_clean(text: Union[str, float]) -> str:
    """
    Cleans text by normalizing whitespace while carefully preserving
    accented and language-specific characters (e.g., Italian 'perché', 'è', Dutch).
    """
    text = str(text).lower()
    # Replace line breaks and tabs with space
    text = re.sub(r'[\n\t\r]+', ' ', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def encode_target(target_str: str) -> np.ndarray:
    """
    Encodes the target string into a 10-dimensional binary vector:
      - indices 0..8: identity bits ['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+']
      - index 9: scope bit (1.0 = 'group', 0.0 = 'individual')
    If 'none', returns an all-zero vector.
    """
    v = np.zeros(TARGET_DIM, dtype=np.float32)
    if not isinstance(target_str, str) or target_str.strip() == 'none' or target_str.strip() == '':
        return v
    parts = target_str.strip().split('_', 1)
    if len(parts) != 2:
        return v
    scope, ids = parts
    v[SCOPE_DIM] = 1.0 if scope == 'group' else 0.0
    for ident in ids.split(','):
        ident = ident.strip()
        if ident in ID_ORDER:
            v[ID_ORDER.index(ident)] = 1.0
    return v


def decode_target(v: Union[np.ndarray, torch.Tensor], thresh: float = 0.5) -> str:
    """
    Decodes a 10-dimensional binary vector into canonical SemEval string format.
    Returns 'none' if no identity bit meets the threshold.
    """
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    ids = [ID_ORDER[i] for i in range(SCOPE_DIM) if v[i] >= thresh]
    if not ids:
        return 'none'
    scope = 'group' if v[SCOPE_DIM] >= thresh else 'individual'
    return f"{scope}_" + ",".join(ids)


def build_vocab(texts: List[str], max_vocab_size: int = 30000) -> Dict[str, int]:
    """Builds a shared multi-lingual word-to-index vocabulary from tokenized text."""
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    common_words = word_counts.most_common(max_vocab_size - 2)
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, _) in enumerate(common_words):
        word_to_idx[word] = i + 2
    return word_to_idx


class StereoQueerDataset(Dataset):
    """Dataset for training scratch models (RNN, LSTM, Transformer) using word vocabulary."""
    def __init__(self, texts, st_labels, hs_labels, tg_labels, word_to_idx, max_len=256):
        self.texts = list(texts)
        self.st_labels = np.asarray(st_labels, dtype=np.float32)
        self.hs_labels = np.asarray(hs_labels, dtype=np.int64)
        self.tg_labels = np.stack(list(tg_labels)).astype(np.float32)
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = str(self.texts[idx]).split()
        ids = [self.word_to_idx.get(w, 1) for w in words]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.st_labels[idx], dtype=torch.float32),
            torch.tensor(self.hs_labels[idx], dtype=torch.long),
            torch.tensor(self.tg_labels[idx], dtype=torch.float32)
        )


class MMBertSeqDataset(Dataset):
    """
    Dataset using Pretrained mmBERT/ModernBERT AutoTokenizer.
    Encodes and caches all sequences once at initialization to maximize training throughput.
    """
    def __init__(self, texts, st_labels, hs_labels, tg_labels, tokenizer, max_len=256):
        self.st_labels = np.asarray(st_labels, dtype=np.float32)
        self.hs_labels = np.asarray(hs_labels, dtype=np.int64)
        self.tg_labels = np.asarray(tg_labels, dtype=np.float32)
        self.ids, self.mask = self._encode(list(texts), tokenizer, max_len)

    @staticmethod
    def _encode(texts, tokenizer, max_len):
        sep = getattr(tokenizer, 'sep_token', '[SEP]') or '[SEP]'
        processed_texts = [str(t).replace('[SEP]', sep) for t in texts]
        enc = tokenizer(
            processed_texts,
            truncation=True,
            max_length=max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask']

    def __len__(self):
        return len(self.st_labels)

    def __getitem__(self, idx):
        return (
            self.ids[idx],
            self.mask[idx],
            torch.tensor(self.st_labels[idx], dtype=torch.float32),
            torch.tensor(self.hs_labels[idx], dtype=torch.long),
            torch.tensor(self.tg_labels[idx], dtype=torch.float32)
        )


class DataPipeline:
    """Manages loading, parsing, group splitting, and data loader creation."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.df_all: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_val: Optional[pd.DataFrame] = None
        self.vocab: Optional[Dict[str, int]] = None
        self.tokenizer = None

    def find_data_files(self) -> List[str]:
        patterns = [
            os.path.join(self.config.data_dir, "*_training.tsv"),
            os.path.join(self.config.data_dir, "StereoQueerEval_*_training.tsv"),
            "data/*_training.tsv",
            "StereoQueerEval_*_training.tsv",
            "../LGBT/*_training.tsv",
            "LGBT/*_training.tsv",
        ]
        found = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
        return found

    def load_data(self, file_paths: Optional[List[str]] = None) -> pd.DataFrame:
        if file_paths is None:
            file_paths = self.find_data_files()
        
        if not file_paths:
            raise FileNotFoundError(
                f"No StereoQueerEval TSV files found in data directory '{self.config.data_dir}'. "
                f"Please ensure StereoQueerEval_{{EN,IT,NL}}_training.tsv are present."
            )

        dfs = []
        for path in file_paths:
            match = re.search(r'_([A-Z]{2})_training\.tsv$', path)
            lang = match.group(1) if match else "EN"
            # Note: quoting=1 is critical as comments/titles contain embedded newlines
            df = pd.read_csv(path, sep='\t', quoting=1)
            df['lang'] = lang
            dfs.append(df)

        self.df_all = pd.concat(dfs, ignore_index=True)

        # Build compound text: comment: <comment> [SEP] title: <title> [SEP] desc: <description>
        self.df_all['yt_title'] = self.df_all['yt_title'].fillna('')
        self.df_all['yt_description'] = self.df_all['yt_description'].fillna('')
        self.df_all['yt_comment'] = self.df_all['yt_comment'].fillna('')

        self.df_all['text'] = (
            'comment: ' + self.df_all['yt_comment'] + ' [SEP] ' +
            'title: ' + self.df_all['yt_title'] + ' [SEP] ' +
            'desc: ' + self.df_all['yt_description']
        ).map(safe_clean)

        # Labels
        self.df_all['st_y'] = (self.df_all['stereotype'] == 'yes').astype(np.float32)
        self.df_all['hs_y'] = self.df_all['hate_speech'].map(self.config.hate2idx).fillna(0).astype(np.int64)
        self.df_all['tg_y'] = [encode_target(t) for t in self.df_all['target']]

        return self.df_all

    def split_data(self, test_size: Optional[float] = None,
                   random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        GroupShuffleSplit by video title (`yt_title`) to prevent data leakage.
        Since video title and description are part of the input, the model must not
        evaluate on comments from videos seen during training.
        """
        if self.df_all is None:
            self.load_data()

        split_ratio = test_size if test_size is not None else self.config.val_split_ratio
        seed = random_state if random_state is not None else self.config.random_seed

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=split_ratio,
            random_state=seed
        )
        train_idx, val_idx = next(gss.split(self.df_all, groups=self.df_all['yt_title']))
        self.df_train = self.df_all.iloc[train_idx].reset_index(drop=True)
        self.df_val = self.df_all.iloc[val_idx].reset_index(drop=True)

        overlap = len(set(self.df_train['yt_title']) & set(self.df_val['yt_title']))
        if overlap > 0:
            print(f"Warning: Detected {overlap} overlapping video titles across splits!")
        return self.df_train, self.df_val

    def create_dataloaders(self, tokenizer=None) -> Tuple[DataLoader, DataLoader]:
        if self.df_train is None or self.df_val is None:
            self.split_data()

        if self.config.embed_source == 'mmbert':
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.config.mmbert_model_name)
            self.tokenizer = tokenizer

            train_ds = MMBertSeqDataset(
                self.df_train['text'].values,
                self.df_train['st_y'].values,
                self.df_train['hs_y'].values,
                self.df_train['tg_y'].values,
                tokenizer,
                max_len=self.config.max_length
            )
            val_ds = MMBertSeqDataset(
                self.df_val['text'].values,
                self.df_val['st_y'].values,
                self.df_val['hs_y'].values,
                self.df_val['tg_y'].values,
                tokenizer,
                max_len=self.config.max_length
            )
        else:
            self.vocab = build_vocab(self.df_train['text'].values, max_vocab_size=self.config.vocab_size)
            train_ds = StereoQueerDataset(
                self.df_train['text'].values,
                self.df_train['st_y'].values,
                self.df_train['hs_y'].values,
                self.df_train['tg_y'].values,
                self.vocab,
                max_len=self.config.max_length
            )
            val_ds = StereoQueerDataset(
                self.df_val['text'].values,
                self.df_val['st_y'].values,
                self.df_val['hs_y'].values,
                self.df_val['tg_y'].values,
                self.vocab,
                max_len=self.config.max_length
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        return train_loader, val_loader
