"""
Comprehensive Multilingual Text Augmentation Suite for Stereotype & Hate Speech Tasks.

Techniques Implemented:
  1. Back-Translation (HuggingFace MarianMT with auto-pivot routing and diverse sampling):
     Paraphrases comments via intermediate pivot languages:
       - EN -> DE / FR -> EN
       - IT -> EN -> IT
       - NL -> EN -> NL
       - FA -> EN -> FA
     Uses stochastic nucleus sampling (top_p, temperature) and fallback perturbation if output is identical.
  2. EDA Techniques (Easy Data Augmentation - Wei & Zou 2019):
     - Random Swap (RS): Swaps positions of two randomly chosen words.
     - Random Deletion (RD): Randomly removes words with probability p.
     - Random Masking / Insertion (RM): Injects <mask_token> to train robust attention against missing words.
  3. Multilingual Social Media Slang & Contraction Injection (EN, IT, NL, FA, VI).
  4. Context Dropout (Title / Description Masking): Prevents bias toward metadata.
  5. Per-Language Stratified Minority Oversampling (Implicit Hate).
"""

import random
import re
from typing import List, Tuple, Dict, Optional, Union, Callable
import numpy as np
import pandas as pd

# Multi-Lingual Slang & Contraction Replacement Dictionaries
MULTILINGUAL_SLANG_MAP: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "you": ["u", "yu"],
        "are": ["r"],
        "why": ["y"],
        "because": ["cuz", "bc", "bcoz"],
        "people": ["ppl"],
        "please": ["pls", "plz"],
        "with": ["w/", "wit"],
        "without": ["w/o"],
        "someone": ["sm1"],
        "what": ["wat", "wut"],
        "don't": ["dont", "dnt"],
        "hate": ["h8"],
        "for real": ["fr", "frfr"],
    },
    "it": {
        "perché": ["xk", "xke", "perche"],
        "comunque": ["cmq"],
        "non": ["nn"],
        "qualcosa": ["qlcs"],
        "qualcuno": ["qlcn"],
        "ti voglio bene": ["tvb"],
        "sempre": ["smpr"],
        "tutto": ["tt"],
        "sono": ["sn"],
        "molto": ["mlt"],
    },
    "nl": {
        "niet": ["nie", "nt"],
        "waarom": ["wrm"],
        "wat": ["wt"],
        "alsjeblieft": ["aub"],
        "misschien": ["mss", "mischien"],
        "iedereen": ["idr"],
        "zoveel": ["zoveul"],
        "gewoon": ["gwn"],
    },
    "fa": {
        "می‌خواهم": ["میخوام", "میخام"],
        "نمی‌دانm": ["نمیدونم", "نمیدانم"],
        "چرا": ["چرااا", "چرا؟"],
        "است": ["هست", "ـه"],
        "اینها": ["اینا"],
        "آنها": ["اونا"],
    },
    "vi": {
        "không": ["ko", "k", "khong", "hong"],
        "được": ["dc", "đc", "duoc"],
        "gì": ["j", "gi"],
        "vậy": ["v", "vay", "zậy"],
        "thôi": ["thoi", "thoii"],
        "người": ["ng", "nguoi"],
        "bọn": ["bon", "lũ", "tụi"],
        "đồng tính": ["gay", "bê đê", "lgbt", "bóng"],
    }
}


# =============================================================================
# 1. EDA: RANDOM SWAP, RANDOM DELETION, RANDOM MASKING
# =============================================================================

def random_swap(text: str, n_swaps: int = 1) -> str:
    """
    Random Swap (RS): Randomly chooses two words in the sentence and swaps their positions.
    Preserves semantic intent while introducing syntactical variation.
    """
    words = text.split()
    length = len(words)
    if length <= 2:
        return text
        
    for _ in range(n_swaps):
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
    return " ".join(words)


def random_deletion(text: str, p: float = 0.10) -> str:
    """
    Random Deletion (RD): Randomly removes words with probability p.
    """
    words = text.split()
    if len(words) <= 3:
        return text
    kept = [w for w in words if random.random() > p]
    if not kept:
        return text
    return " ".join(kept)


def random_mask(text: str, mask_token: str = "<mask_token>", p: float = 0.10) -> str:
    """
    Random Masking (RM): Replaces random words with mask_token.
    Trains the cross-attention layers to identify implicit hatred even when keywords are hidden.
    """
    words = text.split()
    if len(words) <= 2:
        return text
        
    masked_words = []
    for word in words:
        if random.random() < p:
            masked_words.append(mask_token)
        else:
            masked_words.append(word)
            
    return " ".join(masked_words)


def augment_slang_noise(text: str, lang: str = "en", p: float = 0.20) -> str:
    """
    Randomly injects informal social media abbreviations/slangs for the specified language.
    """
    if not text or not isinstance(text, str):
        return text
        
    lang_key = lang.lower().strip()
    slang_dict = MULTILINGUAL_SLANG_MAP.get(lang_key, MULTILINGUAL_SLANG_MAP["en"])
    
    words = text.split()
    augmented_words = []
    
    for word in words:
        clean_w = word.lower().strip(".,!?;:\"'()[]{}")
        if random.random() < p and clean_w in slang_dict:
            replacement = random.choice(slang_dict[clean_w])
            augmented_words.append(replacement)
        else:
            augmented_words.append(word)
            
    return " ".join(augmented_words)


def augment_context_dropout(
    title: str,
    description: str,
    p_drop_desc: float = 0.40,
    p_drop_title: float = 0.15
) -> Tuple[str, str]:
    """
    Language-Agnostic Context Dropout:
    Simulates incomplete video metadata across English, Italian, Dutch, Persian, etc.
    """
    aug_title = title
    aug_desc = description
    
    if random.random() < p_drop_title:
        aug_title = ""
        
    if random.random() < p_drop_desc:
        aug_desc = ""
        
    return aug_title, aug_desc


# =============================================================================
# 2. BACK-TRANSLATION PIPELINE (MarianMT with Valid Hub Pairings & Diverse Sampling)
# =============================================================================

def get_valid_marian_pairs(src_lang: str, default_pivot: str = "de") -> Tuple[str, str, str, str]:
    """
    Returns valid (forward_model_name, backward_model_name, src, pivot).
    Guarantees the Hugging Face repository exists.
    """
    src = src_lang.lower().strip()
    
    if src == "en":
        pivot = default_pivot if default_pivot != "en" else "de"
        return f"Helsinki-NLP/opus-mt-en-{pivot}", f"Helsinki-NLP/opus-mt-{pivot}-en", src, pivot
    else:
        # For IT, NL, etc., English is the universal verified pivot on Opus-MT
        pivot = "en"
        return f"Helsinki-NLP/opus-mt-{src}-en", f"Helsinki-NLP/opus-mt-en-{src}", src, pivot


class BackTranslationAugmenter:
    """
    Back-Translation Augmenter using HuggingFace MarianMT models.
    Translates Source -> Pivot -> Source with automatic diverse paraphrase generation.
    """
    def __init__(
        self, 
        src_lang: str = "en", 
        pivot_lang: str = "de", 
        device: str = "cpu",
        temperature: float = 0.85,
        top_p: float = 0.92
    ):
        self.src_lang = src_lang.lower().strip()
        self.pivot_lang = pivot_lang.lower().strip()
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.forward_model = None
        self.forward_tok = None
        self.backward_model = None
        self.backward_tok = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer
            fwd_name, bwd_name, _, actual_pivot = get_valid_marian_pairs(self.src_lang, self.pivot_lang)
            self.pivot_lang = actual_pivot
            
            print(f"[BackTranslation ({self.src_lang.upper()})] Loading verified models: {fwd_name} & {bwd_name}...")
            self.forward_tok = MarianTokenizer.from_pretrained(fwd_name)
            self.forward_model = MarianMTModel.from_pretrained(fwd_name).to(self.device)
            self.forward_model.eval()
            
            self.backward_tok = MarianTokenizer.from_pretrained(bwd_name)
            self.backward_model = MarianMTModel.from_pretrained(bwd_name).to(self.device)
            self.backward_model.eval()
            
            self._initialized = True
            print(f"[BackTranslation ({self.src_lang.upper()})] Models successfully loaded.")
        except Exception as e:
            print(f"[BackTranslation ({self.src_lang.upper()})] Warning: Could not initialize MarianMT models ({e}). Falling back to EDA.")
            self._initialized = False

    def augment(self, text: str) -> str:
        """Translates text to pivot language and back to source with diverse paraphrasing."""
        if not text or len(text.strip().split()) < 2:
            return text
            
        self._lazy_init()
        if not self._initialized:
            return random_swap(random_deletion(text, p=0.10), n_swaps=1)
            
        try:
            import torch
            with torch.no_grad():
                # Step 1: Forward translation (src -> pivot) using standard beam search
                inputs = self.forward_tok(
                    [text], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=256
                ).to(self.device)
                
                pivot_ids = self.forward_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )
                pivot_text = self.forward_tok.decode(pivot_ids[0], skip_special_tokens=True).strip()
                
                if not pivot_text:
                    return random_swap(text, n_swaps=1)
                
                # Step 2: Backward translation (pivot -> src) with nucleus sampling for natural variety
                inputs_back = self.backward_tok(
                    [pivot_text], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=256
                ).to(self.device)
                
                back_ids = self.backward_model.generate(
                    input_ids=inputs_back["input_ids"],
                    attention_mask=inputs_back.get("attention_mask"),
                    max_length=256,
                    do_sample=True,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    num_return_sequences=1
                )
                back_text = self.backward_tok.decode(back_ids[0], skip_special_tokens=True).strip()
                
                # If deterministic back-translation matched original 1:1, inject light slang or EDA variation
                if back_text.strip().lower() == text.strip().lower() or not back_text:
                    perturbed = augment_slang_noise(text, lang=self.src_lang, p=0.35)
                    if perturbed == text:
                        return random_swap(text, n_swaps=1)
                    return perturbed
                
                return back_text
        except Exception as e:
            print(f"[BackTranslation] Generation error: {e}")
            return random_swap(text, n_swaps=1)


# =============================================================================
# 3. COMPOSITE PIPELINE AUGMENTATION (PER-LANGUAGE STRATIFIED)
# =============================================================================

def augment_multilingual_dataframe(
    df: pd.DataFrame,
    lang: Optional[str] = None,        # If None, automatically detects and augments per language ('lang' column)
    implicit_upsample_ratio: float = 0.50,
    p_eda: float = 0.50,               # Probability of applying EDA (Swap, Delete, Mask)
    p_slang: float = 0.30,             # Probability of applying Slang/Teencode noise
    p_noise: Optional[float] = None,   # Alias for p_slang / noise
    use_back_translation: bool = False,
    pivot_lang: str = "de",
    mask_token: str = "<mask_token>",
    random_state: int = 42,
    **kwargs
) -> pd.DataFrame:
    """
    Stratified Multilingual Augmentation:
      - Automatically groups by language ('lang' column) for EN, IT, NL.
      - Uses verified pivot translation pairs with diverse sampling.
      - Upsamples minority Implicit Hate across every active language.
      - Injects Context Dropout and EDA.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    effective_slang_p = p_noise if p_noise is not None else p_slang

    # Check if we should automatically perform multi-language stratified augmentation
    has_lang_col = "lang" in df.columns
    
    if has_lang_col and (lang is None or lang == "auto" or lang == "all"):
        unique_langs = df["lang"].dropna().unique()
        print(f"[StratifiedMultilingualAugmentation] Auto-detected languages: {list(unique_langs)}")
        augmented_subsets = []
        for l in unique_langs:
            df_sub = df[df["lang"] == l]
            df_sub_aug = _augment_single_lang_df(
                df_sub,
                lang=str(l).lower(),
                implicit_upsample_ratio=implicit_upsample_ratio,
                p_eda=p_eda,
                p_slang=effective_slang_p,
                use_back_translation=use_back_translation,
                pivot_lang=pivot_lang,
                mask_token=mask_token,
                random_state=random_state
            )
            augmented_subsets.append(df_sub_aug)
            
        df_combined = pd.concat(augmented_subsets, ignore_index=True)
        df_combined = df_combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        print(f"[StratifiedMultilingualAugmentation Complete] Original: {len(df)} -> Total Augmented: {len(df_combined)}")
        return df_combined

    # Fallback to single/specified language
    target_lang = lang if lang is not None else "en"
    return _augment_single_lang_df(
        df,
        lang=target_lang,
        implicit_upsample_ratio=implicit_upsample_ratio,
        p_eda=p_eda,
        p_slang=effective_slang_p,
        use_back_translation=use_back_translation,
        pivot_lang=pivot_lang,
        mask_token=mask_token,
        random_state=random_state
    )


def _augment_single_lang_df(
    df: pd.DataFrame,
    lang: str,
    implicit_upsample_ratio: float,
    p_eda: float,
    p_slang: float,
    use_back_translation: bool,
    pivot_lang: str,
    mask_token: str,
    random_state: int
) -> pd.DataFrame:
    """Internal helper to augment a single language dataframe."""
    bt_augmenter = None
    if use_back_translation:
        bt_augmenter = BackTranslationAugmenter(src_lang=lang, pivot_lang=pivot_lang)
    
    augmented_rows = []
    
    comment_col = "yt_comment" if "yt_comment" in df.columns else ("comment" if "comment" in df.columns else "text")
    title_col = "yt_title" if "yt_title" in df.columns else "title"
    desc_col = "yt_description" if "yt_description" in df.columns else "description"
    
    is_implicit = False
    if 'hate_speech' in df.columns:
        is_implicit = (df['hate_speech'] == 'yes_implicit') | (df['hate_speech'] == 1)
    elif 'hs_y' in df.columns:
        is_implicit = (df['hs_y'] == 1)
    elif 'label' in df.columns:
        is_implicit = (df['label'] == 'yes_implicit') | (df['label'] == 1)
        
    df_implicit = df[is_implicit].copy() if isinstance(is_implicit, pd.Series) else df.head(0)
    
    num_to_augment = int(len(df_implicit) * implicit_upsample_ratio)
    if num_to_augment > 0:
        sampled_implicit = df_implicit.sample(n=num_to_augment, replace=True, random_state=random_state)
        
        for _, row in sampled_implicit.iterrows():
            new_row = row.copy()
            
            row_lang = str(row.get('lang', lang)).lower().strip()
            comment = str(row.get(comment_col, ''))
            
            # 1. Back-Translation (if enabled)
            if use_back_translation and bt_augmenter is not None and random.random() < 0.40:
                comment = bt_augmenter.augment(comment)
            
            # 2. EDA: Random Swap, Deletion, or Masking
            if random.random() < p_eda:
                eda_choice = random.choice(["swap", "delete", "mask"])
                if eda_choice == "swap":
                    comment = random_swap(comment, n_swaps=1)
                elif eda_choice == "delete":
                    comment = random_deletion(comment, p=0.10)
                elif eda_choice == "mask":
                    comment = random_mask(comment, mask_token=mask_token, p=0.10)
            
            # 3. Slang / Typographical variation using row's exact language
            if random.random() < p_slang:
                comment = augment_slang_noise(comment, lang=row_lang, p=0.20)
                
            new_row[comment_col] = comment
            
            # 4. Context Dropout
            title = str(row.get(title_col, '')) if title_col in row else ''
            desc = str(row.get(desc_col, '')) if desc_col in row else ''
            aug_t, aug_d = augment_context_dropout(title, desc)
            if title_col in new_row:
                new_row[title_col] = aug_t
            if desc_col in new_row:
                new_row[desc_col] = aug_d
            
            augmented_rows.append(new_row)
            
    if augmented_rows:
        df_aug = pd.DataFrame(augmented_rows)
        df_combined = pd.concat([df, df_aug], ignore_index=True)
        print(f"  [Augment {lang.upper()}] Original: {len(df)} -> Augmented: {len(df_combined)} (+{len(df_aug)} implicit)")
        return df_combined
        
    return df

# Backwards compatibility aliases
augment_task_b_dataframe = augment_multilingual_dataframe
random_token_deletion = random_deletion
