#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pilot & Unit Test for Quality-Gated Diagnostic Generation,
Leak-Proof Sanitizer, and Dataset Integration.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from gen.llm_rationalize import (
    sanitize_leak_free_text,
    compile_latent_hint,
    format_target_string
)
from pipeline.task_b_data import TaskBAdditiveLatentDataset

def run_pilot_and_unit_tests():
    print("==================================================")
    print("  RUNNING PILOT & INTEGRATION VERIFICATION")
    print("==================================================")

    # 1. Test Leak-Proof Regex Sanitizer
    print("\n[TEST 1/4] Testing Sanitizer Edge Cases...")
    test_cases = [
        ("This comment contains yes_implicit hate speech.", "This comment contains [MASKED] [MASKED]."),
        ("Explicit hostility against trans people.", "[MASKED] hostility against trans people."),
        ("A neutral, non-hate remark on the video.", "A [MASKED], [MASKED] remark on the video."),
        ("Implicitly mocking non-binary individuals.", "[MASKED] mocking non-binary individuals.")
    ]
    for raw, _ in test_cases:
        cleaned = sanitize_leak_free_text(raw)
        assert "implicit" not in cleaned.lower()
        assert "explicit" not in cleaned.lower()
        assert "hate" not in cleaned.lower()
        assert "neutral" not in cleaned.lower()
        assert "non-hate" not in cleaned.lower()
        assert "[MASKED]" in cleaned
    print("  ✅ All sanitizer test cases scrubbed label tokens completely.")

    # 2. Test Hint Compilation & Budget (<=64 tokens)
    print("\n[TEST 2/4] Testing Hint Compilation & Token Budget...")
    axes = {
        'direct_hostility': 'weak',
        'indirect_subtext': 'strong',
        'context_dependence': 'moderate',
        'counter_speech': 'weak',
        'target_reference': 'present'
    }
    why = "Speaker uses subtle irony to undermine the validity of transgender identities."
    boundary = "Might appear benign at first glance due to polite wording."
    hint = compile_latent_hint(axes, why, boundary, max_tokens=64)
    print("  Compiled Hint:", hint)
    words = hint.split()
    assert len(words) <= 64, f"Hint word count {len(words)} exceeds 64"
    assert "axes:" in hint
    assert "why:" in hint
    assert "flip:" in hint
    print(f"  ✅ Hint compiled smoothly (word count = {len(words)} <= 64).")

    # 3. Test Target String Formatting
    print("\n[TEST 3/4] Testing Target String Formatting...")
    t1 = format_target_string(["t", "group_lgbtqia+"], "group")
    assert t1 == "lgbtqia+,t;group", f"Expected lgbtqia+,t;group, got {t1}"
    t2 = format_target_string([], "none")
    assert t2 == "none"
    print("  ✅ Target string formatting matches SemEval Task C convention.")

    # 4. Test TaskBAdditiveLatentDataset with Hint & Empty Hint
    print("\n[TEST 4/4] Testing TaskBAdditiveLatentDataset Integration...")
    class DummyTokenizer:
        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [hash(w) % 1000 + 1 for w in text.split()]
            
    df_pilot = pd.DataFrame([
        {'StereoQueerEval_id': '1', 'yt_title': 'Video 1', 'yt_description': 'Desc 1', 'yt_comment': 'Comment 1', 'hs_y': 1, 'st_y': 1, 'tg_y': [0]*10, 'hint': hint},
        {'StereoQueerEval_id': '2', 'yt_title': 'Video 2', 'yt_description': 'Desc 2', 'yt_comment': 'Comment 2', 'hs_y': 0, 'st_y': 0, 'tg_y': [0]*10, 'hint': ''}, # empty hint fallback
    ])
    
    tok = DummyTokenizer()
    dataset = TaskBAdditiveLatentDataset(df_pilot, tok, max_len=32, hint_max_len=16, hint_col='hint')
    item_with_hint = dataset[0]
    item_empty_hint = dataset[1]
    
    # Check item with hint
    assert item_with_hint[3].shape[0] == 16 # hint_ids
    assert (item_with_hint[4] == 1).sum() > 0 # hint_mask has active tokens
    
    # Check item with empty hint (all pads)
    assert (item_empty_hint[4] == 0).all() # hint_mask is all 0
    print("  ✅ Dataset correctly loads pre-compiled hint and safely handles empty-hint fallback.")

    print("\n==================================================")
    print("  🎉 ALL PILOT TESTS PASSED CLEANLY!")
    print("==================================================")

if __name__ == '__main__':
    run_pilot_and_unit_tests()
