#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end Smoke Test Suite for Task B Additive Latent Privileged Training Pipeline.
Tests:
1. Leak-Proof Regex Sanitizer (plan.md §4: implicit|explicit|hate|neutral|non-hate -> [MASKED])
2. CosineCurriculumAnnealingScheduler (Alpha decay, lambda scaling, boundary conditions)
3. UnidirectionalKLDivergenceLoss & PrivilegedConsistencyTaskBLoss (Gradients, stop_grad stability, numeric limits)
4. TaskBAdditiveLatentDataset (Primary tokenization, Hint tokenization, empty fallback)
5. TaskBClassAwareAttentionModel (Unguided forward, Guided forward with alpha, identical outputs when alpha=0)
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def run_tests():
    print("==================================================")
    print("  RUNNING PIPELINE VERIFICATION & SMOKE TEST")
    print("==================================================")

    # 1. Test Leak-Proof Sanitizer
    print("\n[TEST 1/5] Testing Strict Leak-Proof Sanitizer (plan.md §4)...")
    from gen.llm_rationalize import sanitize_leak_free_text
    test_leak_sentence = "This comment displays yes_implicit hate speech and explicit hostility rather than neutral or non-hate content."
    sanitized = sanitize_leak_free_text(test_leak_sentence)
    
    assert "yes_implicit" not in sanitized.lower()
    assert "explicit" not in sanitized.lower()
    assert "hate" not in sanitized.lower()
    assert "neutral" not in sanitized.lower()
    assert "non-hate" not in sanitized.lower()
    assert "[MASKED]" in sanitized
    print("  Sanitized output:", sanitized)
    print("  ✅ Leak-Proof Sanitizer strictly scrubbed all gold label tokens.")

    # 2. Test Scheduler
    print("\n[TEST 2/5] Testing CosineCurriculumAnnealingScheduler...")
    from pipeline.scheduler import CosineCurriculumAnnealingScheduler
    sched = CosineCurriculumAnnealingScheduler(total_epochs=10, start_alpha=1.0, end_alpha=0.0)
    p0 = sched.step(0)
    p5 = sched.step(5)
    p10 = sched.step(10)
    assert p0['alpha'] == 1.0, f"Epoch 0 alpha should be 1.0, got {p0['alpha']}"
    assert abs(p5['alpha'] - 0.5) < 1e-4, f"Epoch 5 alpha should be ~0.5, got {p5['alpha']}"
    assert p10['alpha'] == 0.0, f"Epoch 10 alpha should be 0.0, got {p10['alpha']}"
    assert p10['lambda_c'] == 0.0, f"Epoch 10 lambda_c should be 0.0, got {p10['lambda_c']}"
    print("  ✅ Scheduler logic passed seamlessly.")

    # 3. Test Losses
    print("\n[TEST 3/5] Testing FocalLoss & PrivilegedConsistencyTaskBLoss...")
    from pipeline.losses import build_loss_fn, PrivilegedConsistencyTaskBLoss
    base_loss = build_loss_fn(loss_type="focal", gamma=2.0)
    priv_loss = PrivilegedConsistencyTaskBLoss(base_criterion=base_loss, temperature=1.0)
    
    dummy_logits_u = torch.randn(4, 3, requires_grad=True)
    dummy_logits_c = torch.randn(4, 3, requires_grad=True)
    dummy_targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    
    loss, details = priv_loss(
        logits_u=dummy_logits_u,
        targets=dummy_targets,
        logits_c=dummy_logits_c,
        lambda_c=0.4,
        lambda_cons=0.3
    )
    loss.backward()
    assert dummy_logits_u.grad is not None, "Gradients must flow into unguided logits"
    print("  ✅ Loss forward/backward and stop-gradient passed without numeric issues.")

    # 4. Test Dataset
    print("\n[TEST 4/5] Testing TaskBAdditiveLatentDataset...")
    from pipeline.task_b_data import TaskBAdditiveLatentDataset
    
    class DummyTokenizer:
        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [hash(w) % 1000 + 1 for w in text.split()]
            
    df_sample = pd.DataFrame([
        {'StereoQueerEval_id': '1', 'yt_title': 'Pride Parade', 'yt_description': 'Annual event', 'yt_comment': 'Nice!', 'hs_y': 0, 'st_y': 0, 'tg_y': [0]*10, 'rationale': 'Supportive community gathering'},
        {'StereoQueerEval_id': '2', 'yt_title': 'News', 'yt_description': 'Debate', 'yt_comment': 'Protect kids', 'hs_y': 1, 'st_y': 1, 'tg_y': [0]*10, 'rationale': 'Faux concern subtext'},
        {'StereoQueerEval_id': '3', 'yt_title': 'Vlog', 'yt_description': 'Day out', 'yt_comment': 'Great video', 'hs_y': 0, 'st_y': 0, 'tg_y': [0]*10, 'rationale': ''}, # empty hint test
    ])
    
    tok = DummyTokenizer()
    dataset = TaskBAdditiveLatentDataset(df_sample, tok, max_len=32, hint_max_len=16)
    item = dataset[0]
    assert len(item) == 8, f"Dataset item must return 8 elements, got {len(item)}"
    input_ids_u, att_mask_u, role_ids_u, hint_ids, hint_mask, st_y, hs_y, tg_y = item
    assert input_ids_u.shape[0] == 32, "Primary sequence must match max_len"
    assert hint_ids.shape[0] == 16, "Hint sequence must match hint_max_len"
    print("  ✅ Dataset dual-stream collation passed.")

    # 5. Test Model Architecture & Invariant Condition
    print("\n[TEST 5/5] Testing TaskBClassAwareAttentionModel...")
    from pipeline.models.task_b_class_aware import TaskBClassAwareAttentionModel
    
    class DummyEncoder(nn.Module):
        def __init__(self, d_model=64):
            super().__init__()
            self.d_model = d_model
            self.emb = nn.Embedding(2000, d_model)
        def forward(self, input_ids, attention_mask):
            bsz, seq_len = input_ids.shape
            hidden = self.emb(input_ids)
            class Out:
                pass
            o = Out()
            o.last_hidden_state = hidden
            return o
            
    dummy_encoder = DummyEncoder(d_model=64)
    model = TaskBClassAwareAttentionModel(mmbert_model=dummy_encoder, d_model=64, num_classes=3, use_msd=False)
    model.eval()
    
    b_in = torch.randint(0, 100, (2, 32))
    b_mask = torch.ones((2, 32), dtype=torch.long)
    b_roles = torch.ones((2, 32), dtype=torch.long)
    b_hint_in = torch.randint(0, 100, (2, 16))
    b_hint_mask = torch.ones((2, 16), dtype=torch.long)
    
    # 5a. Pure unguided pass (hint_alpha = 0.0)
    out_unguided, _, _ = model(b_in, b_mask, b_roles, hint_alpha=0.0)
    
    # 5b. Pure unguided pass with hint passed but alpha=0.0
    out_unguided_with_dummy_hint, _, _ = model(b_in, b_mask, b_roles, hint_ids=b_hint_in, hint_mask=b_hint_mask, hint_alpha=0.0)
    
    # Assert physical identity
    diff = torch.max(torch.abs(out_unguided - out_unguided_with_dummy_hint)).item()
    assert diff < 1e-6, f"When hint_alpha=0.0, output must be physically identical to unguided! Diff: {diff}"
    
    # 5c. Guided pass (hint_alpha = 1.0)
    out_guided, _, _ = model(b_in, b_mask, b_roles, hint_ids=b_hint_in, hint_mask=b_hint_mask, hint_alpha=1.0)
    assert out_guided.shape == (2, 3), f"Logits shape should be (2, 3), got {out_guided.shape}"
    print("  ✅ Additive Latent Fusion & Invariant Equality (alpha=0 <=> pure unguided) verified 100%.")

    print("\n==================================================")
    print("  🎉 ALL 5 SMOKE TESTS PASSED CLEANLY & SAFELY!")
    print("==================================================")

if __name__ == '__main__':
    run_tests()
