"""
Unit Tests for Task B Dual-Stream Cross-Context Architecture.

Verifies:
1. Dual-Stream Tokenization & Role Alignment
2. Gated Cross-Attention Context Block Mechanics & Gate Bias Init
3. Full Model Forward Pass with MSD (use_msd=True) and Single-Dropout (use_msd=False)
4. Gradient Flow from both Text Streams through Backbone
5. Two-Phase Trainer Loop Sanity
"""

import sys
import unittest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from pipeline.models.task_b_cross_context import (
    TaskBCrossContextAttentionModel,
    GatedCrossAttentionContextBlock,
    ROLE_TITLE,
    ROLE_COMMENT
)
from pipeline.task_b_cross_data import create_task_b_dual_stream_loaders
from pipeline.config import PipelineConfig
from pipeline.task_b_cross_trainer import TaskBCrossContextTrainer


class DummyBackbone(nn.Module):
    """Dummy transformer backbone returning [B, S, d_model] embeddings."""
    def __init__(self, vocab_size=1000, d_model=64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask=None):
        h = self.embeddings(input_ids)
        h = self.linear(h)
        return h


class TestTaskBCrossContext(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        self.d_model = 64

    def test_01_dual_stream_tokenization_and_roles(self):
        """Test dataset collator builds valid token IDs, masks, and role sequences."""
        df_mock = pd.DataFrame({
            'yt_title': ['Video Title One', 'Another Video'],
            'yt_comment': ['User comment text here.', 'Another short comment.'],
            'yt_description': ['Video description long context.', 'Short desc.'],
            'hs_y': [0, 1]
        })

        train_loader, _ = create_task_b_dual_stream_loaders(
            df_train=df_mock,
            df_val=df_mock,
            tokenizer=self.tokenizer,
            batch_size=2,
            tc_max_len=32,
            desc_max_len=24
        )

        batch = next(iter(train_loader))
        tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, labels = batch

        # Check dimensions
        self.assertEqual(tc_ids.shape, (2, 32))
        self.assertEqual(tc_mask.shape, (2, 32))
        self.assertEqual(tc_roles.shape, (2, 32))
        self.assertEqual(desc_ids.shape, (2, 24))
        self.assertEqual(desc_mask.shape, (2, 24))
        self.assertEqual(labels.shape, (2,))

        # Check roles contain both ROLE_TITLE and ROLE_COMMENT
        self.assertTrue((tc_roles == ROLE_TITLE).any())
        self.assertTrue((tc_roles == ROLE_COMMENT).any())

    def test_02_gated_cross_attention_block(self):
        """Test GatedCrossAttentionContextBlock shape consistency and conservative gate init."""
        block = GatedCrossAttentionContextBlock(
            d_model=self.d_model,
            num_heads=4,
            dropout=0.10,
            gate_bias_init=-1.50
        )

        B, S_tc, S_d = 2, 16, 12
        h_comment = torch.randn(B, S_tc, self.d_model)
        h_desc = torch.randn(B, S_d, self.d_model)
        desc_mask = torch.zeros(B, S_d, dtype=torch.bool) # False means valid token

        fused, gate = block(h_comment, h_desc, desc_key_padding_mask=desc_mask, return_gate_values=True)

        self.assertEqual(fused.shape, (B, S_tc, self.d_model))
        self.assertEqual(gate.shape, (B, S_tc, self.d_model))

        # Check initial gate mean is conservative (~0.18 due to bias = -1.5)
        gate_mean = gate.mean().item()
        self.assertTrue(0.10 <= gate_mean <= 0.35, f"Expected gate init ~0.18, got {gate_mean}")

    def test_03_full_model_forward_and_msd(self):
        """Test full model forward pass with both MSD (use_msd=True) and Single-Dropout (use_msd=False)."""
        backbone = DummyBackbone(d_model=self.d_model)
        
        # 1. Test with MSD enabled
        model_msd = TaskBCrossContextAttentionModel(
            mmbert_model=backbone,
            d_model=self.d_model,
            num_classes=3,
            num_heads=4,
            use_msd=True,
            msd_dropout_rates=[0.1, 0.2, 0.3, 0.4, 0.5]
        )

        B, S_tc, S_d = 2, 20, 15
        tc_ids = torch.randint(1, 500, (B, S_tc))
        tc_mask = torch.ones(B, S_tc, dtype=torch.long)
        tc_roles = torch.tensor([[1]*8 + [2]*12, [1]*10 + [2]*10], dtype=torch.long)
        desc_ids = torch.randint(1, 500, (B, S_d))
        desc_mask = torch.ones(B, S_d, dtype=torch.long)

        # Single averaged logits
        out = model_msd(tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, return_all_msd_logits=False)
        self.assertEqual(out.shape, torch.Size([B, 3]))

        # MSD multi-sample logits (5 samples)
        msd_outs = model_msd(tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, return_all_msd_logits=True)
        self.assertIsInstance(msd_outs, list)
        self.assertEqual(len(msd_outs), 5)
        for logit in msd_outs:
            self.assertEqual(logit.shape, torch.Size([B, 3]))

        # 2. Test with Single Dropout (use_msd=False, dropout=0.50)
        model_single = TaskBCrossContextAttentionModel(
            mmbert_model=backbone,
            d_model=self.d_model,
            num_classes=3,
            num_heads=4,
            dropout=0.50,
            use_msd=False
        )
        out_single = model_single(tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, return_all_msd_logits=False)
        self.assertEqual(out_single.shape, torch.Size([B, 3]))

    def test_04_gradient_flow_through_both_streams(self):
        """Test gradients backpropagate smoothly to both TC stream and Desc stream parameters."""
        backbone = DummyBackbone(d_model=self.d_model)
        model = TaskBCrossContextAttentionModel(
            mmbert_model=backbone,
            d_model=self.d_model,
            num_classes=3,
            num_heads=4,
            use_msd=True
        )

        B, S_tc, S_d = 2, 16, 12
        tc_ids = torch.randint(1, 500, (B, S_tc))
        tc_mask = torch.ones(B, S_tc, dtype=torch.long)
        tc_roles = torch.tensor([[1]*8 + [2]*8, [1]*8 + [2]*8], dtype=torch.long)
        desc_ids = torch.randint(1, 500, (B, S_d))
        desc_mask = torch.ones(B, S_d, dtype=torch.long)
        labels = torch.tensor([0, 2], dtype=torch.long)

        logits = model(tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, return_all_msd_logits=False)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(model.role_embedding.weight.grad)
        self.assertIsNotNone(model.cross_context.gate_proj.weight.grad)
        self.assertIsNotNone(model.classifier.classifier.weight.grad)
        self.assertIsNotNone(backbone.embeddings.weight.grad)

    def test_05_trainer_epoch_sanity(self):
        """Test TaskBCrossContextTrainer executes a training and validation epoch without crash."""
        backbone = DummyBackbone(d_model=self.d_model)
        model = TaskBCrossContextAttentionModel(
            mmbert_model=backbone,
            d_model=self.d_model,
            num_classes=3,
            num_heads=4,
            use_msd=True
        )

        df_mock = pd.DataFrame({
            'yt_title': ['Title A', 'Title B', 'Title C', 'Title D'],
            'yt_comment': ['Comment A', 'Comment B', 'Comment C', 'Comment D'],
            'yt_description': ['Desc A', 'Desc B', 'Desc C', 'Desc D'],
            'hs_y': [0, 1, 2, 0]
        })

        train_loader, val_loader = create_task_b_dual_stream_loaders(
            df_train=df_mock,
            df_val=df_mock,
            tokenizer=self.tokenizer,
            batch_size=2,
            tc_max_len=16,
            desc_max_len=12
        )

        config = PipelineConfig()
        config.learning_rate = 1e-3
        config.use_amp = False
        config.use_fgm = False
        config.output_dir = "/tmp/task_b_smoke_test"

        trainer = TaskBCrossContextTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu")
        )

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loss, train_metrics = trainer.train_epoch(optimizer=opt, apply_fgm=False)
        self.assertGreater(train_loss, 0.0)
        self.assertIn('hs_macro_f1', train_metrics)

        val_loss, val_metrics, y_true, y_pred, gate_mean = trainer.eval_epoch()
        self.assertGreater(val_loss, 0.0)
        self.assertEqual(len(y_true), 4)
        self.assertEqual(len(y_pred), 4)
        self.assertGreater(gate_mean, 0.0)


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTaskBCrossContext)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
