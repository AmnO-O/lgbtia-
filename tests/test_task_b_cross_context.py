"""
Comprehensive Smoke Test Suite for Task B Dual-Stream Cross-Context Architecture.
Validates:
1. Dataset & Tokenizer Dual-Stream Packing (TC Stream with precise role IDs vs Desc Stream).
2. Backbone Forward & Dimension propagation.
3. Cross-Attention Query-Key-Value computation and dynamic masking.
4. Element-wise Gating: bounds check (0 < g < 1), negative bias initialization check (g_init ~ 0.18).
5. Multi-Sample Dropout (MSD) classification head consistency.
6. Gradient Flow: Ensures loss backpropagates through both Title and Description streams.
7. Full Training Step and Eval Step sanity check with Dummy DataLoader.
"""

import sys
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.models.task_b_cross_context import (
    TaskBCrossContextAttentionModel,
    GatedCrossAttentionContextBlock,
    ROLE_PAD,
    ROLE_TITLE,
    ROLE_COMMENT,
    NUM_ROLES,
    NUM_CLASSES
)
from pipeline.task_b_cross_data import TaskBDualStreamDataset, create_task_b_dual_stream_loaders
from pipeline.config import PipelineConfig
from pipeline.task_b_cross_trainer import TaskBCrossContextTrainer


class DummyTokenizer:
    """Mock HuggingFace Tokenizer for deterministic fast unit testing."""
    def __init__(self):
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        # Generate simple hash-based pseudo token IDs
        words = text.strip().split()
        return [(abs(hash(w)) % 1000) + 10 for w in words]


class DummyBackbone(nn.Module):
    """Mock Transformer Backbone outputting shape [B, S, 768]."""
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(2000, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embeddings(input_ids) # [B, S, d]
        return self.dense(x)


class TestTaskBCrossContext(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.tokenizer = DummyTokenizer()
        self.d_model = 64 # lightweight for fast smoke testing

    def test_01_dual_stream_dataset_shapes_and_roles(self):
        """Test Dataset builds paired TC and Desc tensors with valid role assignments."""
        titles = ["Breaking News in City", "Live Stream Game"]
        comments = ["This is great content!", "I really dislike those people."]
        descriptions = ["Official channel video description with links.", "Subscribe for more content."]
        labels = [0, 1]

        ds = TaskBDualStreamDataset(
            titles=titles,
            comments=comments,
            descriptions=descriptions,
            labels=labels,
            tokenizer=self.tokenizer,
            tc_max_len=32,
            desc_max_len=24
        )

        self.assertEqual(len(ds), 2)
        item = ds[0]
        self.assertEqual(len(item), 6) # tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, label

        tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, label = item
        self.assertEqual(tc_ids.shape, torch.Size([32]))
        self.assertEqual(tc_mask.shape, torch.Size([32]))
        self.assertEqual(tc_roles.shape, torch.Size([32]))
        self.assertEqual(desc_ids.shape, torch.Size([24]))
        self.assertEqual(desc_mask.shape, torch.Size([24]))
        self.assertEqual(label.item(), 0)

        # Check role values are in valid domain {0, 1, 2}
        unique_roles = torch.unique(tc_roles).tolist()
        for r in unique_roles:
            self.assertIn(r, [ROLE_PAD, ROLE_TITLE, ROLE_COMMENT])

    def test_02_gated_cross_attention_block(self):
        """Test GatedCrossAttentionContextBlock outputs shape, element-wise gating in (0, 1), and negative bias init."""
        B, S_tc, S_d = 4, 16, 12
        h_comment = torch.randn(B, S_tc, self.d_model)
        h_desc = torch.randn(B, S_d, self.d_model)
        desc_pad_mask = torch.zeros(B, S_d, dtype=torch.bool)
        desc_pad_mask[:, 8:] = True # simulate pad tokens

        block = GatedCrossAttentionContextBlock(
            d_model=self.d_model,
            num_heads=4,
            gate_bias_init=-1.50
        )

        h_fused, gate = block(h_comment, h_desc, desc_key_padding_mask=desc_pad_mask, return_gate_values=True)
        self.assertEqual(h_fused.shape, torch.Size([B, S_tc, self.d_model]))
        self.assertEqual(gate.shape, torch.Size([B, S_tc, self.d_model]))

        # Check gate values strictly bounded in (0, 1)
        self.assertTrue(torch.all(gate >= 0.0) and torch.all(gate <= 1.0))
        
        # Check initial gate mean is conservative (~0.18 due to bias = -1.5)
        gate_mean = gate.mean().item()
        self.assertTrue(0.10 <= gate_mean <= 0.35, f"Expected gate init ~0.18, got {gate_mean}")

    def test_03_full_model_forward_and_msd(self):
        """Test full model forward pass, dual token pooling, and MSD logits aggregation."""
        backbone = DummyBackbone(d_model=self.d_model)
        model = TaskBCrossContextAttentionModel(
            mmbert_model=backbone,
            d_model=self.d_model,
            num_classes=3,
            num_heads=4
        )

        B, S_tc, S_d = 2, 20, 15
        tc_ids = torch.randint(1, 500, (B, S_tc))
        tc_mask = torch.ones(B, S_tc, dtype=torch.long)
        tc_roles = torch.tensor([[1]*8 + [2]*12, [1]*10 + [2]*10], dtype=torch.long)
        desc_ids = torch.randint(1, 500, (B, S_d))
        desc_mask = torch.ones(B, S_d, dtype=torch.long)

        # Single averaged logits
        out = model(tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, return_all_msd_logits=False)
        self.assertEqual(out.shape, torch.Size([B, 3]))

        # MSD multi-sample logits
        msd_outs = model(tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, return_all_msd_logits=True)
        self.assertIsInstance(msd_outs, list)
        self.assertEqual(len(msd_outs), 5)
        for logit in msd_outs:
            self.assertEqual(logit.shape, torch.Size([B, 3]))

    def test_04_gradient_flow_through_both_streams(self):
        """Test gradients backpropagate smoothly to both TC stream and Desc stream parameters."""
        backbone = DummyBackbone(d_model=self.d_model)
        model = TaskBCrossContextAttentionModel(
            mmbert_model=backbone,
            d_model=self.d_model,
            num_classes=3,
            num_heads=4
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B, S_tc, S_d = 2, 16, 12
        tc_ids = torch.randint(1, 500, (B, S_tc))
        tc_mask = torch.ones(B, S_tc, dtype=torch.long)
        tc_roles = torch.tensor([[1]*6 + [2]*10, [1]*8 + [2]*8], dtype=torch.long)
        desc_ids = torch.randint(1, 500, (B, S_d))
        desc_mask = torch.ones(B, S_d, dtype=torch.long)
        targets = torch.tensor([0, 2], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(tc_ids, tc_mask, tc_roles, desc_ids, desc_mask)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()

        # Verify gradients exist in gate, cross attention, and backbone embeddings
        self.assertIsNotNone(model.cross_context.gate_proj.weight.grad)
        self.assertIsNotNone(model.cross_context.cross_attn.in_proj_weight.grad)
        self.assertIsNotNone(model.role_embedding.weight.grad)
        self.assertIsNotNone(backbone.embeddings.weight.grad)

    def test_05_trainer_epoch_sanity(self):
        """Test TaskBCrossContextTrainer executes a training and validation epoch without crash."""
        backbone = DummyBackbone(d_model=self.d_model)
        model = TaskBCrossContextAttentionModel(
            mmbert_model=backbone,
            d_model=self.d_model,
            num_classes=3,
            num_heads=4
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
