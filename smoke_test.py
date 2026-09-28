"""
Smoke test suite for the StereoQueerEval Task B 4-Expert MoE pipeline and components.
Executes end-to-end forward/backward passes, unit tests on modular components, and training checks.
"""
import unittest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.config import PipelineConfig, HATE_CLASSES, HATE2IDX, IDX2HATE
from pipeline.data import DataPipeline, encode_target
from pipeline.models.norm import RMSNorm
from pipeline.models.router import ContextRouter, masked_mean_pooling
from pipeline.models.query_bank import ParallelQueryBankCrossAttention
from pipeline.models.head import MultiSampleDropoutHead
from pipeline.models.task_b_moe import (
    TaskB4ExpertMoEModel,
    TaskBClassAwareAttentionModel,
    ROLE_PAD, ROLE_TITLE, ROLE_DESC, ROLE_COMMENT
)
from pipeline.losses import FocalLoss, MoELoadBalanceLoss, TaskBLoss, MultiTaskLoss
from pipeline.task_b_data import TaskBRoleDataset
from pipeline.task_b_trainer import TaskBTrainer


class MockBackbone(nn.Module):
    """Mock HuggingFace backbone producing [B, S, d_model] hidden states."""
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(1000, d_model)

    def forward(self, input_ids, attention_mask=None):
        out = self.embed(input_ids % 1000)
        class Output:
            pass
        res = Output()
        res.last_hidden_state = out
        return res


class MockTokenizer:
    """Mock tokenizer providing token IDs for unit tests."""
    def __init__(self):
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        words = text.split()
        return [abs(hash(w)) % 900 + 10 for w in words]

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{i}" for i in ids]


class TestPipelineSmoke(unittest.TestCase):
    def setUp(self):
        self.raw_data = {
            'yt_title': ['Video Alpha', 'Video Alpha', 'Video Beta', 'Video Gamma', 'Video Gamma'],
            'yt_description': ['Desc A', 'Desc A', 'Desc B', 'Desc C', 'Desc C'],
            'yt_comment': [
                'Great video about rights',
                'Hate comment here',
                'Neutral comment',
                'Implicit hate message',
                'Another normal comment'
            ],
            'stereotype': ['no', 'yes', 'no', 'yes', 'no'],
            'hate_speech': ['no', 'yes_explicit', 'no', 'yes_implicit', 'no'],
            'target': ['none', 'group_lgbtqia+', 'none', 'individual_t', 'none'],
        }
        self.df = pd.DataFrame(self.raw_data)
        self.config = PipelineConfig(
            task="stereoqueer",
            target_task="hs",
            model_type="task_b_class_aware",
            batch_size=2,
            max_length=32,
            two_phase=False,
            device="cpu",
            num_experts=4,
            loss_balance_weight=0.01
        )

    def test_01_constants_and_imports(self):
        self.assertEqual(len(HATE_CLASSES), 3)
        self.assertIn('yes_implicit', HATE2IDX)
        self.assertEqual(IDX2HATE[0], 'no')

    def test_02_rmsnorm(self):
        dim = 32
        norm = RMSNorm(dim)
        x = torch.randn(4, 10, dim) * 5.0
        out = norm(x)
        self.assertEqual(out.shape, (4, 10, dim))
        # Variance along last dim should be close to 1
        rms = torch.sqrt(out.pow(2).mean(-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-2))

    def test_03_masked_mean_pooling(self):
        B, S, D = 2, 4, 8
        h = torch.ones(B, S, D)
        # Sample 0 has 2 valid tokens, Sample 1 has 4 valid tokens
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long)
        pooled = masked_mean_pooling(h, mask)
        self.assertEqual(pooled.shape, (B, D))
        self.assertTrue(torch.allclose(pooled, torch.ones(B, D)))

    def test_04_context_router(self):
        B, S, D, N = 3, 10, 64, 4
        router = ContextRouter(d_model=D, num_experts=N, hidden_dim=32, temperature=1.0)
        h = torch.randn(B, S, D)
        mask = torch.ones(B, S, dtype=torch.long)
        gates, logits = router(h, mask, return_logits=True)
        
        self.assertEqual(gates.shape, (B, N))
        self.assertEqual(logits.shape, (B, N))
        # Check sum to 1.0 per sample
        gate_sums = gates.sum(dim=-1)
        self.assertTrue(torch.allclose(gate_sums, torch.ones(B), atol=1e-5))

    def test_05_parallel_query_bank(self):
        B, S, D, N = 2, 16, 64, 4
        qb = ParallelQueryBankCrossAttention(d_model=D, num_experts=N, num_heads=2)
        h = torch.randn(B, S, D)
        mask = torch.ones(B, S, dtype=torch.long)
        
        z_experts, attn = qb(h, mask, return_attention_map=True)
        self.assertEqual(z_experts.shape, (B, N, D))
        self.assertEqual(attn.shape, (B, N, S))

    def test_06_load_balance_loss(self):
        loss_fn = MoELoadBalanceLoss(num_experts=4)
        
        # Perfect uniform distribution: mean(g) = [0.25, 0.25, 0.25, 0.25]
        uniform_gates = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
        loss_uniform = loss_fn(uniform_gates)
        self.assertAlmostEqual(loss_uniform.item(), 0.0, places=4)

        # Complete collapse: all samples pick expert 0
        collapsed_gates = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        loss_collapsed = loss_fn(collapsed_gates)
        # N * (1.0^2 + 0 + 0 + 0) - 1 = 4 * 1 - 1 = 3.0
        self.assertAlmostEqual(loss_collapsed.item(), 3.0, places=4)

    def test_07_model_forward_backward_moe(self):
        backbone = MockBackbone(d_model=64)
        model = TaskB4ExpertMoEModel(
            mmbert_model=backbone,
            d_model=64,
            num_experts=4,
            num_heads=2,
            dropout=0.1
        )

        B, S = 2, 16
        dummy_ids = torch.randint(0, 500, (B, S))
        dummy_mask = torch.ones((B, S), dtype=torch.long)
        dummy_roles = torch.randint(0, 4, (B, S), dtype=torch.long)
        labels = torch.tensor([0, 2], dtype=torch.long)

        # Forward with return_gates=True and return_attention_map=True
        logits, h_B, gates, attn = model(
            dummy_ids, dummy_mask, dummy_roles,
            return_gates=True,
            return_attention_map=True
        )
        self.assertEqual(logits.shape, (B, 3))
        self.assertEqual(h_B.shape, (B, 64))
        self.assertEqual(gates.shape, (B, 4))
        self.assertEqual(attn.shape, (B, 4, S))

        # Backward gradient flow check
        loss_fn = TaskBLoss(base_criterion=nn.CrossEntropyLoss(), num_experts=4, loss_balance_weight=0.01)
        loss = loss_fn(logits, labels, gates=gates)
        loss.backward()

        self.assertIsNotNone(model.role_embeddings.weight.grad)
        self.assertIsNotNone(model.query_banks.expert_queries.grad)
        self.assertIsNotNone(model.router.mlp[0].weight.grad)

    def test_08_trainer_step_moe(self):
        backbone = MockBackbone(d_model=64)
        model = TaskBClassAwareAttentionModel(
            mmbert_model=backbone,
            d_model=64,
            num_experts=4,
            num_heads=2,
            dropout=0.1
        )
        tok = MockTokenizer()
        df_test = self.df.copy()
        df_test['st_y'] = 0.0
        df_test['hs_y'] = [0, 2, 0, 1, 0]
        df_test['tg_y'] = [[0.0] * 10 for _ in range(len(df_test))]

        ds = TaskBRoleDataset(df_test, tok, max_len=16)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        trainer = TaskBTrainer(
            model=model,
            config=self.config,
            train_loader=loader,
            val_loader=loader,
            df_val=df_test
        )
        optimizer = trainer.build_optimizer(lr=1e-3)
        train_loss, train_gates = trainer.train_epoch(optimizer)
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0.0)
        self.assertEqual(len(train_gates), 4)

        val_loss, metrics, preds, probs, val_gates = trainer.eval_epoch()
        self.assertIn('hs_macro_f1', metrics)
        self.assertIn('hs_acc', metrics)
        self.assertEqual(val_gates.shape[1], 4)


if __name__ == '__main__':
    unittest.main()
