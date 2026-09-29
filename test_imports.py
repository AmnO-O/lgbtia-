#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Import Verification Test
"""

def test_imports():
    print("Testing pipeline exports...")
    from pipeline import (
        PipelineConfig,
        MultiTaskLoss,
        FocalLoss,
        build_loss_fn,
        PrivilegedConsistencyTaskBLoss,
        TaskBAdditiveLatentDataset,
        TaskBRoleDataset,
        TaskBClassAwareAttentionModel,
        TaskBTrainer,
    )
    print("All top-level pipeline exports imported successfully!")

if __name__ == '__main__':
    test_imports()
