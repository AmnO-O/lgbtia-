"""
Task B 4-Expert Mixture of Latent Query Banks Architecture (task_b_class_aware module).
Re-exports the modular MoE architecture while preserving full backward compatibility.
"""
from .norm import RMSNorm
from .router import ContextRouter, masked_mean_pooling
from .query_bank import ParallelQueryBankCrossAttention
from .head import MultiSampleDropoutHead
from .task_b_moe import (
    TaskB4ExpertMoEModel,
    TaskBClassAwareAttentionModel,
    CLASS_NO_HATE,
    CLASS_IMPLICIT_HATE,
    CLASS_EXPLICIT_HATE,
    NUM_CLASSES,
    ROLE_PAD,
    ROLE_TITLE,
    ROLE_DESC,
    ROLE_COMMENT,
    NUM_ROLES,
)

__all__ = [
    "RMSNorm",
    "ContextRouter",
    "masked_mean_pooling",
    "ParallelQueryBankCrossAttention",
    "MultiSampleDropoutHead",
    "TaskB4ExpertMoEModel",
    "TaskBClassAwareAttentionModel",
    "CLASS_NO_HATE",
    "CLASS_IMPLICIT_HATE",
    "CLASS_EXPLICIT_HATE",
    "NUM_CLASSES",
    "ROLE_PAD",
    "ROLE_TITLE",
    "ROLE_DESC",
    "ROLE_COMMENT",
    "NUM_ROLES",
]
