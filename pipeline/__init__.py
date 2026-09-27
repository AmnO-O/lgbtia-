"""
StereoQueerEval & Toxic Classification Training Pipeline
A modular PyTorch pipeline supporting multi-task stereotype & hate-speech classification,
ModernBERT/mmBERT backbones, Bi-LSTM, and Transformer baselines.
"""

from .config import PipelineConfig, HATE_CLASSES, HATE2IDX, IDX2HATE
from .data import StereoQueerDataset, MMBertSeqDataset, DataPipeline, safe_clean, encode_target, decode_target
from .task_b_data import TaskBRoleDataset
from .augmentation import (
    random_swap,
    random_deletion,
    random_mask,
    augment_slang_noise,
    augment_context_dropout,
    BackTranslationAugmenter,
    augment_multilingual_dataframe,
    augment_task_b_dataframe,
)
from .models.task_b_class_aware import (
    TaskBClassAwareAttentionModel,
    RMSNorm,
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
from .losses import MultiTaskLoss, FocalLoss, build_loss_fn
from .metrics import evaluate_stereoqueer, print_metrics, compute_classification_metrics
from .trainer import StereoQueerTrainer
from .task_b_trainer import TaskBTrainer, FGM
from .inference import StereoQueerPredictor

__all__ = [
    "PipelineConfig",
    "HATE_CLASSES",
    "HATE2IDX",
    "IDX2HATE",
    "StereoQueerDataset",
    "MMBertSeqDataset",
    "TaskBRoleDataset",
    "random_swap",
    "random_deletion",
    "random_mask",
    "augment_slang_noise",
    "augment_context_dropout",
    "BackTranslationAugmenter",
    "augment_multilingual_dataframe",
    "augment_task_b_dataframe",
    "RMSNorm",
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
    "DataPipeline",
    "safe_clean",
    "encode_target",
    "decode_target",
    "MultiTaskLoss",
    "FocalLoss",
    "build_loss_fn",
    "evaluate_stereoqueer",
    "print_metrics",
    "compute_classification_metrics",
    "StereoQueerTrainer",
    "TaskBTrainer",
    "FGM",
    "StereoQueerPredictor",
]
