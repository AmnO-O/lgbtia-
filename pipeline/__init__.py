"""
StereoQueerEval & Toxic Classification Training Pipeline
A modular PyTorch pipeline supporting multi-task stereotype & hate-speech classification,
ModernBERT/mmBERT backbones, Bi-LSTM, and Transformer baselines.
"""

from .config import (
    PipelineConfig,
    HATE_CLASSES,
    NUM_CLASSES,
    HATE2IDX,
    IDX2HATE,
    ROLE_PAD,
    ROLE_TITLE,
    ROLE_COMMENT,
    ROLE_DESC,
)
from .data import StereoQueerDataset, MMBertSeqDataset, DataPipeline, safe_clean, encode_target, decode_target
from .task_b_data import TaskBAdditiveLatentDataset, TaskBRoleDataset
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
from .models.task_b_cross_context import (
    TaskBCrossContextAttentionModel,
    GatedCrossAttentionContextBlock,
    HierarchicalMSDClassifier,
)
from .task_b_cross_data import (
    TaskBDualStreamDataset,
    create_task_b_dual_stream_loaders,
)
from .task_b_cross_trainer import TaskBCrossContextTrainer
from .models.task_b_questions import (
    QueryProbe,
    DEFAULT_TASK_B_PROBES,
    get_default_probes,
    get_probes_for_language,
)
from .losses import (
    MultiTaskLoss,
    FocalLoss,
    build_loss_fn,
    UnidirectionalKLDivergenceLoss,
    PrivilegedConsistencyTaskBLoss,
    HierarchicalTaskBLoss,
    TaskBLoss
)
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
    "TaskBAdditiveLatentDataset",
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
    "TaskBCrossContextAttentionModel",
    "GatedCrossAttentionContextBlock",
    "TaskBDualStreamDataset",
    "create_task_b_dual_stream_loaders",
    "TaskBCrossContextTrainer",
    "CLASS_NO_HATE",
    "CLASS_IMPLICIT_HATE",
    "CLASS_EXPLICIT_HATE",
    "NUM_CLASSES",
    "ROLE_PAD",
    "ROLE_TITLE",
    "ROLE_DESC",
    "ROLE_COMMENT",
    "ROLE_HINT",
    "NUM_ROLES",
    "QueryProbe",
    "DEFAULT_TASK_B_PROBES",
    "get_default_probes",
    "get_probes_for_language",
    "DataPipeline",
    "safe_clean",
    "encode_target",
    "decode_target",
    "MultiTaskLoss",
    "FocalLoss",
    "build_loss_fn",
    "UnidirectionalKLDivergenceLoss",
    "PrivilegedConsistencyTaskBLoss",
    "TaskBLoss",
    "evaluate_stereoqueer",
    "print_metrics",
    "compute_classification_metrics",
    "StereoQueerTrainer",
    "TaskBTrainer",
    "FGM",
    "StereoQueerPredictor",
]
