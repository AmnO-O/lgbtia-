"""
StereoQueerEval & Toxic Classification Training Pipeline
A modular PyTorch pipeline supporting multi-task stereotype & hate-speech classification,
ModernBERT/mmBERT backbones, Bi-LSTM, and Transformer baselines.
"""

from .config import PipelineConfig, HATE_CLASSES, HATE2IDX, IDX2HATE
from .data import StereoQueerDataset, MMBertSeqDataset, DataPipeline, safe_clean, encode_target, decode_target
from .task_b_data import TaskBRoleDataset
from .models.task_b_questions import QueryProbe, DEFAULT_TASK_B_PROBES, get_default_probes
from .models.task_b_class_aware import TaskBClassAwareAttentionModel
from .losses import MultiTaskLoss, FocalLoss
from .metrics import evaluate_stereoqueer, print_metrics
from .trainer import StereoQueerTrainer
from .task_b_trainer import TaskBTrainer
from .inference import StereoQueerPredictor

__all__ = [
    "PipelineConfig",
    "HATE_CLASSES",
    "HATE2IDX",
    "IDX2HATE",
    "StereoQueerDataset",
    "MMBertSeqDataset",
    "TaskBRoleDataset",
    "QueryProbe",
    "DEFAULT_TASK_B_PROBES",
    "get_default_probes",
    "TaskBClassAwareAttentionModel",
    "DataPipeline",
    "safe_clean",
    "encode_target",
    "decode_target",
    "MultiTaskLoss",
    "FocalLoss",
    "evaluate_stereoqueer",
    "print_metrics",
    "StereoQueerTrainer",
    "TaskBTrainer",
    "StereoQueerPredictor",
]
