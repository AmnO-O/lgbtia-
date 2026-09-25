"""
StereoQueerEval & Toxic Classification Training Pipeline
A modular PyTorch pipeline supporting multi-task stereotype & hate-speech classification,
ModernBERT/mmBERT backbones, Bi-LSTM, and Transformer baselines.
"""

from .config import PipelineConfig
from .data import StereoQueerDataset, MMBertSeqDataset, DataPipeline, safe_clean, encode_target, decode_target
from .losses import MultiTaskLoss
from .metrics import evaluate_stereoqueer, print_metrics
from .trainer import StereoQueerTrainer
from .inference import StereoQueerPredictor

__all__ = [
    "PipelineConfig",
    "StereoQueerDataset",
    "MMBertSeqDataset",
    "DataPipeline",
    "safe_clean",
    "encode_target",
    "decode_target",
    "MultiTaskLoss",
    "evaluate_stereoqueer",
    "print_metrics",
    "StereoQueerTrainer",
    "StereoQueerPredictor",
]
