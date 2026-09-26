import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

# Standard identity ordering for SemEval StereoQueerEval
ID_ORDER = ['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+']
SCOPE_DIM = len(ID_ORDER)   # index 9 represents group (1.0) vs individual (0.0)
TARGET_DIM = SCOPE_DIM + 1  # 9 identities + 1 scope = 10 bits

HATE_CLASSES = ['no', 'yes_implicit', 'yes_explicit']
HATE2IDX = {'no': 0, 'yes_implicit': 1, 'yes_explicit': 2}
IDX2HATE = {v: k for k, v in HATE2IDX.items()}

@dataclass
class PipelineConfig:
    """Configuration class for the model training pipeline."""
    task: str = "stereoqueer"                   # 'stereoqueer' or 'toxic'
    target_task: str = "all"                    # 'all', 'st' (stereotype), 'hs' (hate_speech), or 'tg' (target)
    embed_source: str = "mmbert"                # 'mmbert' or 'scratch'
    model_type: str = "mmbert_transformer"      # 'mmbert_transformer', 'feature_mlp', 'transformer', 'bilstm', 'rnn'
    
    # Pretrained Transformer settings
    mmbert_model_name: str = "jhu-clsp/mmbert-base"
    mmbert_dim: int = 768
    max_length: int = 256
    use_query_interaction: bool = True          # Layer 2 MHSA Ablation Hypothesis H2
    
    # Scratch model parameters
    vocab_size: int = 30000
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.3
    
    # Advanced Regularization & Adversarial Training
    use_msd: bool = True                        # Multi-Sample Dropout in classification head
    msd_num_samples: int = 5                    # Number of parallel dropout masks in MSD
    msd_dropout_rates: List[float] = field(default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30])
    use_fgm: bool = True                        # Fast Gradient Method (Adversarial Training on embeddings)
    fgm_epsilon: float = 1.0                    # FGM perturbation scale epsilon
    fgm_emb_name: str = "word_embeddings"       # Name substring of embedding layer to perturb
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    patience: int = 7
    epochs: int = 30
    loss_type: str = "focal"                    # 'focal' or 'cross_entropy'
    focal_gamma: float = 2.0                    # Focusing parameter for FocalLoss
    label_smoothing: float = 0.05               # Regularization for loss
    class_weights: Optional[List[float]] = None # Balanced class weights [no, implicit, explicit]
    
    # 2-Phase Training settings for pretrained backbones
    two_phase: bool = True
    freeze_phase_epochs: int = 15
    unfreeze_phase_epochs: int = 15
    unfreeze_layers: int = 2                    # Unfreeze last N encoder layers
    unfreeze_lr: float = 2e-5                   # Learning rate for unfrozen backbone
    head_unfreeze_lr: float = 1e-5              # Fine-tuning learning rate for heads
    
    # Multi-task loss weights (CE has higher raw value than BCE, so boost st & tg)
    loss_st_weight: float = 1.5
    loss_hs_weight: float = 1.0
    loss_tg_weight: float = 1.5
    
    # Target and label mapping constants
    id_order: List[str] = field(default_factory=lambda: list(ID_ORDER))
    scope_dim: int = SCOPE_DIM
    target_dim: int = TARGET_DIM
    hate2idx: Dict[str, int] = field(default_factory=lambda: dict(HATE2IDX))
    
    # Paths & execution
    data_dir: str = "data"
    output_dir: str = "checkpoints"
    save_predictions: bool = True
    val_split_ratio: float = 0.1
    random_seed: int = 42
    device: Optional[str] = None                # Auto-detected if None ('cuda', 'mps', 'cpu')
    num_workers: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
