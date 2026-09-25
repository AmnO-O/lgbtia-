from typing import Tuple, Dict
import torch
import torch.nn as nn
from .config import PipelineConfig

class MultiTaskLoss(nn.Module):
    """
    Weighted combination of subtask losses for StereoQueerEval:
      - Stereotype (Binary classification): BCEWithLogitsLoss
      - Hate Speech (3-class classification): CrossEntropyLoss
      - Target (10-dimensional multi-label bitmask): BCEWithLogitsLoss
    
    Weights account for the larger magnitude of CrossEntropy vs BCE to ensure
    balanced gradient descent across all three subtasks.
    """
    def __init__(self, config: PipelineConfig):
        super(MultiTaskLoss, self).__init__()
        self.target_task = config.target_task
        self.st_weight = config.loss_st_weight
        self.hs_weight = config.loss_hs_weight
        self.tg_weight = config.loss_tg_weight

        self.loss_st_fn = nn.BCEWithLogitsLoss()
        self.loss_hs_fn = nn.CrossEntropyLoss()
        self.loss_tg_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        st_logits: torch.Tensor,
        hs_logits: torch.Tensor,
        tg_logits: torch.Tensor,
        st_true: torch.Tensor,
        hs_true: torch.Tensor,
        tg_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_st = self.loss_st_fn(st_logits.squeeze(-1), st_true)
        loss_hs = self.loss_hs_fn(hs_logits, hs_true)
        loss_tg = self.loss_tg_fn(tg_logits, tg_true)

        if self.target_task == 'st':
            total_loss = loss_st
        elif self.target_task == 'hs':
            total_loss = loss_hs
        elif self.target_task == 'tg':
            total_loss = loss_tg
        else:
            total_loss = (
                self.st_weight * loss_st +
                self.hs_weight * loss_hs +
                self.tg_weight * loss_tg
            )

        breakdown = {
            "loss_total": total_loss.item(),
            "loss_st": loss_st.item(),
            "loss_hs": loss_hs.item(),
            "loss_tg": loss_tg.item()
        }
        return total_loss, breakdown
