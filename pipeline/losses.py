import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional class-weighting (alpha) and label smoothing.
    
    Formula:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
        
    Args:
        gamma (float): Focusing parameter (gamma >= 0). 
                       When gamma = 0, Focal Loss is equivalent to CrossEntropyLoss.
                       Higher gamma puts more emphasis on hard/misclassified examples.
        alpha (Tensor or List[float], optional): Class balancing weights. 
                       Shape [C] matching the number of classes.
        label_smoothing (float): Label smoothing epsilon (0.0 to 1.0).
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[torch.Tensor, List[float]]] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits tensor of shape [B, C]
            targets: Ground truth class indices of shape [B]
        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-sample loss [B] (if reduction='none')
        """
        B, C = inputs.shape
        log_p = F.log_softmax(inputs, dim=-1) # [B, C]
        p = torch.exp(log_p)                  # [B, C]

        # Gather target probabilities and log probabilities: [B]
        target_p = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)       # [B]
        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # [B]

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_p, self.gamma) # [B]

        # Standard hard-target CE loss per sample: - log(p_t)
        ce_loss = - target_log_p # [B]

        # Apply label smoothing if requested
        if self.label_smoothing > 0.0:
            # smoothed CE = (1 - eps) * CE(target) + eps * (- mean(log_p))
            smooth_loss = - log_p.mean(dim=-1) # [B]
            ce_loss = (1.0 - self.label_smoothing) * ce_loss + self.label_smoothing * smooth_loss

        # Apply focal modulation
        focal_loss = focal_weight * ce_loss # [B]

        # Apply alpha class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets] # [B]
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
