import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Tuple
from .config import PipelineConfig


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

        # Numerical stability clamp to avoid 0 * inf = NaN under AMP/float16
        p = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)

        # Gather target probabilities and log probabilities: [B]
        target_p = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_p = torch.clamp(target_p, min=1e-7, max=1.0 - 1e-7)
        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_p, self.gamma) # [B]

        # Standard hard-target CE loss per sample: - log(p_t)
        ce_loss = - target_log_p # [B]

        # Apply label smoothing if requested
        if self.label_smoothing > 0.0:
            smooth_loss = - log_p.mean(dim=-1) # [B]
            ce_loss = (1.0 - self.label_smoothing) * ce_loss + self.label_smoothing * smooth_loss

        # Apply focal modulation
        focal_loss = focal_weight * ce_loss # [B]

        # Apply alpha class weights if provided (device-safe buffer)
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha[targets] # [B]
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MoELoadBalanceLoss(nn.Module):
    """
    Mixture-of-Experts (MoE) Router Load Balancing Loss.
    
    Prevents expert collapse where the router overwhelmingly selects only one expert.
    Computes normalized square-deviation across the batch:
        L_balance = N * sum_{i=1}^N (mean(g_i)^2) - 1.0
        
    When distribution across experts is uniform (mean(g_i) = 1/N), loss is 0.0.
    """
    def __init__(self, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, gates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gates: Tensor of routing probabilities [B, num_experts]
        Returns:
            Scalar load balance regularization loss
        """
        if gates is None or gates.shape[0] == 0:
            return torch.tensor(0.0, device=gates.device if gates is not None else 'cpu')
        
        # Batch-average expert utilization: [num_experts]
        mean_gates = torch.mean(gates, dim=0)
        
        # L_balance = N * sum(mean_gates^2) - 1.0
        loss = self.num_experts * torch.sum(mean_gates ** 2) - 1.0
        return torch.clamp(loss, min=0.0)


class TaskBLoss(nn.Module):
    """
    Unified Task B Loss combining multi-class focal loss (with Multi-Sample Dropout averaging)
    and optional MoE Router Load Balancing regularization.
    """
    def __init__(
        self,
        base_criterion: nn.Module,
        num_experts: int = 4,
        loss_balance_weight: float = 0.01
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.load_balance = MoELoadBalanceLoss(num_experts=num_experts)
        self.loss_balance_weight = loss_balance_weight

    def forward(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
        gates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: Single tensor [B, 3] or list of MSD branch tensors
            targets: Class target indices [B]
            gates: Optional router gates [B, num_experts]
        """
        if isinstance(logits, list):
            branch_losses = [self.base_criterion(branch_logit, targets) for branch_logit in logits]
            cls_loss = torch.mean(torch.stack(branch_losses))
        else:
            cls_loss = self.base_criterion(logits, targets)

        if gates is not None and self.loss_balance_weight > 0.0:
            bal_loss = self.load_balance(gates)
            return cls_loss + self.loss_balance_weight * bal_loss

        return cls_loss


def build_loss_fn(
    loss_type: str = "focal",
    class_weights: Optional[List[float]] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory builder for Task B base loss functions (Focal Loss or CrossEntropyLoss).
    """
    if loss_type == "focal":
        alpha = None
        if class_weights is not None:
            alpha = torch.tensor(class_weights, dtype=torch.float32)
            if device is not None:
                alpha = alpha.to(device)
        return FocalLoss(
            gamma=gamma,
            alpha=alpha,
            label_smoothing=label_smoothing
        )
    else:
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            if device is not None:
                weight_tensor = weight_tensor.to(device)
        return nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=label_smoothing
        )


class MultiTaskLoss(nn.Module):
    """
    Weighted Multi-Task Loss for StereoQueer:
      - Stereotype Presence (ST): BCEWithLogitsLoss
      - Hate Speech Type (HS): CrossEntropyLoss or FocalLoss
      - Stereotype Target Group (TG): BCEWithLogitsLoss
    """
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.w_st = config.loss_st_weight
        self.w_hs = config.loss_hs_weight
        self.w_tg = config.loss_tg_weight
        
        self.loss_st = nn.BCEWithLogitsLoss()
        
        if getattr(config, 'loss_type', 'focal') == 'focal':
            class_weights = getattr(config, 'class_weights', None)
            focal_gamma = getattr(config, 'focal_gamma', 2.0)
            label_smoothing = getattr(config, 'label_smoothing', 0.05)
            self.loss_hs = FocalLoss(
                gamma=focal_gamma,
                alpha=class_weights,
                label_smoothing=label_smoothing
            )
        else:
            class_weights = getattr(config, 'class_weights', None)
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
            self.loss_hs = nn.CrossEntropyLoss(weight=weight_tensor)
            
        self.loss_tg = nn.BCEWithLogitsLoss()

    def forward(
        self,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Accepts:
          - (preds_dict, targets_dict)
          - (st_logits, hs_logits, tg_logits, st_tgt, hs_tgt, tg_tgt)
        Returns:
          - (total_loss, {'total': total_loss, 'st': l_st, 'hs': l_hs, 'tg': l_tg})
        """
        if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            preds, targets = args[0], args[1]
            st_pred, hs_pred, tg_pred = preds['st'], preds['hs'], preds['tg']
            st_tgt, hs_tgt, tg_tgt = targets['st'], targets['hs'], targets['tg']
        elif len(args) == 6:
            st_pred, hs_pred, tg_pred, st_tgt, hs_tgt, tg_tgt = args
        elif 'preds' in kwargs and 'targets' in kwargs:
            preds, targets = kwargs['preds'], kwargs['targets']
            st_pred, hs_pred, tg_pred = preds['st'], preds['hs'], preds['tg']
            st_tgt, hs_tgt, tg_tgt = targets['st'], targets['hs'], targets['tg']
        else:
            raise ValueError("MultiTaskLoss expects either 2 dicts (preds, targets) or 6 positional tensors.")

        # Ensure matching shapes for ST binary classification
        st_pred = st_pred.squeeze(-1) if st_pred.ndim > 1 and st_pred.shape[-1] == 1 else st_pred
        st_tgt = st_tgt.view_as(st_pred)

        l_st = self.loss_st(st_pred, st_tgt)
        l_hs = self.loss_hs(hs_pred, hs_tgt)
        l_tg = self.loss_tg(tg_pred, tg_tgt)

        total = self.w_st * l_st + self.w_hs * l_hs + self.w_tg * l_tg
        loss_dict = {
            'total': total,
            'st': l_st,
            'hs': l_hs,
            'tg': l_tg,
        }

        return total, loss_dict
