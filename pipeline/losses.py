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
        B, C = inputs.shape
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        p = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)

        target_p = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_p = torch.clamp(target_p, min=1e-7, max=1.0 - 1e-7)
        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = torch.pow(1.0 - target_p, self.gamma)
        ce_loss = - target_log_p

        if self.label_smoothing > 0.0:
            smooth_loss = - log_p.mean(dim=-1)
            ce_loss = (1.0 - self.label_smoothing) * ce_loss + self.label_smoothing * smooth_loss

        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing support and optional class weighting.
    """
    def __init__(
        self,
        label_smoothing: float = 0.05,
        weights: Optional[Union[torch.Tensor, List[float]]] = None,
        reduction: str = "mean",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32)
            if device is not None:
                weights = weights.to(device)
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weights,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )


class UnidirectionalKLDivergenceLoss(nn.Module):
    """
    Unidirectional KL Divergence Consistency Distillation Loss with Stop-Gradient.
    """
    def __init__(self, temperature: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        s_logits = student_logits / self.temperature
        t_logits = teacher_logits.detach() / self.temperature

        log_p_student = F.log_softmax(s_logits, dim=-1)
        p_teacher = F.softmax(t_logits, dim=-1)
        p_teacher = torch.clamp(p_teacher, min=1e-7, max=1.0 - 1e-7)

        kl_per_sample = torch.sum(p_teacher * (torch.log(p_teacher) - log_p_student), dim=-1)
        kl_per_sample = (self.temperature ** 2) * kl_per_sample

        if self.reduction == "mean":
            return kl_per_sample.mean()
        elif self.reduction == "sum":
            return kl_per_sample.sum()
        return kl_per_sample


class PrivilegedConsistencyTaskBLoss(nn.Module):
    """
    Unified Multi-Objective Loss for Privileged Information Training:
        L_total = L_task(y, p_u) + lambda_c * L_task(y, p_c) + lambda_cons * KL(sg(p_c) || p_u)
    """
    def __init__(
        self,
        base_criterion: nn.Module,
        temperature: float = 1.0
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.kl_criterion = UnidirectionalKLDivergenceLoss(temperature=temperature)

    def forward(
        self,
        logits_u: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
        logits_c: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        lambda_c: float = 0.0,
        lambda_cons: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(logits_u, list):
            losses_u = [self.base_criterion(b, targets) for b in logits_u]
            loss_u = torch.mean(torch.stack(losses_u))
            eval_logits_u = torch.mean(torch.stack(logits_u, dim=0), dim=0)
        else:
            loss_u = self.base_criterion(logits_u, targets)
            eval_logits_u = logits_u

        if logits_c is None or (lambda_c <= 0.0 and lambda_cons <= 0.0):
            return loss_u, {
                'loss_total': float(loss_u.detach().item()),
                'loss_u': float(loss_u.detach().item()),
                'loss_c': 0.0,
                'loss_cons': 0.0
            }

        if isinstance(logits_c, list):
            losses_c = [self.base_criterion(b, targets) for b in logits_c]
            loss_c = torch.mean(torch.stack(losses_c))
            eval_logits_c = torch.mean(torch.stack(logits_c, dim=0), dim=0)
        else:
            loss_c = self.base_criterion(logits_c, targets)
            eval_logits_c = logits_c

        loss_cons = self.kl_criterion(eval_logits_u, eval_logits_c)
        total_loss = loss_u + (lambda_c * loss_c) + (lambda_cons * loss_cons)

        return total_loss, {
            'loss_total': float(total_loss.detach().item()),
            'loss_u': float(loss_u.detach().item()),
            'loss_c': float(loss_c.detach().item()),
            'loss_cons': float(loss_cons.detach().item())
        }


class MoELoadBalanceLoss(nn.Module):
    def __init__(self, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, gates: torch.Tensor) -> torch.Tensor:
        if gates is None or gates.shape[0] == 0:
            return torch.tensor(0.0, device=gates.device if gates is not None else 'cpu')
        mean_gates = torch.mean(gates, dim=0)
        loss = self.num_experts * torch.sum(mean_gates ** 2) - 1.0
        return torch.clamp(loss, min=0.0)


class TaskBLoss(nn.Module):
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
        if isinstance(logits, list):
            branch_losses = [self.base_criterion(branch_logit, targets) for branch_logit in logits]
            cls_loss = torch.mean(torch.stack(branch_losses))
        else:
            cls_loss = self.base_criterion(logits, targets)

        if gates is not None and self.loss_balance_weight > 0.0:
            bal_loss = self.load_balance(gates)
            return cls_loss + self.loss_balance_weight * bal_loss

        return cls_loss


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


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for class imbalance and hard example mining:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        eps = 1e-7
        p = torch.clamp(p, min=eps, max=1.0 - eps)

        # p_t: probability of true class
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
            reduction='none'
        )
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HierarchicalTaskBLoss(nn.Module):
    """
    Multi-Task Conditional Hierarchical Loss with optional Focal Loss for Hate Speech Detection:
        L_total = alpha * L_binary(is_hate) + beta * I_{is_hate=1} * L_fine(implicit vs explicit) + gamma * L_joint(3-class Focal/NLL)
    """
    def __init__(
        self,
        alpha: float = 0.50,
        beta: float = 0.50,
        gamma: float = 1.00,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.05,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.use_focal = use_focal
        self.label_smoothing = float(label_smoothing)

        # Binary hate weight (w_hate / w_no)
        pos_weight = None
        if class_weights is not None and len(class_weights) >= 3:
            w_no = class_weights[0]
            w_hate = (class_weights[1] + class_weights[2]) / 2.0
            pos_weight = torch.tensor([w_hate / max(1e-5, w_no)], dtype=torch.float32)
            if device is not None:
                pos_weight = pos_weight.to(device)

        if use_focal:
            self.binary_hate_loss = BinaryFocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
            self.binary_fine_loss = BinaryFocalLoss(gamma=focal_gamma)
            self.joint_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights, label_smoothing=label_smoothing)
        else:
            self.binary_hate_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.binary_fine_loss = nn.BCEWithLogitsLoss()
            self.joint_loss = None

    def forward(
        self,
        logits_or_dict: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        targets: [B] in {0: no, 1: implicit, 2: explicit}
        """
        if isinstance(logits_or_dict, dict):
            logit_hate = logits_or_dict['logit_hate'].squeeze(-1) # [B]
            logit_type = logits_or_dict['logit_type'].squeeze(-1) # [B]
            probs_3cls = logits_or_dict.get('probs')              # [B, 3]
            compound_logits = logits_or_dict.get('compound_logits') # [B, 3]
        else:
            # If standard 3-class logits passed, treat as joint
            return F.cross_entropy(logits_or_dict, targets, label_smoothing=self.label_smoothing)

        # 1. Binary target: 0 for 'no', 1 for 'implicit' or 'explicit'
        is_hate_target = (targets > 0).float()
        l_binary = self.binary_hate_loss(logit_hate, is_hate_target)

        # 2. Conditional fine-grained target: calculated ONLY on hateful samples (targets > 0)
        # 1.0 for 'implicit' (targets==1), 0.0 for 'explicit' (targets==2)
        hate_mask = (targets > 0)
        if hate_mask.sum() > 0:
            hate_sub_targets = (targets[hate_mask] == 1).float()
            hate_sub_logits = logit_type[hate_mask]
            l_fine = self.binary_fine_loss(hate_sub_logits, hate_sub_targets)
        else:
            l_fine = torch.tensor(0.0, device=targets.device, dtype=logit_type.dtype)

        # 3. Compound 3-class joint loss (Focal or NLL)
        if self.use_focal and compound_logits is not None:
            l_joint = self.joint_loss(compound_logits, targets)
        elif probs_3cls is not None:
            eps = 1e-7
            probs_clamped = torch.clamp(probs_3cls, min=eps, max=1.0 - eps)
            log_probs = torch.log(probs_clamped)
            l_joint = F.nll_loss(log_probs, targets)
        else:
            l_joint = torch.tensor(0.0, device=targets.device)

        total_loss = self.alpha * l_binary + self.beta * l_fine + self.gamma * l_joint
        return total_loss


def build_loss_fn(
    loss_type: str = "focal",
    class_weights: Optional[List[float]] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
    device: Optional[torch.device] = None
) -> nn.Module:
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
