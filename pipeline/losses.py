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

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], targets: torch.Tensor) -> torch.Tensor:
        is_already_log_probs = False
        if isinstance(inputs, dict):
            if 'log_probs' in inputs and inputs['log_probs'] is not None:
                inputs = inputs['log_probs']
                is_already_log_probs = True
            elif 'compound_logits' in inputs and inputs['compound_logits'] is not None:
                inputs = inputs['compound_logits']
                is_already_log_probs = True
            elif 'probs' in inputs and inputs['probs'] is not None:
                eps = 1e-7
                probs_clamped = torch.clamp(inputs['probs'], min=eps, max=1.0 - eps)
                inputs = torch.log(probs_clamped)
                is_already_log_probs = True
            elif 'logits' in inputs:
                inputs = inputs['logits']

        B, C = inputs.shape
        if is_already_log_probs:
            # Clamp log_p to prevent -inf / NaN in AMP
            log_p = torch.clamp(inputs, min=-30.0, max=0.0)
        else:
            log_p = F.log_softmax(inputs, dim=-1)

        # 1. Exact valid log probability of target class
        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # 2. Probability p_t for focal weight with lower/upper clamp to prevent underflow/overflow
        target_p = torch.clamp(torch.exp(target_log_p), min=1e-7, max=1.0)

        focal_weight = torch.pow(torch.clamp(1.0 - target_p, min=0.0, max=1.0), self.gamma)
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

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], targets: torch.Tensor) -> torch.Tensor:
        is_already_log_probs = False
        if isinstance(inputs, dict):
            if 'log_probs' in inputs and inputs['log_probs'] is not None:
                inputs = inputs['log_probs']
                is_already_log_probs = True
            elif 'compound_logits' in inputs and inputs['compound_logits'] is not None:
                inputs = inputs['compound_logits']
                is_already_log_probs = True
            elif 'probs' in inputs and inputs['probs'] is not None:
                eps = 1e-7
                probs_clamped = torch.clamp(inputs['probs'], min=eps, max=1.0 - eps)
                inputs = torch.log(probs_clamped)
                is_already_log_probs = True
            elif 'logits' in inputs:
                inputs = inputs['logits']

        if is_already_log_probs:
            log_p = torch.clamp(inputs, min=-30.0, max=0.0)
            if self.label_smoothing > 0.0:
                target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
                smooth_loss = - log_p.mean(dim=-1)
                ce_loss = (1.0 - self.label_smoothing) * (-target_log_p) + self.label_smoothing * smooth_loss
                if self.weights is not None:
                    w = self.weights.to(targets.device)[targets]
                    ce_loss = w * ce_loss
                return ce_loss.mean() if self.reduction == "mean" else ce_loss.sum()
            else:
                return F.nll_loss(log_p, targets, weight=self.weights, reduction=self.reduction)

        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weights,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )


class UnidirectionalKLDivergenceLoss(nn.Module):
    """
    Unidirectional KL Divergence Consistency Distillation Loss with Stop-Gradient on Teacher:
        KL(p_teacher || p_student) = sum p_teacher * (log p_teacher - log p_student)
    Accepts either probabilities or log-probabilities directly.
    """
    def __init__(self, temperature: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor
    ) -> torch.Tensor:
        eps = 1e-7
        p_s = torch.clamp(student_probs, min=eps, max=1.0 - eps)
        p_t = torch.clamp(teacher_probs.detach(), min=eps, max=1.0 - eps)

        if self.temperature != 1.0:
            # Temperature scaling in probability space
            p_s = F.softmax(torch.log(p_s) / self.temperature, dim=-1)
            p_t = F.softmax(torch.log(p_t) / self.temperature, dim=-1)

        kl_per_sample = torch.sum(p_t * (torch.log(p_t) - torch.log(p_s)), dim=-1)
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
        logits_u: Union[torch.Tensor, Dict[str, torch.Tensor], List[Union[torch.Tensor, Dict[str, torch.Tensor]]]],
        targets: torch.Tensor,
        logits_c: Optional[Union[torch.Tensor, Dict[str, torch.Tensor], List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]] = None,
        lambda_c: float = 0.0,
        lambda_cons: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        def _to_probs(x):
            if isinstance(x, dict):
                if x.get('probs') is not None:
                    return torch.clamp(x['probs'], min=1e-7, max=1.0)
                if x.get('log_probs') is not None:
                    return torch.clamp(torch.exp(x['log_probs']), min=1e-7, max=1.0)
                if x.get('compound_logits') is not None:
                    return torch.clamp(torch.exp(x['compound_logits']), min=1e-7, max=1.0)
                if x.get('logits') is not None:
                    return F.softmax(x['logits'], dim=-1)
            return F.softmax(x, dim=-1)

        if isinstance(logits_u, list):
            losses_u = [self.base_criterion(b, targets) for b in logits_u]
            loss_u = torch.mean(torch.stack(losses_u))
            eval_probs_u = torch.mean(torch.stack([_to_probs(b) for b in logits_u], dim=0), dim=0)
        else:
            loss_u = self.base_criterion(logits_u, targets)
            eval_probs_u = _to_probs(logits_u)

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
            eval_probs_c = torch.mean(torch.stack([_to_probs(b) for b in logits_c], dim=0), dim=0)
        else:
            loss_c = self.base_criterion(logits_c, targets)
            eval_probs_c = _to_probs(logits_c)

        loss_cons = self.kl_criterion(eval_probs_u, eval_probs_c)
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
    def __init__(
        self,
        loss_st_weight: float = 1.5,
        loss_hs_weight: float = 1.0,
        loss_tg_weight: float = 1.5,
        loss_type: str = "focal",
        class_weights: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        target_dim: int = 10,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.loss_st_weight = loss_st_weight
        self.loss_hs_weight = loss_hs_weight
        self.loss_tg_weight = loss_tg_weight
        self.target_dim = target_dim

        self.loss_st = nn.BCEWithLogitsLoss()
        self.loss_hs = build_loss_fn(
            loss_type=loss_type,
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
            device=device
        )
        self.loss_tg = nn.BCEWithLogitsLoss()

    def forward(
        self,
        preds: Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        st_pred, hs_pred, tg_pred = preds
        st_target, hs_target, tg_target = targets

        loss_st = self.loss_st(st_pred.squeeze(-1), st_target.float())
        loss_hs = self.loss_hs(hs_pred, hs_target)
        loss_tg = self.loss_tg(tg_pred, tg_target.float())

        total_loss = (
            self.loss_st_weight * loss_st +
            self.loss_hs_weight * loss_hs +
            self.loss_tg_weight * loss_tg
        )

        breakdown = {
            'loss_total': float(total_loss.detach().item()),
            'loss_st': float(loss_st.detach().item()),
            'loss_hs': float(loss_hs.detach().item()),
            'loss_tg': float(loss_tg.detach().item())
        }
        return total_loss, breakdown


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        eps = 1e-7
        p = torch.clamp(p, min=eps, max=1.0 - eps)

        # p_t: probability of true class
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_weight = torch.pow(torch.clamp(1.0 - p_t, min=0.0, max=1.0), self.gamma)

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
    Hierarchical Loss with Joint 3-Class Compound Objective as Primary,
    and Decoupled Binary/Fine Heads as Auxiliary Regularization:
        L_total = L_joint(3-class) + lambda_hate * L_binary(is_hate) + lambda_type * I_{is_hate=1} * L_fine(implicit vs explicit)
    """
    def __init__(
        self,
        lambda_hate: float = 0.20,
        lambda_type: float = 0.20,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        hate_pos_weight: Optional[float] = None,
        type_pos_weight: Optional[float] = None,
        label_smoothing: float = 0.0,
        device: Optional[torch.device] = None,
        # Backward-compatibility alias params
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ):
        super().__init__()
        # If legacy alpha/beta/gamma provided, normalize with gamma=1.0 as base
        if alpha is not None:
            self.lambda_hate = float(alpha)
        else:
            self.lambda_hate = float(lambda_hate)

        if beta is not None:
            self.lambda_type = float(beta)
        else:
            self.lambda_type = float(lambda_type)

        self.use_focal = use_focal
        self.label_smoothing = float(label_smoothing)

        pos_weight_hate_tensor = None
        if hate_pos_weight is not None:
            pos_weight_hate_tensor = torch.tensor([hate_pos_weight], dtype=torch.float32)
            if device is not None:
                pos_weight_hate_tensor = pos_weight_hate_tensor.to(device)

        pos_weight_type_tensor = None
        if type_pos_weight is not None:
            pos_weight_type_tensor = torch.tensor([type_pos_weight], dtype=torch.float32)
            if device is not None:
                pos_weight_type_tensor = pos_weight_type_tensor.to(device)

        if use_focal:
            self.binary_hate_loss = BinaryFocalLoss(gamma=focal_gamma, pos_weight=pos_weight_hate_tensor)
            self.binary_fine_loss = BinaryFocalLoss(gamma=focal_gamma, pos_weight=pos_weight_type_tensor)
            self.joint_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights, label_smoothing=label_smoothing)
        else:
            self.binary_hate_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_hate_tensor)
            self.binary_fine_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_type_tensor)
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
            log_probs = logits_or_dict.get('log_probs', logits_or_dict.get('compound_logits')) # [B, 3]
        else:
            # If standard 3-class logits passed, treat as joint
            return F.cross_entropy(logits_or_dict, targets, label_smoothing=self.label_smoothing)

        # 1. Primary Objective: Compound 3-Class Joint Loss
        if self.use_focal and log_probs is not None:
            l_joint = self.joint_loss(log_probs, targets)
        elif log_probs is not None:
            l_joint = F.nll_loss(torch.clamp(log_probs, min=-30.0, max=0.0), targets)
        elif probs_3cls is not None:
            eps = 1e-7
            probs_clamped = torch.clamp(probs_3cls, min=eps, max=1.0)
            log_probs_fallback = torch.log(probs_clamped)
            l_joint = F.nll_loss(log_probs_fallback, targets)
        else:
            l_joint = torch.tensor(0.0, device=targets.device)

        # 2. Auxiliary Level 1: Super-Class Binary Hate Loss
        if self.lambda_hate > 0.0:
            is_hate_target = (targets > 0).float()
            l_binary = self.binary_hate_loss(logit_hate, is_hate_target)
        else:
            l_binary = torch.tensor(0.0, device=targets.device)

        # 3. Auxiliary Level 2: Conditional Fine-Grained Implicit vs Explicit Loss
        if self.lambda_type > 0.0:
            hate_mask = (targets > 0)
            if hate_mask.sum() > 0:
                hate_sub_targets = (targets[hate_mask] == 1).float()
                hate_sub_logits = logit_type[hate_mask]
                if self.use_focal:
                    p = torch.sigmoid(hate_sub_logits)
                    eps = 1e-7
                    p = torch.clamp(p, min=eps, max=1.0 - eps)
                    p_t = p * hate_sub_targets + (1.0 - p) * (1.0 - hate_sub_targets)
                    focal_weight = torch.pow(torch.clamp(1.0 - p_t, min=0.0, max=1.0), 2.0)
                    bce = F.binary_cross_entropy_with_logits(hate_sub_logits, hate_sub_targets, reduction='none')
                    l_fine = (focal_weight * bce).sum() / float(targets.shape[0])
                else:
                    l_fine = F.binary_cross_entropy_with_logits(hate_sub_logits, hate_sub_targets, reduction='sum') / float(targets.shape[0])
            else:
                l_fine = torch.tensor(0.0, device=targets.device, dtype=logit_type.dtype)
        else:
            l_fine = torch.tensor(0.0, device=targets.device)

        total_loss = l_joint + (self.lambda_hate * l_binary) + (self.lambda_type * l_fine)
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
