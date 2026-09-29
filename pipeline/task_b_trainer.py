"""
Task B Specialized Trainer with Additive Latent Privileged Guidance,
Curriculum Annealing Scheduler, Unidirectional Consistency Distillation,
Fast Gradient Method (FGM) Adversarial Regularization, and Multi-Sample Dropout.
"""

import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch._dynamo
except ImportError:
    torch._dynamo = None
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List, Union

from .config import PipelineConfig, IDX2HATE
from .losses import build_loss_fn, PrivilegedConsistencyTaskBLoss, TaskBLoss
from .metrics import compute_classification_metrics
from .models.task_b_class_aware import TaskBClassAwareAttentionModel
from .models.mmbert import unfreeze_last_n
from .scheduler import CosineCurriculumAnnealingScheduler


class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training on Transformer Word Embeddings.
    """
    def __init__(self, model: nn.Module, epsilon: float = 1.0, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.emb_name in name or "tok_embeddings" in name or "word_embeddings" in name) and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.emb_name in name or "tok_embeddings" in name or "word_embeddings" in name) and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class TaskBTrainer:
    """
    Trainer for Task B with Additive Latent Privileged Guidance:
      - Curriculum Annealing (Alpha -> 0, Lambda_C -> 0, Lambda_Cons -> 0)
      - Unidirectional Consistency Distillation with stop_gradient(p_c)
      - 2-Phase Backbone Fine-Tuning (Frozen -> Layer-Unfrozen: top N layers)
      - Early Stopping with Configurable Patience
      - Multi-Sample Dropout Loss Averaging
      - Fast Gradient Method (FGM) Adversarial Regularization
      - Mixed Precision (AMP)
    """
    def __init__(
        self,
        model: TaskBClassAwareAttentionModel,
        config: PipelineConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        df_val: Optional[pd.DataFrame] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.df_val = df_val

        if device is not None:
            self.device = device
        elif config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.use_amp = use_amp and (self.device.type == 'cuda')
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_amp)

        # Base Loss Function & Privileged Multi-Objective Criterion
        base_loss_fn = build_loss_fn(
            loss_type=config.loss_type,
            class_weights=config.class_weights,
            gamma=getattr(config, 'focal_gamma', 2.0),
            label_smoothing=config.label_smoothing,
            device=self.device
        )
        self.criterion = PrivilegedConsistencyTaskBLoss(
            base_criterion=base_loss_fn,
            temperature=getattr(config, 'kd_temperature', 1.0)
        )

        # FGM
        if getattr(config, 'use_fgm', False):
            self.fgm = FGM(self.model, epsilon=getattr(config, 'fgm_epsilon', 1.0))
        else:
            self.fgm = None

        self.best_macro_f1 = 0.0
        self.best_checkpoint_path = ""
        self.history: List[Dict[str, Any]] = []

    def build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight", "norm.weight", "norm_q.weight", "norm_kv.weight", "norm_self.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )

    def build_scheduler(self, optimizer: torch.optim.Optimizer, num_epochs: int):
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * 0.10)
        
        try:
            from transformers import get_cosine_schedule_with_warmup
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=0.01
            )
        except Exception:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps, 1),
                eta_min=1e-7
            )

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        apply_fgm: bool = True,
        hint_alpha: float = 0.0,
        lambda_c: float = 0.0,
        lambda_cons: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        all_train_preds: List[np.ndarray] = []
        all_train_targets: List[np.ndarray] = []

        for batch in self.train_loader:
            if len(batch) >= 8:
                input_ids, attention_mask, role_ids, hint_ids, hint_mask, _, hs_labels, _ = batch[:8]
                hint_ids = hint_ids.to(self.device, non_blocking=True)
                hint_mask = hint_mask.to(self.device, non_blocking=True)
            else:
                input_ids, attention_mask, role_ids, _, hs_labels, _ = batch[:6]
                hint_ids, hint_mask = None, None

            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            role_ids = role_ids.to(self.device, non_blocking=True)
            hs_labels = hs_labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                # 1. Unconditional Forward Pass (p_u)
                out_u = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    role_ids=role_ids,
                    hint_ids=None,
                    hint_mask=None,
                    hint_alpha=0.0,
                    return_all_msd_logits=True
                )
                logits_u = out_u[0] if isinstance(out_u, tuple) else out_u

                # 2. Conditional Forward Pass (p_c)
                logits_c = None
                if hint_ids is not None and hint_alpha > 0.0 and (lambda_c > 0.0 or lambda_cons > 0.0):
                    out_c = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        role_ids=role_ids,
                        hint_ids=hint_ids,
                        hint_mask=hint_mask,
                        hint_alpha=hint_alpha,
                        return_all_msd_logits=True
                    )
                    logits_c = out_c[0] if isinstance(out_c, tuple) else out_c

                # 3. Multi-Objective Loss
                loss, loss_breakdown = self.criterion(
                    logits_u=logits_u,
                    targets=hs_labels,
                    logits_c=logits_c,
                    lambda_c=lambda_c,
                    lambda_cons=lambda_cons
                )

            with torch.no_grad():
                eval_logits = torch.mean(torch.stack(logits_u, dim=0), dim=0) if isinstance(logits_u, list) else logits_u
                preds = torch.argmax(eval_logits, dim=-1)
                all_train_preds.append(preds.detach().cpu().numpy())
                all_train_targets.append(hs_labels.detach().cpu().numpy())

            is_unscaled = False

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if apply_fgm and self.fgm is not None:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    is_unscaled = True
                
                self.fgm.attack()
                
                with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                    adv_out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        role_ids=role_ids,
                        hint_ids=None,
                        hint_mask=None,
                        hint_alpha=0.0,
                        return_all_msd_logits=False
                    )
                    adv_logits = adv_out[0] if isinstance(adv_out, tuple) else adv_out
                    adv_loss, _ = self.criterion(logits_u=adv_logits, targets=hs_labels)
                
                if self.use_amp:
                    self.scaler.scale(adv_loss).backward()
                else:
                    adv_loss.backward()
                    
                self.fgm.restore()

            if self.use_amp:
                if self.config.clip_grad_norm > 0:
                    if not is_unscaled:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(self.train_loader), 1)
        y_train_true = np.concatenate(all_train_targets, axis=0) if all_train_targets else np.array([])
        y_train_pred = np.concatenate(all_train_preds, axis=0) if all_train_preds else np.array([])
        train_metrics = compute_classification_metrics(y_train_true, y_train_pred, task="hs") if len(y_train_true) > 0 else {}
        
        return avg_loss, train_metrics

    @torch.no_grad()
    def eval_epoch(self) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluation is STRICTLY UNCONDITIONAL (hint_alpha=0.0).
        """
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.inference_mode():
            for batch in self.val_loader:
                if len(batch) >= 8:
                    input_ids, attention_mask, role_ids, _, _, _, hs_labels, _ = batch[:8]
                else:
                    input_ids, attention_mask, role_ids, _, hs_labels, _ = batch[:6]

                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)
                role_ids = role_ids.to(self.device, non_blocking=True)
                hs_labels = hs_labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        role_ids=role_ids,
                        hint_ids=None,
                        hint_mask=None,
                        hint_alpha=0.0,
                        return_all_msd_logits=False
                    )
                    logits = out[0] if isinstance(out, tuple) else out
                    loss, _ = self.criterion(logits_u=logits, targets=hs_labels)

                total_loss += loss.item()
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(hs_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(len(self.val_loader), 1)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = compute_classification_metrics(y_true, y_pred, y_prob, task="hs")
        return avg_loss, metrics, y_pred, y_prob

    def fit(self) -> Dict[str, Any]:
        """
        Executes complete training lifecycle with Curriculum Annealing & Early Stopping.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        patience_limit = getattr(self.config, 'patience', 5)
        model_name = self.model.__class__.__name__

        use_privileged = getattr(self.config, 'use_privileged_guidance', False)
        start_alpha = getattr(self.config, 'hint_start_alpha', 1.0) if use_privileged else 0.0
        
        curriculum_scheduler = CosineCurriculumAnnealingScheduler(
            total_epochs=self.config.unfreeze_phase_epochs,
            start_alpha=start_alpha,
            end_alpha=0.0,
            lambda_c_max=getattr(self.config, 'lambda_c', 0.40),
            lambda_cons_max=getattr(self.config, 'lambda_cons', 0.30)
        )

        print(f"\n=======================================================")
        print(f" Task B Training Pipeline [{model_name}]")
        print(f" Device: {self.device} | AMP: {self.use_amp} | Patience: {patience_limit}")
        print(f" Privileged Teacher Guidance: {'ENABLED' if use_privileged else 'DISABLED (Pure Baseline Mode)'}")
        print(f"=======================================================")

        last_metrics: Dict[str, float] = {}
        self.history = []

        # PHASE 1: Warmup Heads (mmBERT Backbone Frozen)
        if self.config.two_phase:
            print(f"\n>>> [Phase 1/2] Training Heads (mmBERT Backbone Frozen) for {self.config.freeze_phase_epochs} epochs...")
            for param in self.model.mmbert.parameters():
                param.requires_grad = False
                
            optimizer = self.build_optimizer(lr=self.config.learning_rate)
            scheduler = self.build_scheduler(optimizer, self.config.freeze_phase_epochs)
            p1_patience_counter = 0

            for epoch in range(1, self.config.freeze_phase_epochs + 1):
                t0 = time.time()
                train_loss, train_metrics = self.train_epoch(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    apply_fgm=False,
                    hint_alpha=start_alpha,
                    lambda_c=0.40 if use_privileged else 0.0,
                    lambda_cons=0.0
                )
                val_loss, val_metrics, _, _ = self.eval_epoch()
                elapsed = time.time() - t0

                # Universal metric extraction (supports hs_macro_f1 or macro_f1, hs_f1_implicit, hs_f1_no, hs_f1_explicit)
                macro_f1 = val_metrics.get('hs_macro_f1', val_metrics.get('macro_f1', 0.0))
                imp_f1 = val_metrics.get('hs_f1_implicit', val_metrics.get('class_f1_yes_implicit', 0.0))
                no_f1 = val_metrics.get('hs_f1_no', val_metrics.get('class_f1_no', 0.0))
                exp_f1 = val_metrics.get('hs_f1_explicit', val_metrics.get('class_f1_yes_explicit', 0.0))
                val_acc = val_metrics.get('hs_acc', val_metrics.get('accuracy', 0.0))

                self.history.append({
                    'phase': 1,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'hs_macro_f1': macro_f1,
                    'hs_f1_no': no_f1,
                    'hs_f1_implicit': imp_f1,
                    'hs_f1_explicit': exp_f1,
                    'hs_acc': val_acc
                })

                print(f"  [P1 Epoch {epoch:02d}/{self.config.freeze_phase_epochs:02d}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Macro-F1: {macro_f1:.4f} | No-F1: {no_f1:.4f} | Imp-F1: {imp_f1:.4f} | Exp-F1: {exp_f1:.4f} | Acc: {val_acc:.4f} [{elapsed:.1f}s]")

                if macro_f1 > self.best_macro_f1:
                    self.best_macro_f1 = macro_f1
                    self.best_checkpoint_path = os.path.join(self.config.output_dir, "best_task_b_model.pt")
                    torch.save(self.model.state_dict(), self.best_checkpoint_path)
                    print(f"    ⭐ New Best Task B Macro-F1: {macro_f1:.4f} -> Saved checkpoint.")
                    p1_patience_counter = 0
                else:
                    p1_patience_counter += 1

                last_metrics = val_metrics

        # PHASE 2: Differential Fine-Tuning + Curriculum Annealing
        print(f"\n>>> [Phase 2/2] Fine-Tuning Top {self.config.unfreeze_layers} Layers for {self.config.unfreeze_phase_epochs} epochs...")
        unfreeze_last_n(self.model.mmbert, self.config.unfreeze_layers)

        backbone_params = [p for p in self.model.mmbert.parameters() if p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith('mmbert') and p.requires_grad]
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.config.unfreeze_lr, 'weight_decay': self.config.weight_decay},
            {'params': head_params, 'lr': self.config.head_unfreeze_lr, 'weight_decay': self.config.weight_decay},
        ], betas=(0.9, 0.98), eps=1e-6)

        scheduler = self.build_scheduler(optimizer, self.config.unfreeze_phase_epochs)
        p2_patience_counter = 0

        for epoch in range(1, self.config.unfreeze_phase_epochs + 1):
            t0 = time.time()
            
            curr_params = curriculum_scheduler.step(epoch - 1)
            h_alpha = curr_params['alpha']
            l_c = curr_params['lambda_c']
            l_cons = curr_params['lambda_cons']

            train_loss, train_metrics = self.train_epoch(
                optimizer=optimizer,
                scheduler=scheduler,
                apply_fgm=getattr(self.config, 'use_fgm', False),
                hint_alpha=h_alpha,
                lambda_c=l_c,
                lambda_cons=l_cons
            )
            val_loss, val_metrics, _, _ = self.eval_epoch()
            elapsed = time.time() - t0

            macro_f1 = val_metrics.get('hs_macro_f1', val_metrics.get('macro_f1', 0.0))
            imp_f1 = val_metrics.get('hs_f1_implicit', val_metrics.get('class_f1_yes_implicit', 0.0))
            no_f1 = val_metrics.get('hs_f1_no', val_metrics.get('class_f1_no', 0.0))
            exp_f1 = val_metrics.get('hs_f1_explicit', val_metrics.get('class_f1_yes_explicit', 0.0))
            val_acc = val_metrics.get('hs_acc', val_metrics.get('accuracy', 0.0))

            global_epoch = (self.config.freeze_phase_epochs if self.config.two_phase else 0) + epoch
            self.history.append({
                'phase': 2,
                'epoch': global_epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'hs_macro_f1': macro_f1,
                'hs_f1_no': no_f1,
                'hs_f1_implicit': imp_f1,
                'hs_f1_explicit': exp_f1,
                'hs_acc': val_acc
            })

            print(f"  [P2 Epoch {epoch:02d}/{self.config.unfreeze_phase_epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Macro-F1: {macro_f1:.4f} | No-F1: {no_f1:.4f} | Imp-F1: {imp_f1:.4f} | Exp-F1: {exp_f1:.4f} | Acc: {val_acc:.4f} | "
                  f"α: {h_alpha:.2f} [{elapsed:.1f}s]")

            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                self.best_checkpoint_path = os.path.join(self.config.output_dir, "best_task_b_model.pt")
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
                print(f"    ⭐ New Best Task B Macro-F1: {macro_f1:.4f} -> Saved checkpoint.")
                p2_patience_counter = 0
            else:
                p2_patience_counter += 1
                if p2_patience_counter >= patience_limit:
                    print(f"\n⏹ Early stopping triggered after {patience_limit} epochs without improvement.")
                    break

            last_metrics = val_metrics

        print(f"\n=======================================================")
        print(f" Training Complete! Best Validation Macro-F1: {self.best_macro_f1:.4f}")
        print(f" Best Checkpoint Saved: {self.best_checkpoint_path}")
        print(f"=======================================================")

        return {
            'best_macro_f1': self.best_macro_f1,
            'best_checkpoint_path': self.best_checkpoint_path,
            'checkpoint_path': self.best_checkpoint_path,
            'final_metrics': last_metrics,
            'last_metrics': last_metrics,
            'history': self.history
        }
