"""
Task B Specialized Trainer with Fast Gradient Method (FGM) Adversarial Regularization & Multi-Sample Dropout.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List, Union

from .config import PipelineConfig
from .losses import build_loss_fn
from .metrics import compute_classification_metrics
from .models.task_b_class_aware import TaskBClassAwareAttentionModel


class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training on Transformer Word Embeddings.
    Perturbs token embeddings in the direction of the loss gradient by scale epsilon:
        delta = epsilon * (grad / ||grad||_2)
    Forces the loss landscape around subtle phrases to remain smooth and prevents overfitting.
    """
    def __init__(self, model: nn.Module, epsilon: float = 1.0, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class TaskBTrainer:
    """
    Trainer for Task B Hate Speech Classification with:
      - 2-Phase Backbone Fine-Tuning (Frozen -> Layer-Unfrozen)
      - Cosine Annealing with Warmup
      - Multi-Sample Dropout Loss Averaging
      - Fast Gradient Method (FGM) Adversarial Regularization
      - Mixed Precision (AMP)
      - Per-Language Validation Metric Tracking & Optional df_val integration
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

        # Loss function
        self.criterion = build_loss_fn(
            loss_type=config.loss_type,
            class_weights=config.class_weights,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            device=self.device
        )

        # Adversarial Trainer
        self.fgm = FGM(
            model=self.model,
            epsilon=config.fgm_epsilon,
            emb_name=config.fgm_emb_name
        ) if getattr(config, 'use_fgm', True) else None

        self.best_macro_f1 = -1.0
        self.best_checkpoint_path = ""
        self.history: List[Dict[str, Any]] = []

    def build_optimizer(self, lr: float, backbone_lr: Optional[float] = None) -> torch.optim.Optimizer:
        """
        Builds an AdamW optimizer with differential learning rates for backbone vs heads.
        """
        backbone_prms = [p for n, p in self.model.named_parameters()
                         if p.requires_grad and n.startswith('mmbert.')]
        head_prms = [p for n, p in self.model.named_parameters()
                     if p.requires_grad and not n.startswith('mmbert.')]

        if backbone_prms and backbone_lr is not None:
            return torch.optim.AdamW([
                {'params': backbone_prms, 'lr': backbone_lr, 'weight_decay': self.config.weight_decay},
                {'params': head_prms, 'lr': lr, 'weight_decay': self.config.weight_decay},
            ], betas=(0.9, 0.98), eps=1e-6)
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )

    def build_scheduler(self, optimizer: torch.optim.Optimizer, num_epochs: int):
        """
        Builds a Cosine Annealing learning rate scheduler with Linear Warmup.
        """
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * 0.10) # 10% warmup
        
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
                T_max=total_steps,
                eta_min=1e-7
            )

    def _compute_loss(self, logits: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor) -> torch.Tensor:
        """Handles single logits tensor or list of Multi-Sample Dropout branch logits."""
        if isinstance(logits, list):
            branch_losses = [self.criterion(branch_logit, targets) for branch_logit in logits]
            return torch.mean(torch.stack(branch_losses))
        return self.criterion(logits, targets)

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        apply_fgm: bool = True
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            input_ids, attention_mask, role_ids, _, hs_labels, _ = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            role_ids = role_ids.to(self.device, non_blocking=True)
            hs_labels = hs_labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            
            # Forward 1: Clean pass
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                logits, _, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    role_ids=role_ids,
                    return_all_msd_logits=True
                )
                loss = self._compute_loss(logits, hs_labels)

            is_unscaled = False

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Forward 2: Fast Gradient Method (Adversarial perturbation)
            if apply_fgm and self.fgm is not None:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    is_unscaled = True
                
                self.fgm.attack() # Inject epsilon * grad into word embeddings
                
                with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                    adv_logits, _, _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        role_ids=role_ids,
                        return_all_msd_logits=False
                    )
                    adv_loss = self._compute_loss(adv_logits, hs_labels)
                
                if self.use_amp:
                    self.scaler.scale(adv_loss).backward()
                else:
                    adv_loss.backward()
                    
                self.fgm.restore() # Restore original unperturbed weights

            # Step optimizer & scaler
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

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def eval_epoch(self) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in self.val_loader:
            input_ids, attention_mask, role_ids, _, hs_labels, _ = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            role_ids = role_ids.to(self.device, non_blocking=True)
            hs_labels = hs_labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                logits, _, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    role_ids=role_ids,
                    return_all_msd_logits=False
                )
                loss = self.criterion(logits, hs_labels)

            total_loss += loss.item()
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(hs_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / max(len(self.val_loader), 1)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = compute_classification_metrics(y_true, y_pred, y_prob, task="hs")

        # Per-language breakdown if df_val is provided and contains 'lang'
        if self.df_val is not None and 'lang' in self.df_val.columns and len(self.df_val) == len(y_true):
            langs = self.df_val['lang'].values
            for l in np.unique(langs):
                idx = (langs == l)
                if idx.sum() > 0:
                    sub_m = compute_classification_metrics(y_true[idx], y_pred[idx], y_prob[idx], task=f"hs_{l}")
                    metrics[f"hs_macro_f1_{l}"] = sub_m[f"hs_{l}_macro_f1"]

        return avg_loss, metrics, y_pred, y_prob

    def train_pipeline(self) -> Dict[str, Any]:
        """
        Executes complete training:
          Phase 1: Frozen backbone (Head warmup + MSD)
          Phase 2: Unfrozen backbone (End-to-end + MSD + FGM Adversarial Regularization)
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(self.config.output_dir, "task_b_best_model.pt")

        print("=" * 70)
        print("STARTING TASK B CLASS-AWARE MULTI-QUERY TRAINING PIPELINE")
        print(f"  Device:         {self.device}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Loss Function:  {self.config.loss_type} (gamma={self.config.focal_gamma})")
        print(f"  Multi-Sample Dropout: {getattr(self.config, 'use_msd', True)}")
        print(f"  Fast Gradient Method (FGM): {getattr(self.config, 'use_fgm', True)} (eps={self.config.fgm_epsilon})")
        print("=" * 70)

        # -----------------------------------------------------------------
        # PHASE 1: Frozen Backbone
        # -----------------------------------------------------------------
        print(f"\n--- PHASE 1: Training Classification Heads ({self.config.freeze_phase_epochs} Epochs) ---")
        for p in self.model.mmbert.parameters():
            p.requires_grad = False

        opt_p1 = self.build_optimizer(lr=self.config.learning_rate)
        sched_p1 = self.build_scheduler(opt_p1, self.config.freeze_phase_epochs)
        
        # In phase 1, FGM is inactive on frozen backbone embeddings
        for epoch in range(1, self.config.freeze_phase_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(opt_p1, sched_p1, apply_fgm=False)
            val_loss, metrics, _, _ = self.eval_epoch()
            elapsed = time.time() - t0

            macro_f1 = metrics.get('hs_macro_f1', 0.0)
            f1_no = metrics.get('hs_f1_no', 0.0)
            f1_imp = metrics.get('hs_f1_implicit', 0.0)
            f1_exp = metrics.get('hs_f1_explicit', 0.0)

            print(f"Epoch {epoch:02d}/{self.config.freeze_phase_epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Macro F1: {macro_f1:.4f} (No: {f1_no:.3f}, Imp: {f1_imp:.3f}, Exp: {f1_exp:.3f}) | {elapsed:.1f}s")

            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
                print(f"  --> Saved Best Checkpoint (Macro-F1: {macro_f1:.4f}): {self.best_checkpoint_path}")

        # -----------------------------------------------------------------
        # PHASE 2: Unfreeze Backbone Layers + FGM Adversarial Training
        # -----------------------------------------------------------------
        if self.config.two_phase and self.config.unfreeze_phase_epochs > 0:
            print(f"\n--- PHASE 2: End-to-End Fine-Tuning + FGM Adversarial Training ({self.config.unfreeze_phase_epochs} Epochs) ---")
            
            # Load best checkpoint from Phase 1 before unfreezing
            if os.path.exists(self.best_checkpoint_path):
                self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device))
                print(f"  Loaded Phase 1 Best Weights (Macro-F1: {self.best_macro_f1:.4f})")

            # Unfreeze top N layers
            for p in self.model.mmbert.parameters():
                p.requires_grad = True

            opt_p2 = self.build_optimizer(
                lr=self.config.head_unfreeze_lr,
                backbone_lr=self.config.unfreeze_lr
            )
            sched_p2 = self.build_scheduler(opt_p2, self.config.unfreeze_phase_epochs)

            use_fgm_p2 = getattr(self.config, 'use_fgm', True)

            for epoch in range(1, self.config.unfreeze_phase_epochs + 1):
                t0 = time.time()
                train_loss = self.train_epoch(opt_p2, sched_p2, apply_fgm=use_fgm_p2)
                val_loss, metrics, _, _ = self.eval_epoch()
                elapsed = time.time() - t0

                macro_f1 = metrics.get('hs_macro_f1', 0.0)
                f1_no = metrics.get('hs_f1_no', 0.0)
                f1_imp = metrics.get('hs_f1_implicit', 0.0)
                f1_exp = metrics.get('hs_f1_explicit', 0.0)

                print(f"Epoch {epoch:02d}/{self.config.unfreeze_phase_epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Macro F1: {macro_f1:.4f} (No: {f1_no:.3f}, Imp: {f1_imp:.3f}, Exp: {f1_exp:.3f}) | {elapsed:.1f}s")

                if macro_f1 > self.best_macro_f1:
                    self.best_macro_f1 = macro_f1
                    torch.save(self.model.state_dict(), self.best_checkpoint_path)
                    print(f"  --> Saved Best Checkpoint (Macro-F1: {macro_f1:.4f}): {self.best_checkpoint_path}")

        # Final Evaluation on Best Model
        if os.path.exists(self.best_checkpoint_path):
            self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device))
            
        final_loss, final_metrics, final_preds, final_probs = self.eval_epoch()
        print("\n" + "=" * 20 + " FINAL TASK B METRICS " + "=" * 20)
        print(f"  Overall Accuracy:  {final_metrics.get('hs_acc', 0.0):.4f}")
        print(f"  Macro-F1:          {final_metrics.get('hs_macro_f1', 0.0):.4f}")
        print(f"  F1 (No Hate):      {final_metrics.get('hs_f1_no', 0.0):.4f}")
        print(f"  F1 (Implicit):     {final_metrics.get('hs_f1_implicit', 0.0):.4f}")
        print(f"  F1 (Explicit):     {final_metrics.get('hs_f1_explicit', 0.0):.4f}")
        print("=" * 62)

        if self.config.save_predictions:
            pred_df = pd.DataFrame({
                'pred_class': final_preds,
                'prob_no': final_probs[:, 0],
                'prob_implicit': final_probs[:, 1],
                'prob_explicit': final_probs[:, 2]
            })
            if self.df_val is not None:
                # Merge with metadata if available
                for col in ['lang', 'yt_comment', 'yt_title']:
                    if col in self.df_val.columns:
                        pred_df[col] = self.df_val[col].values
            pred_csv = os.path.join(self.config.output_dir, "task_b_val_predictions.csv")
            pred_df.to_csv(pred_csv, index=False)
            print(f"\nPredictions saved to: {pred_csv}")

        return {
            'best_macro_f1': self.best_macro_f1,
            'final_metrics': final_metrics,
            'checkpoint_path': self.best_checkpoint_path
        }

    def train(self) -> Dict[str, Any]:
        """Standard entrypoint alias matching StereoQueerTrainer interface."""
        return self.train_pipeline()
