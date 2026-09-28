"""
Task B Specialized Trainer with 4-Expert Mixture of Latent Query Banks (MoE),
Fast Gradient Method (FGM) Adversarial Regularization, Multi-Sample Dropout,
Early Stopping with Patience, and Dynamic Router Gate Utilization Tracking.
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
from .losses import build_loss_fn, TaskBLoss
from .metrics import compute_classification_metrics
from .models.task_b_class_aware import TaskBClassAwareAttentionModel
from .models.mmbert import unfreeze_last_n


class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training on Transformer Word Embeddings.
    Perturbs token embeddings in the direction of the loss gradient by scale epsilon:
        delta = epsilon * (grad / ||grad||_2)
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
    Trainer for Task B 4-Expert MoE Architecture with:
      - Early Stopping with Configurable Patience
      - Dynamic Router Gate Utilization Tracking & Load Balancing
      - 2-Phase Backbone Fine-Tuning (Frozen -> Layer-Unfrozen: top N layers)
      - Cosine Annealing with Warmup
      - Multi-Sample Dropout Loss Averaging
      - Fast Gradient Method (FGM) Adversarial Regularization
      - Mixed Precision (AMP)
      - Per-Language Validation Metric Tracking
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

        # Base Loss Function
        base_loss_fn = build_loss_fn(
            loss_type=config.loss_type,
            class_weights=config.class_weights,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            device=self.device
        )

        # Task B Loss with MoE Load Balancing
        self.criterion = TaskBLoss(
            base_criterion=base_loss_fn,
            num_experts=getattr(config, 'num_experts', 4),
            loss_balance_weight=getattr(config, 'loss_balance_weight', 0.01)
        )

        # Adversarial Trainer
        self.fgm = FGM(
            model=self.model,
            epsilon=config.fgm_epsilon,
            emb_name=config.fgm_emb_name
        ) if getattr(config, 'use_fgm', False) else None

        self.best_macro_f1 = -1.0
        self.best_checkpoint_path = ""
        self.history: List[Dict[str, Any]] = []

    def enable_gradient_checkpointing(self):
        """Reduces activation memory during unfrozen backprop."""
        try:
            if hasattr(self.model.mmbert, "gradient_checkpointing_enable"):
                self.model.mmbert.gradient_checkpointing_enable()
                print("  [Memory Optimization] Gradient checkpointing enabled for mmBERT.")
        except Exception as e:
            print(f"  [Memory Optimization] Gradient checkpointing skipped: {e}")

    def build_optimizer(self, lr: float, backbone_lr: Optional[float] = None) -> torch.optim.Optimizer:
        """
        Builds AdamW optimizer with differential learning rates for backbone vs heads.
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
        Cosine Annealing schedule with 10% linear warmup.
        """
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
        apply_fgm: bool = True
    ) -> Tuple[float, np.ndarray]:
        self.model.train()
        total_loss = 0.0
        all_gates: List[np.ndarray] = []

        for batch in self.train_loader:
            input_ids, attention_mask, role_ids, _, hs_labels, _ = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            role_ids = role_ids.to(self.device, non_blocking=True)
            hs_labels = hs_labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            
            # Forward 1: Clean pass
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    role_ids=role_ids,
                    return_gates=True,
                    return_all_msd_logits=True
                )
                logits, _, gates, _ = out
                loss = self.criterion(logits, hs_labels, gates=gates)

            if gates is not None:
                all_gates.append(gates.detach().cpu().numpy())

            is_unscaled = False

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Forward 2: Fast Gradient Method
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
                        return_gates=True,
                        return_all_msd_logits=False
                    )
                    adv_logits, _, adv_gates, _ = adv_out
                    adv_loss = self.criterion(adv_logits, hs_labels, gates=adv_gates)
                
                if self.use_amp:
                    self.scaler.scale(adv_loss).backward()
                else:
                    adv_loss.backward()
                    
                self.fgm.restore()

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

        avg_loss = total_loss / max(len(self.train_loader), 1)
        mean_gates = np.mean(np.concatenate(all_gates, axis=0), axis=0) if all_gates else np.array([])
        return avg_loss, mean_gates

    @torch.no_grad()
    def eval_epoch(self) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        all_gates = []

        with torch.inference_mode():
            for batch in self.val_loader:
                input_ids, attention_mask, role_ids, _, hs_labels, _ = batch
                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)
                role_ids = role_ids.to(self.device, non_blocking=True)
                hs_labels = hs_labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        role_ids=role_ids,
                        return_gates=True,
                        return_all_msd_logits=False
                    )
                    logits, _, gates, _ = out
                    loss = self.criterion(logits, hs_labels, gates=gates)

                total_loss += loss.item()
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(hs_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                if gates is not None:
                    all_gates.extend(gates.cpu().numpy())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(len(self.val_loader), 1)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        y_gates = np.array(all_gates) if all_gates else np.zeros((len(y_true), 4))

        metrics = compute_classification_metrics(y_true, y_pred, y_prob, task="hs")
        return avg_loss, metrics, y_pred, y_prob, y_gates

    def fit(self) -> Dict[str, Any]:
        """
        Executes complete training lifecycle with Early Stopping (Patience).
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        patience_limit = getattr(self.config, 'patience', 5)
        print(f"\n=======================================================")
        print(f" Task B 4-Expert MoE Training Pipeline")
        print(f" Device: {self.device} | AMP: {self.use_amp} | Experts: {self.config.num_experts} | Patience: {patience_limit}")
        print(f"=======================================================")

        last_metrics: Dict[str, float] = {}

        # PHASE 1: Train Heads with Frozen Backbone
        if self.config.two_phase:
            print(f"\n>>> [Phase 1/2] Training Heads (mmBERT Backbone Frozen) for {self.config.freeze_phase_epochs} epochs...")
            for param in self.model.mmbert.parameters():
                param.requires_grad = False
                
            optimizer = self.build_optimizer(lr=self.config.learning_rate)
            scheduler = self.build_scheduler(optimizer, self.config.freeze_phase_epochs)
            p1_patience_counter = 0

            for epoch in range(1, self.config.freeze_phase_epochs + 1):
                t0 = time.time()
                train_loss, train_gates = self.train_epoch(optimizer, scheduler, apply_fgm=False)
                val_loss, metrics, y_pred, y_prob, y_gates = self.eval_epoch()
                elapsed = time.time() - t0
                last_metrics = metrics

                macro_f1 = metrics.get('hs_macro_f1', 0.0)
                gate_str = ", ".join([f"E{i}:{g:.2f}" for i, g in enumerate(train_gates)]) if len(train_gates) > 0 else "N/A"
                print(f"Epoch {epoch:02d}/{self.config.freeze_phase_epochs:02d} [{elapsed:.1f}s] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Macro-F1: {macro_f1:.4f} | Gates: [{gate_str}]")

                self.history.append({
                    'epoch': epoch,
                    'phase': 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'hs_macro_f1': macro_f1,
                    'gates': train_gates
                })

                if macro_f1 > self.best_macro_f1:
                    self.best_macro_f1 = macro_f1
                    self.best_checkpoint_path = self.save_checkpoint("best_phase1.pt")
                    p1_patience_counter = 0
                else:
                    p1_patience_counter += 1
                    if p1_patience_counter >= patience_limit:
                        print(f"  [EarlyStopping] Phase 1 Early stopped after {patience_limit} non-improving epochs.")
                        break

            # PHASE 2: Unfreeze Top N Layers of mmBERT
            print(f"\n>>> [Phase 2/2] Unfreezing Top {self.config.unfreeze_layers} Backbone Layers for {self.config.unfreeze_phase_epochs} epochs...")
            unfreeze_last_n(self.model.mmbert, n=self.config.unfreeze_layers)
            if getattr(self.config, 'use_gradient_checkpointing', False):
                self.enable_gradient_checkpointing()

            optimizer = self.build_optimizer(
                lr=self.config.head_unfreeze_lr,
                backbone_lr=self.config.unfreeze_lr
            )
            scheduler = self.build_scheduler(optimizer, self.config.unfreeze_phase_epochs)
            total_p2_epochs = self.config.unfreeze_phase_epochs
            p2_patience_counter = 0

            for epoch in range(1, total_p2_epochs + 1):
                t0 = time.time()
                train_loss, train_gates = self.train_epoch(optimizer, scheduler, apply_fgm=getattr(self.config, 'use_fgm', False))
                val_loss, metrics, y_pred, y_prob, y_gates = self.eval_epoch()
                elapsed = time.time() - t0
                last_metrics = metrics

                macro_f1 = metrics.get('hs_macro_f1', 0.0)
                gate_str = ", ".join([f"E{i}:{g:.2f}" for i, g in enumerate(train_gates)]) if len(train_gates) > 0 else "N/A"
                print(f"Epoch {epoch:02d}/{total_p2_epochs:02d} [{elapsed:.1f}s] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Macro-F1: {macro_f1:.4f} | Gates: [{gate_str}]")

                self.history.append({
                    'epoch': self.config.freeze_phase_epochs + epoch,
                    'phase': 2,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'hs_macro_f1': macro_f1,
                    'gates': train_gates
                })

                if macro_f1 > self.best_macro_f1:
                    self.best_macro_f1 = macro_f1
                    self.best_checkpoint_path = self.save_checkpoint("best_model.pt")
                    if self.config.save_predictions and self.df_val is not None:
                        self.save_val_predictions(y_pred, y_prob, y_gates)
                    p2_patience_counter = 0
                else:
                    p2_patience_counter += 1
                    print(f"  [EarlyStopping] No improvement in Macro-F1 ({p2_patience_counter}/{patience_limit}).")
                    if p2_patience_counter >= patience_limit:
                        print(f"  [EarlyStopping] Triggered at Epoch {epoch:02d}! Best Val Macro-F1: {self.best_macro_f1:.4f}")
                        break
        else:
            # Single-phase training
            optimizer = self.build_optimizer(lr=self.config.learning_rate)
            scheduler = self.build_scheduler(optimizer, self.config.epochs)
            patience_counter = 0
            for epoch in range(1, self.config.epochs + 1):
                t0 = time.time()
                train_loss, train_gates = self.train_epoch(optimizer, scheduler, apply_fgm=getattr(self.config, 'use_fgm', False))
                val_loss, metrics, y_pred, y_prob, y_gates = self.eval_epoch()
                elapsed = time.time() - t0
                last_metrics = metrics

                macro_f1 = metrics.get('hs_macro_f1', 0.0)
                gate_str = ", ".join([f"E{i}:{g:.2f}" for i, g in enumerate(train_gates)]) if len(train_gates) > 0 else "N/A"
                print(f"Epoch {epoch:02d}/{self.config.epochs:02d} [{elapsed:.1f}s] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Macro-F1: {macro_f1:.4f} | Gates: [{gate_str}]")

                self.history.append({
                    'epoch': epoch,
                    'phase': 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'hs_macro_f1': macro_f1,
                    'gates': train_gates
                })

                if macro_f1 > self.best_macro_f1:
                    self.best_macro_f1 = macro_f1
                    self.best_checkpoint_path = self.save_checkpoint("best_model.pt")
                    if self.config.save_predictions and self.df_val is not None:
                        self.save_val_predictions(y_pred, y_prob, y_gates)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  [EarlyStopping] No improvement in Macro-F1 ({patience_counter}/{patience_limit}).")
                    if patience_counter >= patience_limit:
                        print(f"  [EarlyStopping] Triggered at Epoch {epoch:02d}! Best Val Macro-F1: {self.best_macro_f1:.4f}")
                        break

        return {
            "best_macro_f1": self.best_macro_f1,
            "best_checkpoint": self.best_checkpoint_path,
            "checkpoint_path": self.best_checkpoint_path,
            "history": self.history,
            "final_metrics": last_metrics
        }

    train = fit

    def save_checkpoint(self, filename: str) -> str:
        path = os.path.join(self.config.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_macro_f1': self.best_macro_f1,
            'config': self.config.to_dict()
        }, path)
        return path

    def save_val_predictions(self, y_pred: np.ndarray, y_prob: np.ndarray, y_gates: np.ndarray):
        """Exports predictions and router gates to CSV."""
        if self.df_val is None:
            return
        df_out = self.df_val.copy().reset_index(drop=True)
        df_out['pred_class'] = y_pred
        df_out['pred_hate_speech'] = [IDX2HATE.get(p, 'no') for p in y_pred]
        df_out['prob_no'] = y_prob[:, 0]
        df_out['prob_implicit'] = y_prob[:, 1]
        df_out['prob_explicit'] = y_prob[:, 2]
        
        for i in range(y_gates.shape[1]):
            df_out[f'gate_exp{i}'] = y_gates[:, i]

        csv_path = os.path.join(self.config.output_dir, "task_b_val_predictions.csv")
        df_out.to_csv(csv_path, index=False)
