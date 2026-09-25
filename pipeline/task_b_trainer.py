import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from typing import Optional, Dict, Any, Tuple, List, Union

from .config import PipelineConfig, HATE_CLASSES, HATE2IDX, IDX2HATE
from .losses import FocalLoss
from .models.mmbert import unfreeze_last_n
from .models.task_b_class_aware import TaskBClassAwareAttentionModel


class TaskBTrainer:
    """
    Dedicated Trainer for Task B (Class-Aware Multi-Query Cross-Attention Model)
    Features:
      - Two-Phase Transfer Learning (Phase 1: Frozen Backbone Warmup -> Phase 2: Joint Backbone Fine-Tuning)
      - Integrated Linear Warmup + Cosine Annealing Learning Rate Scheduler
      - Differential Layer Learning Rates (Separate Backbone LR and Head LR)
      - Automatic Mixed Precision (AMP) with FP16/BF16 on CUDA / CPU Fallback
      - Class-Balanced Multi-Class Focal Loss with Label Smoothing
      - Dynamic Early Stopping and Macro-F1 Checkpoint Serialization
    """
    def __init__(
        self,
        config: PipelineConfig,
        model: TaskBClassAwareAttentionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        df_val: pd.DataFrame,
        class_weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.df_val = df_val
        self.class_weights = class_weights

        # Hardware Setup
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.use_amp = use_amp and (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        # Loss Function Setup
        label_smoothing = getattr(config, 'label_smoothing', 0.05)
        loss_type = getattr(config, 'loss_type', 'focal')
        focal_gamma = getattr(config, 'focal_gamma', 1.0)
        
        weights_tensor = None
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        if loss_type == "focal":
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                alpha=weights_tensor,
                label_smoothing=label_smoothing,
                reduction="mean"
            )
            print(f"[TaskBTrainer] Loss: Multi-Class Focal Loss (gamma={focal_gamma}, alpha={class_weights}, label_smoothing={label_smoothing})")
        else:
            if weights_tensor is not None:
                self.criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
                print(f"[TaskBTrainer] Loss: Weighted CrossEntropyLoss (weights={class_weights}, label_smoothing={label_smoothing})")
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                print(f"[TaskBTrainer] Loss: Standard CrossEntropyLoss (label_smoothing={label_smoothing})")
            
        self.history = []
        os.makedirs(self.config.output_dir, exist_ok=True)

    def build_optimizer(self, lr: float, backbone_lr: Optional[float] = None) -> torch.optim.Optimizer:
        """
        Builds AdamW optimizer with distinct learning rate parameter groups.
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
            # Fallback to PyTorch native CosineAnnealingLR if transformers helper is not available
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-7
            )

    def train_epoch(self, optimizer: torch.optim.Optimizer, scheduler: Optional[Any] = None) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            input_ids, attention_mask, role_ids, _, hs_labels, _ = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            role_ids = role_ids.to(self.device, non_blocking=True)
            hs_labels = hs_labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                logits, _, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    role_ids=role_ids
                )
                loss = self.criterion(logits, hs_labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
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
                    role_ids=role_ids
                )
                loss = self.criterion(logits, hs_labels)

            total_loss += loss.item()
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(hs_labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        avg_loss = total_loss / max(len(self.val_loader), 1)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else np.zeros((0, 3))

        macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics = {
            'hs_macro_f1': macro_f1,
            'hs_acc': acc,
            'hs_f1_no': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            'hs_f1_implicit': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            'hs_f1_explicit': float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
        }
        return avg_loss, metrics, y_pred, y_prob

    def run_training_loop(self, tag: str, num_epochs: int, lr: float,
                          backbone_lr: Optional[float] = None) -> str:
        optimizer = self.build_optimizer(lr=lr, backbone_lr=backbone_lr)
        scheduler = self.build_scheduler(optimizer, num_epochs=num_epochs)
        
        best_metric = -1.0
        best_val_loss = float('inf')
        counter = 0
        best_checkpoint_path = os.path.join(self.config.output_dir, f"{tag}.pt")

        print(f"\n--- [Task B Class-Aware] Starting {tag} (Max Epochs: {num_epochs}, Patience: {self.config.patience}, Head LR: {lr:.2e}, Backbone LR: {backbone_lr}) ---")

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(optimizer, scheduler=scheduler)
            val_loss, metrics, _, _ = self.eval_epoch()
            elapsed = time.time() - t0

            epoch_record = {
                'epoch': epoch + 1,
                'tag': tag,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'time_s': elapsed,
                **metrics
            }
            self.history.append(epoch_record)

            print(
                f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Macro F1: {metrics['hs_macro_f1']:.4f} (No: {metrics['hs_f1_no']:.3f}, Imp: {metrics['hs_f1_implicit']:.3f}, Exp: {metrics['hs_f1_explicit']:.3f}) | {elapsed:.1f}s"
            )

            # Checkpoint based on highest Macro-F1 (or lower val_loss if F1 tied)
            current_f1 = metrics['hs_macro_f1']
            is_better = (current_f1 > best_metric + 1e-4) or (abs(current_f1 - best_metric) <= 1e-4 and val_loss < best_val_loss)
            
            if is_better:
                best_metric = current_f1
                best_val_loss = val_loss
                counter = 0
                torch.save(self.model.state_dict(), best_checkpoint_path)
                print(f"  --> Saved Best Checkpoint (Macro-F1: {current_f1:.4f}): {best_checkpoint_path}")
            else:
                counter += 1
                if counter >= self.config.patience:
                    print(f"  --> Early stopping triggered at epoch {epoch+1} (patience reached).")
                    break

        return best_checkpoint_path

    def train(self) -> Dict[str, Any]:
        print(f"[Task B Class-Aware Attention Model] Device: {self.device}")
        print(f"Query Interaction Layer (MHSA) Enabled: {self.model.use_query_interaction}")

        if self.config.two_phase:
            # PHASE 1: Frozen Backbone
            print("\n>>> PHASE 1/2: Frozen mmBERT Backbone -> Learning Role Embeddings, Class Queries & MHCA")
            for p in self.model.mmbert.parameters():
                p.requires_grad = False

            phase1_tag = "task_b_phase1_frozen"
            best_p1 = self.run_training_loop(
                tag=phase1_tag,
                num_epochs=self.config.freeze_phase_epochs,
                lr=self.config.learning_rate
            )

            # PHASE 2: Load Phase 1, unfreeze last N layers
            print(f"\n>>> Loading Best Phase 1 Weights: {best_p1}")
            self.model.load_state_dict(torch.load(best_p1, map_location=self.device))

            n_unfrozen = unfreeze_last_n(self.model.mmbert, self.config.unfreeze_layers)
            print(f">>> PHASE 2/2: Fine-Tuning Last {self.config.unfreeze_layers} Encoder Blocks (Found {n_unfrozen})")

            phase2_tag = "task_b_best_model"
            best_final = self.run_training_loop(
                tag=phase2_tag,
                num_epochs=self.config.unfreeze_phase_epochs,
                lr=self.config.head_unfreeze_lr,
                backbone_lr=self.config.unfreeze_lr
            )
        else:
            best_final = self.run_training_loop(
                tag="task_b_best_model",
                num_epochs=self.config.epochs,
                lr=self.config.learning_rate
            )

        # Load best weights
        self.model.load_state_dict(torch.load(best_final, map_location=self.device))
        _, final_metrics, y_pred, y_prob = self.eval_epoch()

        print("\n==================== FINAL TASK B METRICS ====================")
        print(f"  Overall Accuracy:  {final_metrics['hs_acc']:.4f}")
        print(f"  Macro-F1:          {final_metrics['hs_macro_f1']:.4f}")
        print(f"  F1 (No Hate):      {final_metrics['hs_f1_no']:.4f}")
        print(f"  F1 (Implicit):     {final_metrics['hs_f1_implicit']:.4f}")
        print(f"  F1 (Explicit):     {final_metrics['hs_f1_explicit']:.4f}")
        print("==============================================================\n")

        # Save predictions CSV
        if self.config.save_predictions and 'StereoQueerEval_id' in self.df_val.columns:
            res_df = self.df_val[['StereoQueerEval_id', 'lang', 'yt_title', 'yt_comment', 'hate_speech']].copy()
            res_df['pred_hate_speech'] = [IDX2HATE.get(int(p), 'no') for p in y_pred]
            res_df['prob_no'] = y_prob[:, 0]
            res_df['prob_implicit'] = y_prob[:, 1]
            res_df['prob_explicit'] = y_prob[:, 2]
            pred_file = os.path.join(self.config.output_dir, "task_b_val_predictions.csv")
            res_df.to_csv(pred_file, index=False)
            print(f"Predictions saved to: {pred_file}")

        # Save history
        hist_path = os.path.join(self.config.output_dir, "task_b_training_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            'checkpoint_path': best_final,
            'final_metrics': final_metrics,
            'history': self.history
        }
