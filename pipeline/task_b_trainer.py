import os
import time
import json
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .config import PipelineConfig, IDX2HATE
from .models.mmbert import unfreeze_last_n
from .losses import FocalLoss

class TaskBTrainer:
    """
    Dedicated Trainer for Task B: Class-Aware Multi-Head Cross-Attention (MHCA) Architecture.
    Trains hate speech classification (3 classes: no, yes_implicit, yes_explicit) with:
      - Explicit Role Embeddings (<T>, <D>, <C>)
      - Learned Class Queries [q_NonHate, q_Implicit, q_Explicit]
      - Optional Query Interaction (MHSA) Ablation
      - 2-Phase Fine-Tuning (Frozen backbone -> unfreeze last N layers)
    """
    def __init__(
        self,
        model: nn.Module,
        config: PipelineConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        df_val: pd.DataFrame,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.df_val = df_val

        # Device assignment
        if config.device:
            self.device = torch.device(config.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        
        # Mixed Precision (AMP) support for GPU speedup (2-3x faster forward/backward)
        self.use_amp = (self.device.type == "cuda")
        device_type = self.device.type if self.device.type in ["cuda", "cpu"] else "cuda"
        self.device_type = device_type
        
        # Use modern torch.amp.GradScaler if available, fallback to torch.cuda.amp.GradScaler
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(device_type, enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            
        if self.use_amp:
            print(f"[TaskBTrainer] PyTorch Automatic Mixed Precision (AMP - FP16 on {self.device}) Enabled.")
        
        # Configure Loss Function: Focal Loss or CrossEntropyLoss with balanced weights & label smoothing
        class_weights = getattr(config, 'class_weights', None)
        label_smoothing = getattr(config, 'label_smoothing', 0.05)
        loss_type = getattr(config, 'loss_type', 'focal')
        focal_gamma = getattr(config, 'focal_gamma', 2.0)
        
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
        backbone_prms = [p for n, p in self.model.named_parameters()
                         if p.requires_grad and n.startswith('mmbert.')]
        head_prms = [p for n, p in self.model.named_parameters()
                     if p.requires_grad and not n.startswith('mmbert.')]

        if backbone_prms and backbone_lr is not None:
            return torch.optim.AdamW([
                {'params': backbone_prms, 'lr': backbone_lr, 'weight_decay': self.config.weight_decay},
                {'params': head_prms, 'lr': lr, 'weight_decay': self.config.weight_decay},
            ])
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=self.config.weight_decay
        )

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
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

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def eval_epoch(self) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

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

            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(hs_labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs, axis=0)

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        metrics = {
            'hs_acc': acc,
            'hs_macro_f1': macro_f1,
            'hs_f1_no': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            'hs_f1_implicit': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            'hs_f1_explicit': float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
        }
        return avg_loss, metrics, y_pred, y_prob

    def run_training_loop(self, tag: str, num_epochs: int, lr: float,
                          backbone_lr: Optional[float] = None) -> str:
        optimizer = self.build_optimizer(lr=lr, backbone_lr=backbone_lr)
        best_metric = -1.0
        best_val_loss = float('inf')
        counter = 0
        best_checkpoint_path = os.path.join(self.config.output_dir, f"{tag}.pt")

        print(f"\n--- [Task B Class-Aware] Starting {tag} (Max Epochs: {num_epochs}, Patience: {self.config.patience}, Head LR: {lr}) ---")

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(optimizer)
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
