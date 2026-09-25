import os
import time
import json
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import PipelineConfig, IDX2HATE
from .losses import MultiTaskLoss
from .metrics import evaluate_stereoqueer, print_metrics
from .models.mmbert import unfreeze_last_n


class StereoQueerTrainer:
    """
    Production-grade Multi-task Trainer for StereoQueerEval.
    Supports 2-phase fine-tuning (frozen backbone -> discriminative unfreezing),
    early stopping with patience, and multi-lingual validation tracking.
    """
    def __init__(self, model: nn.Module, config: PipelineConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 df_val: pd.DataFrame, is_mmbert_tf: bool = True):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.df_val = df_val
        self.is_mmbert_tf = is_mmbert_tf

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
        self.loss_fn = MultiTaskLoss(config)
        self.history = []

        os.makedirs(self.config.output_dir, exist_ok=True)

    def build_optimizer(self, lr: float, backbone_lr: Optional[float] = None) -> torch.optim.Optimizer:
        """
        Builds optimizer with discriminative learning rates:
        Applies a smaller learning rate to unfrozen pretrained backbone layers
        to prevent catastrophic forgetting while adapting new classification heads.
        """
        if self.is_mmbert_tf:
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
            optimizer.zero_grad()
            if self.is_mmbert_tf:
                ids, mask, st, hs, tg = batch
                ids, mask = ids.to(self.device), mask.to(self.device)
                st_logits, hs_logits, tg_logits = self.model(ids, mask)
            else:
                inputs, st, hs, tg = batch
                inputs = inputs.to(self.device)
                st_logits, hs_logits, tg_logits = self.model(inputs)

            st, hs, tg = st.to(self.device), hs.to(self.device), tg.to(self.device)
            loss, _ = self.loss_fn(st_logits, hs_logits, tg_logits, st, hs, tg)
            loss.backward()

            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad_norm)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def eval_epoch(self) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """
        Runs a single unified validation pass that calculates:
        - Average validation loss
        - Official StereoQueerEval evaluation metrics (ST F1, HS F1, TG Exact Match, etc.)
        """
        self.model.eval()
        total_val_loss = 0.0
        st_list, hs_list, tg_list = [], [], []

        for batch in self.val_loader:
            if self.is_mmbert_tf:
                ids, mask, st, hs, tg = batch
                ids, mask = ids.to(self.device), mask.to(self.device)
                st_logits, hs_logits, tg_logits = self.model(ids, mask)
            else:
                inputs, st, hs, tg = batch
                inputs = inputs.to(self.device)
                st_logits, hs_logits, tg_logits = self.model(inputs)

            st, hs, tg = st.to(self.device), hs.to(self.device), tg.to(self.device)
            loss, _ = self.loss_fn(st_logits, hs_logits, tg_logits, st, hs, tg)
            total_val_loss += loss.item()

            st_list.append(torch.sigmoid(st_logits.squeeze(-1)).cpu().numpy())
            hs_list.append(torch.argmax(hs_logits, dim=1).cpu().numpy())
            tg_list.append(torch.sigmoid(tg_logits).cpu().numpy())

        avg_val_loss = total_val_loss / len(self.val_loader)
        st_probs = np.concatenate(st_list).ravel()
        hs_preds = np.concatenate(hs_list)
        tg_probs = np.concatenate(tg_list, axis=0)

        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score
        from .data import decode_target

        st_preds = (st_probs >= 0.5).astype(int)
        tg_pred_str = np.array([decode_target(v) for v in tg_probs])

        gold_st = self.df_val['st_y'].values.astype(int)
        gold_hs = self.df_val['hs_y'].values.astype(int)
        gold_tg = self.df_val['target'].fillna('none').values

        metrics = {
            'st_acc': float(accuracy_score(gold_st, st_preds)),
            'st_f1': float(f1_score(gold_st, st_preds, average='macro', zero_division=0)),
            'hs_acc': float(accuracy_score(gold_hs, hs_preds)),
            'hs_f1': float(f1_score(gold_hs, hs_preds, average='macro', zero_division=0)),
            'tg_exact_match': float((tg_pred_str == gold_tg).mean()),
        }
        metrics['macro_avg_f1'] = float((metrics['st_f1'] + metrics['hs_f1'] + metrics['tg_exact_match']) / 3.0)

        preds = {
            'st_probs': st_probs,
            'st_preds': st_preds,
            'hs_preds': hs_preds,
            'tg_pred_str': tg_pred_str
        }

        return avg_val_loss, metrics, preds

    def run_training_loop(self, tag: str, num_epochs: int, lr: float,
                          backbone_lr: Optional[float] = None) -> str:
        optimizer = self.build_optimizer(lr=lr, backbone_lr=backbone_lr)
        best_val_loss = float('inf')
        counter = 0
        best_checkpoint_path = os.path.join(self.config.output_dir, f"{tag}.pt")

        print(f"\n--- Starting {tag} (Max Epochs: {num_epochs}, Patience: {self.config.patience}, Head LR: {lr}) ---")

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(optimizer)
            val_loss, metrics, _ = self.eval_epoch()
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

            print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"ST F1: {metrics['st_f1']:.3f} | HS F1: {metrics['hs_f1']:.3f} | TG Exact: {metrics['tg_exact_match']:.3f} | {elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(self.model.state_dict(), best_checkpoint_path)
                print(f"  --> Checkpoint saved: {best_checkpoint_path}")
            else:
                counter += 1
                if counter >= self.config.patience:
                    print(f"  --> Early stopping triggered at epoch {epoch+1} (patience reached).")
                    break

        return best_checkpoint_path

    def train(self) -> Dict[str, Any]:
        """Executes full training schedule (single-phase or two-phase)."""
        print(f"Device: {self.device}")

        if self.is_mmbert_tf and self.config.two_phase:
            # PHASE 1: Completely frozen backbone
            print("\n>>> PHASE 1/2: Frozen Backbone, Training Task Heads & Transformer Layers")
            for p in self.model.mmbert.parameters():
                p.requires_grad = False

            phase1_tag = "phase1_frozen"
            best_p1 = self.run_training_loop(
                tag=phase1_tag,
                num_epochs=self.config.freeze_phase_epochs,
                lr=self.config.learning_rate
            )

            # PHASE 2: Load best checkpoint, unfreeze last N layers
            print(f"\n>>> Loading Best Phase 1 Checkpoint: {best_p1}")
            self.model.load_state_dict(torch.load(best_p1, map_location=self.device))

            n_unfrozen = unfreeze_last_n(self.model.mmbert, self.config.unfreeze_layers)
            print(f">>> PHASE 2/2: Fine-Tuning Last {self.config.unfreeze_layers} Encoder Blocks (Found {n_unfrozen})")

            phase2_tag = "best_model"
            best_final = self.run_training_loop(
                tag=phase2_tag,
                num_epochs=self.config.unfreeze_phase_epochs,
                lr=self.config.head_unfreeze_lr,
                backbone_lr=self.config.unfreeze_lr
            )
        else:
            final_tag = "best_model"
            best_final = self.run_training_loop(
                tag=final_tag,
                num_epochs=self.config.epochs,
                lr=self.config.learning_rate
            )

        # Load best model for final evaluation
        self.model.load_state_dict(torch.load(best_final, map_location=self.device))
        final_metrics, preds = evaluate_stereoqueer(
            self.model, self.df_val, self.val_loader, self.device, self.is_mmbert_tf
        )
        print_metrics(final_metrics, "FINAL VALIDATION METRICS")

        # Save validation predictions CSV
        if self.config.save_predictions and 'StereoQueerEval_id' in self.df_val.columns:
            results_df = self.df_val[[
                'StereoQueerEval_id', 'lang', 'yt_comment', 'stereotype', 'hate_speech', 'target'
            ]].copy()
            results_df['pred_stereotype'] = ['yes' if p == 1 else 'no' for p in preds['st_preds']]
            results_df['pred_hate_speech'] = [IDX2HATE.get(int(h), 'no') for h in preds['hs_preds']]
            results_df['pred_target'] = preds['tg_pred_str']
            pred_path = os.path.join(self.config.output_dir, "stereoqueer_val_predictions.csv")
            results_df.to_csv(pred_path, index=False)
            print(f"Validation predictions saved to: {pred_path}")

        # Save training history JSON
        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            'checkpoint_path': best_final,
            'final_metrics': final_metrics,
            'predictions': preds,
            'history': self.history
        }
