"""
Task B Cross-Context Training Loop.

Features:
- Two-Phase Training: Linear probing of cross-attention & role embeddings followed by fine-tuning.
- Multi-Sample Dropout loss aggregation.
- Cosine Annealing with Linear Warmup.
- Fast Gradient Method (FGM) adversarial training.
- Per-Class Gate Diagnostics (Review #16): Computes and logs E[g | Implicit], E[g | Explicit], and E[g | Non-Hate]
  to scientifically validate whether the model relies on Description context more heavily for implicit hate speech.
"""

import os
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .config import PipelineConfig, HATE2IDX, IDX2HATE
from .metrics import compute_classification_metrics
from .losses import FocalLoss, LabelSmoothingCrossEntropy, build_loss_fn
from .models.task_b_cross_context import TaskBCrossContextAttentionModel


class FGM:
    """Fast Gradient Method (FGM) for embedding space adversarial perturbation."""
    def __init__(self, model: nn.Module, emb_name: str = 'embeddings', epsilon: float = 0.50):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup: Dict[str, torch.Tensor] = {}

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
                param.data = self.backup[name]
        self.backup.clear()


def unfreeze_last_n(model: nn.Module, n: int):
    """
    Freezes or unfreezes backbone transformer layers.
    n = 0: Freeze all backbone parameters.
    n > 0: Freeze all except the last n transformer layers + pooler.
    """
    for param in model.parameters():
        param.requires_grad = False

    if n <= 0:
        return

    encoder_layers = None
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        encoder_layers = model.encoder.layer
    elif hasattr(model, 'layers'):
        encoder_layers = model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
        encoder_layers = model.transformer.layer

    if encoder_layers is not None:
        total_layers = len(encoder_layers)
        start_idx = max(0, total_layers - n)
        for layer in encoder_layers[start_idx:]:
            for p in layer.parameters():
                p.requires_grad = True

    if hasattr(model, 'pooler') and model.pooler is not None:
        for p in model.pooler.parameters():
            p.requires_grad = True


class TaskBCrossTrainer:
    """
    Dedicated Trainer for Task B Dual-Stream Cross-Context Attention Model.
    Accepts flexible argument order and optional parameters (class_weights, df_val, use_amp, device).
    """
    def __init__(
        self,
        model: TaskBCrossContextAttentionModel,
        config: PipelineConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        df_val: Optional[pd.DataFrame] = None,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        **kwargs
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.df_val = df_val

        # Device determination
        if device is not None:
            self.device = device
        elif getattr(config, 'device', None):
            self.device = torch.device(config.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        # Mixed Precision
        self.use_amp = use_amp and (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        # Class weights determination (prioritize parameter if passed, else config)
        weights_to_use = class_weights if class_weights is not None else getattr(config, 'class_weights', None)

        # Loss Function using build_loss_fn
        loss_type = getattr(config, 'loss_type', 'focal')
        self.criterion = build_loss_fn(
            loss_type=loss_type,
            class_weights=weights_to_use,
            gamma=getattr(config, 'focal_gamma', 2.0),
            label_smoothing=getattr(config, 'label_smoothing', 0.05),
            device=self.device
        )

        # FGM
        use_fgm = getattr(config, 'use_fgm', True)
        self.fgm = FGM(self.model, emb_name='embeddings', epsilon=getattr(config, 'fgm_epsilon', 0.50)) if use_fgm else None

        self.history: List[Dict[str, Any]] = []
        self.best_macro_f1: float = -1.0
        self.best_checkpoint_path: Optional[str] = None
        os.makedirs(self.config.output_dir, exist_ok=True)

    def build_scheduler(self, optimizer: torch.optim.Optimizer, num_epochs: int):
        total_steps = len(self.train_loader) * num_epochs
        warmup_ratio = getattr(self.config, 'warmup_ratio', 0.10)
        min_lr = getattr(self.config, 'min_lr', 1e-7)
        lr = getattr(self.config, 'learning_rate', 1e-4)
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr / lr, cosine_decay)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        apply_fgm: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        all_train_preds = []
        all_train_targets = []

        loader_iter = self.train_loader
        if tqdm is not None:
            loader_iter = tqdm(self.train_loader, desc="  Training", leave=False, dynamic_ncols=True)

        for batch in loader_iter:
            tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, labels = batch
            
            tc_ids = tc_ids.to(self.device, non_blocking=True)
            tc_mask = tc_mask.to(self.device, non_blocking=True)
            tc_roles = tc_roles.to(self.device, non_blocking=True)
            desc_ids = desc_ids.to(self.device, non_blocking=True)
            desc_mask = desc_mask.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                # Standard Forward pass with MSD
                if self.model.use_msd:
                    logits_list = self.model(
                        tc_input_ids=tc_ids,
                        tc_attention_mask=tc_mask,
                        tc_role_ids=tc_roles,
                        desc_input_ids=desc_ids,
                        desc_attention_mask=desc_mask,
                        return_all_msd_logits=True
                    )
                    loss = torch.mean(torch.stack([self.criterion(l, labels) for l in logits_list]))
                    logits_avg = torch.mean(torch.stack(logits_list, dim=0), dim=0)
                else:
                    logits = self.model(
                        tc_input_ids=tc_ids,
                        tc_attention_mask=tc_mask,
                        tc_role_ids=tc_roles,
                        desc_input_ids=desc_ids,
                        desc_attention_mask=desc_mask,
                        return_all_msd_logits=False
                    )
                    loss = self.criterion(logits, labels)
                    logits_avg = logits

            preds = torch.argmax(logits_avg, dim=-1)
            all_train_preds.append(preds.detach().cpu().numpy())
            all_train_targets.append(labels.detach().cpu().numpy())

            # Backward pass
            is_unscaled = False
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Adversarial Training (FGM)
            if apply_fgm and self.fgm is not None:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    is_unscaled = True
                
                self.fgm.attack()
                
                with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                    adv_logits = self.model(
                        tc_input_ids=tc_ids,
                        tc_attention_mask=tc_mask,
                        tc_role_ids=tc_roles,
                        desc_input_ids=desc_ids,
                        desc_attention_mask=desc_mask,
                        return_all_msd_logits=False
                    )
                    adv_loss = self.criterion(adv_logits, labels)

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

            if tqdm is not None and hasattr(loader_iter, 'set_postfix'):
                current_lr = optimizer.param_groups[0]['lr']
                loader_iter.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })

        avg_loss = total_loss / max(len(self.train_loader), 1)
        y_train_true = np.concatenate(all_train_targets, axis=0) if all_train_targets else np.array([])
        y_train_pred = np.concatenate(all_train_preds, axis=0) if all_train_preds else np.array([])
        train_metrics = compute_classification_metrics(y_train_true, y_train_pred, task="hs") if len(y_train_true) > 0 else {}

        return avg_loss, train_metrics

    @torch.no_grad()
    def eval_epoch(self) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Evaluates the model on the validation set and returns detailed per-class gate statistics:
        - gate_mean (overall E[g])
        - gate_no (E[g | Non-Hate])
        - gate_implicit (E[g | Implicit])
        - gate_explicit (E[g | Explicit])
        """
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        sample_gate_means = []

        with torch.inference_mode():
            val_iter = self.val_loader
            if tqdm is not None:
                val_iter = tqdm(self.val_loader, desc="  Validation", leave=False, dynamic_ncols=True)

            for batch in val_iter:
                tc_ids, tc_mask, tc_roles, desc_ids, desc_mask, labels = batch
                
                tc_ids = tc_ids.to(self.device, non_blocking=True)
                tc_mask = tc_mask.to(self.device, non_blocking=True)
                tc_roles = tc_roles.to(self.device, non_blocking=True)
                desc_ids = desc_ids.to(self.device, non_blocking=True)
                desc_mask = desc_mask.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                    logits, gate = self.model(
                        tc_input_ids=tc_ids,
                        tc_attention_mask=tc_mask,
                        tc_role_ids=tc_roles,
                        desc_input_ids=desc_ids,
                        desc_attention_mask=desc_mask,
                        return_all_msd_logits=False,
                        return_gate_values=True
                    )
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                if gate is not None:
                    # gate shape: [B, S_c, d_model] -> average over tokens & dims per sample: [B]
                    per_sample_gate = gate.mean(dim=(1, 2)).cpu().numpy()
                    sample_gate_means.append(per_sample_gate)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)
        
        # Calculate Per-Class Gate Diagnostics (Review #16)
        gate_stats = {
            'gate_mean': 0.0,
            'gate_no': 0.0,
            'gate_implicit': 0.0,
            'gate_explicit': 0.0,
        }
        if sample_gate_means:
            all_gates = np.concatenate(sample_gate_means, axis=0)
            gate_stats['gate_mean'] = float(np.mean(all_gates))

            idx_no = HATE2IDX.get('no', 0)
            idx_imp = HATE2IDX.get('yes_implicit', 1)
            idx_exp = HATE2IDX.get('yes_explicit', 2)

            mask_no = (y_true == idx_no)
            mask_imp = (y_true == idx_imp)
            mask_exp = (y_true == idx_exp)

            if mask_no.any():
                gate_stats['gate_no'] = float(np.mean(all_gates[mask_no]))
            if mask_imp.any():
                gate_stats['gate_implicit'] = float(np.mean(all_gates[mask_imp]))
            if mask_exp.any():
                gate_stats['gate_explicit'] = float(np.mean(all_gates[mask_exp]))

        val_metrics = compute_classification_metrics(y_true, y_pred, task="hs")
        return avg_loss, val_metrics, y_true, y_pred, gate_stats

    def train(self) -> Dict[str, Any]:
        """Executes full Two-Phase Differential Training Routine with Early Stopping."""
        print("================================================================================")
        print("🚀 STARTING TASK B DUAL-STREAM CROSS-CONTEXT TRAINING")
        print("================================================================================")
        print(f"Device: {self.device} | AMP: {self.use_amp} | FGM: {self.fgm is not None}")
        print(f"Batch Size: {self.config.batch_size} | Two-Phase: {self.config.two_phase}")

        patience = getattr(self.config, 'patience', 5)
        patience_counter = 0

        # PHASE 1: Freeze Backbone & Train Fusion Head
        if self.config.two_phase and self.config.freeze_phase_epochs > 0:
            print(f"\n>>> [Phase 1/2] Linear Probing (Backbone Frozen) for {self.config.freeze_phase_epochs} epochs...")
            unfreeze_last_n(self.model.mmbert, 0) # Freeze all layers
            
            # Collect trainable parameters (cross_context, role_emb, classifier)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-6
            )
            scheduler = self.build_scheduler(optimizer, self.config.freeze_phase_epochs)

            for epoch in range(1, self.config.freeze_phase_epochs + 1):
                t0 = time.time()
                train_loss, train_metrics = self.train_epoch(optimizer=optimizer, scheduler=scheduler, apply_fgm=False)
                val_loss, val_metrics, _, _, gate_stats = self.eval_epoch()
                elapsed = time.time() - t0

                macro_f1 = val_metrics.get('hs_macro_f1', 0.0)
                no_f1 = val_metrics.get('hs_f1_no', 0.0)
                imp_f1 = val_metrics.get('hs_f1_implicit', 0.0)
                exp_f1 = val_metrics.get('hs_f1_explicit', 0.0)
                val_acc = val_metrics.get('hs_acc', 0.0)
                train_f1 = train_metrics.get('hs_macro_f1', 0.0)

                self.history.append({
                    'phase': 1,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_macro_f1': train_f1,
                    'val_loss': val_loss,
                    'hs_macro_f1': macro_f1,
                    'hs_f1_no': no_f1,
                    'hs_f1_implicit': imp_f1,
                    'hs_f1_explicit': exp_f1,
                    'hs_acc': val_acc,
                    **gate_stats
                })

                print(f"  [P1 Epoch {epoch:02d}/{self.config.freeze_phase_epochs:02d}] "
                      f"Train Loss: {train_loss:.4f} (F1: {train_f1:.4f}) | Val Loss: {val_loss:.4f} | "
                      f"Val Macro-F1: {macro_f1:.4f} | No: {no_f1:.4f} | Imp: {imp_f1:.4f} | Exp: {exp_f1:.4f} | "
                      f"Gate (All/No/Imp/Exp): {gate_stats['gate_mean']:.3f}/{gate_stats['gate_no']:.3f}/{gate_stats['gate_implicit']:.3f}/{gate_stats['gate_explicit']:.3f} [{elapsed:.1f}s]")

                if macro_f1 > self.best_macro_f1:
                    self.best_macro_f1 = macro_f1
                    self.best_checkpoint_path = os.path.join(self.config.output_dir, "best_task_b_cross_model.pt")
                    torch.save(self.model.state_dict(), self.best_checkpoint_path)
                    print(f"    ⭐ New Best Task B Macro-F1: {macro_f1:.4f} -> Saved checkpoint.")

        # PHASE 2: Differential Fine-Tuning
        # Reload best model weights from Phase 1 to prevent carrying over overfitted weights
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            print(f"  🔄 Restoring best Phase 1 checkpoint (Macro-F1: {self.best_macro_f1:.4f}) before fine-tuning...")
            self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device))

        print(f"\n>>> [Phase 2/2] Fine-Tuning Top {self.config.unfreeze_layers} Layers for {self.config.unfreeze_phase_epochs} epochs (Early Stopping Patience: {patience})...")
        unfreeze_last_n(self.model.mmbert, self.config.unfreeze_layers)

        backbone_params = [p for p in self.model.mmbert.parameters() if p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith('mmbert') and p.requires_grad]

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.config.unfreeze_lr, 'weight_decay': self.config.weight_decay},
            {'params': head_params, 'lr': self.config.head_unfreeze_lr, 'weight_decay': self.config.weight_decay},
        ], betas=(0.9, 0.98), eps=1e-6)

        scheduler = self.build_scheduler(optimizer, self.config.unfreeze_phase_epochs)

        for epoch in range(1, self.config.unfreeze_phase_epochs + 1):
            t0 = time.time()
            train_loss, train_metrics = self.train_epoch(
                optimizer=optimizer,
                scheduler=scheduler,
                apply_fgm=getattr(self.config, 'use_fgm', True)
            )
            val_loss, val_metrics, _, _, gate_stats = self.eval_epoch()
            elapsed = time.time() - t0

            macro_f1 = val_metrics.get('hs_macro_f1', 0.0)
            no_f1 = val_metrics.get('hs_f1_no', 0.0)
            imp_f1 = val_metrics.get('hs_f1_implicit', 0.0)
            exp_f1 = val_metrics.get('hs_f1_explicit', 0.0)
            val_acc = val_metrics.get('hs_acc', 0.0)
            train_f1 = train_metrics.get('hs_macro_f1', 0.0)

            global_epoch = (self.config.freeze_phase_epochs if self.config.two_phase else 0) + epoch
            self.history.append({
                'phase': 2,
                'epoch': global_epoch,
                'train_loss': train_loss,
                'train_macro_f1': train_f1,
                'val_loss': val_loss,
                'hs_macro_f1': macro_f1,
                'hs_f1_no': no_f1,
                'hs_f1_implicit': imp_f1,
                'hs_f1_explicit': exp_f1,
                'hs_acc': val_acc,
                **gate_stats
            })

            print(f"  [P2 Epoch {epoch:02d}/{self.config.unfreeze_phase_epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} (F1: {train_f1:.4f}) | Val Loss: {val_loss:.4f} | "
                  f"Val Macro-F1: {macro_f1:.4f} | No: {no_f1:.4f} | Imp: {imp_f1:.4f} | Exp: {exp_f1:.4f} | "
                  f"Gate (All/No/Imp/Exp): {gate_stats['gate_mean']:.3f}/{gate_stats['gate_no']:.3f}/{gate_stats['gate_implicit']:.3f}/{gate_stats['gate_explicit']:.3f} [{elapsed:.1f}s]")

            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                self.best_checkpoint_path = os.path.join(self.config.output_dir, "best_task_b_cross_model.pt")
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
                print(f"    ⭐ New Best Task B Macro-F1: {macro_f1:.4f} -> Saved checkpoint.")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping triggered after {patience} epochs without Macro-F1 improvement!")
                    break

        # Load best weights
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device))
            print(f"\n🏆 Successfully reloaded best checkpoint with Macro-F1: {self.best_macro_f1:.4f}")

        # Final Evaluation
        final_loss, final_metrics, y_true, y_pred, final_gate_stats = self.eval_epoch()
        return {
            'best_macro_f1': self.best_macro_f1,
            'final_metrics': final_metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'history': self.history,
            'checkpoint_path': self.best_checkpoint_path,
            'final_gate_stats': final_gate_stats
        }


# Backward-compatible alias
TaskBCrossContextTrainer = TaskBCrossTrainer
