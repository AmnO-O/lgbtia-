from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from .data import decode_target

@torch.no_grad()
def run_model_inference(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                        device: torch.device, is_mmbert_tf: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs a forward pass across the data loader and extracts prediction arrays."""
    model.eval()
    st_list, hs_list, tg_list = [], [], []

    for batch in loader:
        if is_mmbert_tf:
            ids, mask = batch[0].to(device), batch[1].to(device)
            st_logits, hs_logits, tg_logits = model(ids, mask)
        else:
            inputs = batch[0].to(device)
            st_logits, hs_logits, tg_logits = model(inputs)

        st_list.append(torch.sigmoid(st_logits.squeeze(-1)).cpu().numpy())
        hs_list.append(torch.argmax(hs_logits, dim=1).cpu().numpy())
        tg_list.append(torch.sigmoid(tg_logits).cpu().numpy())

    st_probs = np.concatenate(st_list).ravel()
    hs_preds = np.concatenate(hs_list)
    tg_probs = np.concatenate(tg_list, axis=0)

    return st_probs, hs_preds, tg_probs


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task: str = "hs"
) -> Dict[str, float]:
    """
    Computes detailed per-class and macro classification metrics.
    For Task B (Hate Speech):
      - hs_acc: Overall Accuracy
      - hs_macro_f1: Macro-Averaged F1 across (no, implicit, explicit)
      - hs_f1_no: F1 score for class 0 (no hate)
      - hs_f1_implicit: F1 score for class 1 (implicit hate)
      - hs_f1_explicit: F1 score for class 2 (explicit hate)
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    # Compute per-class precision, recall, f1
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    metrics = {
        f"{task}_acc": acc,
        f"{task}_macro_f1": macro_f1,
        f"{task}_f1_no": float(f1s[0]),
        f"{task}_f1_implicit": float(f1s[1]),
        f"{task}_f1_explicit": float(f1s[2]),
        f"{task}_prec_implicit": float(precisions[1]),
        f"{task}_recall_implicit": float(recalls[1]),
    }
    return metrics


def evaluate_stereoqueer(
    model: torch.nn.Module,
    df: pd.DataFrame,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    is_mmbert_tf: bool = True
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Computes official StereoQueerEval evaluation metrics:
      - Stereotype: accuracy + macro-F1
      - Hate speech: accuracy + macro-F1 (3 classes: no, implicit, explicit)
      - Target: string exact-match accuracy against gold annotations
      - Per-language metrics breakdown (EN, IT, NL)
    """
    st_probs, hs_preds, tg_probs = run_model_inference(model, loader, device, is_mmbert_tf)
    st_preds = (st_probs >= 0.5).astype(int)
    tg_pred_str = np.array([decode_target(v) for v in tg_probs])

    gold_st = df['st_y'].values.astype(int)
    gold_hs = df['hs_y'].values.astype(int)
    gold_tg = df['target'].fillna('none').values

    metrics = {
        'st_acc': float(accuracy_score(gold_st, st_preds)),
        'st_f1': float(f1_score(gold_st, st_preds, average='macro', zero_division=0)),
        'hs_acc': float(accuracy_score(gold_hs, hs_preds)),
        'hs_f1': float(f1_score(gold_hs, hs_preds, average='macro', zero_division=0)),
        'tg_exact_match': float((tg_pred_str == gold_tg).mean()),
    }
    # Weighted composite score
    metrics['macro_avg_f1'] = float((metrics['st_f1'] + metrics['hs_f1'] + metrics['tg_exact_match']) / 3.0)

    # Per-language breakdown if available
    per_language = {}
    if 'lang' in df.columns:
        for lang in sorted(df['lang'].unique()):
            idx = (df['lang'].values == lang)
            sub_gold_st = gold_st[idx]
            sub_st_preds = st_preds[idx]
            sub_gold_hs = gold_hs[idx]
            sub_hs_preds = hs_preds[idx]
            sub_gold_tg = gold_tg[idx]
            sub_tg_preds = tg_pred_str[idx]

            per_language[lang] = {
                'count': int(idx.sum()),
                'st_f1': float(f1_score(sub_gold_st, sub_st_preds, average='macro', zero_division=0)),
                'hs_f1': float(f1_score(sub_gold_hs, sub_hs_preds, average='macro', zero_division=0)),
                'tg_exact_match': float((sub_tg_preds == sub_gold_tg).mean()),
            }

    predictions = {
        'st_probs': st_probs,
        'st_preds': st_preds,
        'hs_preds': hs_preds,
        'tg_pred_str': tg_pred_str,
        'per_language': per_language
    }

    return metrics, predictions


def print_metrics(metrics: Dict[str, float], title: str = "VALIDATION"):
    """Pretty prints evaluation metrics to console."""
    print(f"\n==================== {title} ====================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.4f}")
        else:
            print(f"  {k:<18}: {v}")
    print("==================================================\n")
