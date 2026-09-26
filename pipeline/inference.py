import os
import json
from typing import Dict, Any, List, Optional
import torch
import numpy as np

from .config import PipelineConfig, ID_ORDER, SCOPE_DIM, IDX2HATE
from .data import safe_clean, decode_target
from .models.mmbert import MMBertTransformerModel
from .models.transformer import PytorchTransformerModel
from .models.lstm import PytorchRNNLSTM
from .models.rnn import VanillaRNNModel
from .models.classifier import FeatureClassifier, MMBertFeatureClassifier


class StereoQueerPredictor:
    """Inference engine for StereoQueerEval models."""
    def __init__(self, checkpoint_path: str, config: Optional[PipelineConfig] = None,
                 config_path: Optional[str] = None, vocab_path: Optional[str] = None,
                 device: Optional[str] = None):
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            self.config = PipelineConfig.from_json(config_path)
        else:
            self.config = PipelineConfig()

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = None
        self.vocab = None
        self.model = self._load_model(checkpoint_path, vocab_path)

    def _load_model(self, checkpoint_path: str, vocab_path: Optional[str] = None) -> torch.nn.Module:
        if self.config.embed_source == 'mmbert':
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.mmbert_model_name)
            backbone = AutoModel.from_pretrained(self.config.mmbert_model_name)

            if self.config.model_type == 'task_b_class_aware':
                from .models.task_b_class_aware import TaskBClassAwareAttentionModel
                model = TaskBClassAwareAttentionModel(
                    mmbert_model=backbone,
                    d_model=self.config.mmbert_dim,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout,
                    use_query_interaction=self.config.use_query_interaction
                )
            elif self.config.model_type == 'feature_mlp':
                model = MMBertFeatureClassifier(
                    backbone,
                    d_model=self.config.mmbert_dim,
                    target_dim=self.config.target_dim
                )
            else:
                model = MMBertTransformerModel(
                    backbone,
                    d_model=self.config.mmbert_dim,
                    target_dim=self.config.target_dim
                )
        else:
            if vocab_path and os.path.exists(vocab_path):
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self.vocab = json.load(f)
                vocab_size = len(self.vocab)
            else:
                vocab_size = self.config.vocab_size

            if self.config.model_type == 'bilstm':
                model = PytorchRNNLSTM(
                    vocab_size=vocab_size,
                    embedding_dim=self.config.embedding_dim,
                    hidden_dim=self.config.hidden_dim,
                    target_dim=self.config.target_dim
                )
            elif self.config.model_type == 'rnn':
                model = VanillaRNNModel(
                    vocab_size=vocab_size,
                    embedding_dim=self.config.embedding_dim,
                    hidden_dim=self.config.hidden_dim,
                    target_dim=self.config.target_dim
                )
            elif self.config.model_type == 'feature_mlp':
                model = FeatureClassifier(
                    in_dim=self.config.embedding_dim,
                    target_dim=self.config.target_dim,
                    vocab_size=vocab_size
                )
            else:
                model = PytorchTransformerModel(
                    vocab_size=vocab_size,
                    embedding_dim=self.config.embedding_dim,
                    num_heads=self.config.num_heads,
                    num_layers=self.config.num_layers,
                    target_dim=self.config.target_dim
                )

        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state)
        else:
            print(f"Warning: Checkpoint '{checkpoint_path}' not found. Initialized with uncalibrated weights.")

        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_one(self, comment: str, title: str = "", description: str = "") -> Dict[str, Any]:
        """Runs multi-task or Task B prediction on a single text triplet."""
        clean_c = safe_clean(comment)
        clean_t = safe_clean(title)
        clean_d = safe_clean(description)
        compound_text = f"comment: {clean_c} [SEP] title: {clean_t} [SEP] desc: {clean_d}"

        if self.config.model_type == 'task_b_class_aware':
            from .models.task_b_class_aware import ROLE_PAD, ROLE_TITLE, ROLE_DESC, ROLE_COMMENT
            comm_ids = self.tokenizer.encode(f"comment: {clean_c}", add_special_tokens=False) if clean_c else []
            title_ids = self.tokenizer.encode(f"title: {clean_t}", add_special_tokens=False) if clean_t else []
            desc_ids = self.tokenizer.encode(f"desc: {clean_d}", add_special_tokens=False) if clean_d else []

            cls_id = getattr(self.tokenizer, 'cls_token_id', 101) or 101
            sep_id = getattr(self.tokenizer, 'sep_token_id', 102) or 102
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0

            max_len = self.config.max_length
            overhead = 4
            available = max(10, max_len - overhead)

            c_budget = int(available * 0.70)
            t_budget = int(available * 0.18)
            d_budget = available - c_budget - t_budget

            comm_ids = comm_ids[:c_budget]
            title_ids = title_ids[:t_budget]
            desc_ids = desc_ids[:d_budget]

            all_ids = [cls_id]
            all_roles = [ROLE_PAD]

            # 1. Comment
            if comm_ids:
                all_ids.extend(comm_ids)
                all_roles.extend([ROLE_COMMENT] * len(comm_ids))
            all_ids.append(sep_id)
            all_roles.append(ROLE_PAD)

            # 2. Title
            if title_ids:
                all_ids.extend(title_ids)
                all_roles.extend([ROLE_TITLE] * len(title_ids))
            all_ids.append(sep_id)
            all_roles.append(ROLE_PAD)

            # 3. Description
            if desc_ids:
                all_ids.extend(desc_ids)
                all_roles.extend([ROLE_DESC] * len(desc_ids))
            all_ids.append(sep_id)
            all_roles.append(ROLE_PAD)

            max_len = self.config.max_length
            if len(all_ids) < max_len:
                pad_len = max_len - len(all_ids)
                all_ids += [pad_id] * pad_len
                all_roles += [ROLE_PAD] * pad_len
                mask = [1] * (max_len - pad_len) + [0] * pad_len
            else:
                all_ids = all_ids[:max_len]
                all_roles = all_roles[:max_len]
                mask = [1] * max_len

            inp_ids = torch.tensor([all_ids], dtype=torch.long, device=self.device)
            att_mask = torch.tensor([mask], dtype=torch.long, device=self.device)
            role_ids = torch.tensor([all_roles], dtype=torch.long, device=self.device)

            logits, _, attn = self.model(inp_ids, att_mask, role_ids, return_attention_map=True)
            hs_probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            hs_idx = int(np.argmax(hs_probs))
            hs_label = IDX2HATE.get(hs_idx, "no")

            return {
                "input_text": compound_text,
                "hate_speech": {
                    "label": hs_label,
                    "probabilities": {
                        "no": float(hs_probs[0]),
                        "yes_implicit": float(hs_probs[1]),
                        "yes_explicit": float(hs_probs[2]),
                    }
                }
            }

        if self.config.embed_source == 'mmbert' and self.tokenizer is not None:
            sep = getattr(self.tokenizer, 'sep_token', '[SEP]') or '[SEP]'
            tok_text = compound_text.replace('[SEP]', sep)
            enc = self.tokenizer(
                tok_text,
                truncation=True,
                max_length=self.config.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            ids = enc['input_ids'].to(self.device)
            mask = enc['attention_mask'].to(self.device)
            st_logits, hs_logits, tg_logits = self.model(ids, mask)
        else:
            words = compound_text.split()
            ids = [self.vocab.get(w, 1) if self.vocab else 1 for w in words]
            if len(ids) < self.config.max_length:
                ids += [0] * (self.config.max_length - len(ids))
            else:
                ids = ids[:self.config.max_length]
            inp = torch.tensor([ids], dtype=torch.long, device=self.device)
            st_logits, hs_logits, tg_logits = self.model(inp)

        # 1. Stereotype
        st_prob = float(torch.sigmoid(st_logits.squeeze(-1))[0].cpu().item())
        st_label = "yes" if st_prob >= 0.5 else "no"

        # 2. Hate speech
        hs_probs = torch.softmax(hs_logits[0], dim=-1).cpu().numpy()
        hs_idx = int(np.argmax(hs_probs))
        hs_label = IDX2HATE.get(hs_idx, "no")

        # 3. Target
        tg_probs = torch.sigmoid(tg_logits[0]).cpu().numpy()
        tg_str = decode_target(tg_probs)

        active_identities = [ID_ORDER[i] for i in range(SCOPE_DIM) if tg_probs[i] >= 0.5]
        scope = "none"
        if tg_str != "none":
            scope = "group" if tg_probs[SCOPE_DIM] >= 0.5 else "individual"

        return {
            "input_text": compound_text,
            "stereotype": {
                "label": st_label,
                "confidence": st_prob if st_label == "yes" else 1.0 - st_prob,
                "probability_yes": st_prob,
            },
            "hate_speech": {
                "label": hs_label,
                "probabilities": {
                    "no": float(hs_probs[0]),
                    "yes_implicit": float(hs_probs[1]),
                    "yes_explicit": float(hs_probs[2]),
                }
            },
            "target": {
                "raw_str": tg_str,
                "scope": scope,
                "identities": active_identities,
                "bitmask_probabilities": {ID_ORDER[i]: float(tg_probs[i]) for i in range(SCOPE_DIM)},
                "scope_probability_group": float(tg_probs[SCOPE_DIM])
            }
        }
