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
        """Runs multi-task prediction on a single text triplet."""
        compound_text = safe_clean(f"{comment} [SEP] {title} [SEP] {description}")

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
