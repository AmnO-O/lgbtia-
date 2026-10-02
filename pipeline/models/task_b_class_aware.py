import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union

from .norm import RMSNorm
from .head import MultiSampleDropoutHead

CLASS_NO_HATE = 0
CLASS_IMPLICIT_HATE = 1
CLASS_EXPLICIT_HATE = 2
NUM_CLASSES = 3

# Structured context role IDs
ROLE_PAD = 0
ROLE_TITLE = 1
ROLE_DESC = 2
ROLE_COMMENT = 3
ROLE_SPECIAL = 4  # [CLS], [SEP] special delimiter tokens
NUM_ROLES = 5


class PureClassQueryScoringHead(nn.Module):
    """
    Shared Class Query Scoring Head.
    
    Given N class query representations [B, num_classes, d_model],
    maps each query representation z_c to a scalar score for class c via a shared projection f_theta: R^d -> R^1.
    Supports Multi-Sample Dropout (MSD) for low-variance generalization.
    """
    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: Optional[int] = None,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        dropout: float = 0.20,
        use_rmsnorm: bool = True
    ):
        super(PureClassQueryScoringHead, self).__init__()
        self.d_model = d_model
        self.use_msd = use_msd
        hidden_dim = hidden_dim or (d_model // 2)
        
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm = NormClass(d_model)

        if use_msd:
            self.msd_head = MultiSampleDropoutHead(
                in_dim=d_model,
                hidden_dim=hidden_dim,
                num_classes=1,
                use_msd=True,
                msd_dropout_rates=msd_dropout_rates or [0.10, 0.15, 0.20, 0.25, 0.30],
                use_rmsnorm=use_rmsnorm
            )
        else:
            self.linear = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                NormClass(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

    def forward(
        self,
        z_classes: torch.Tensor,
        return_all_msd_logits: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        B, num_classes, d = z_classes.shape
        # Use .contiguous().reshape(...) instead of .view(...) to handle non-contiguous memory layouts safely
        z_flat = z_classes.contiguous().reshape(B * num_classes, d)
        z_normed = self.norm(z_flat)

        if self.use_msd:
            msd_outputs = self.msd_head(z_normed, return_all_msd_logits=True) # List of [B * num_classes, 1]
            branch_logits = [out.contiguous().reshape(B, num_classes) for out in msd_outputs]
            if return_all_msd_logits and self.training:
                return branch_logits
            return torch.mean(torch.stack(branch_logits, dim=0), dim=0)
        else:
            out = self.linear(z_normed) # [B * num_classes, 1]
            return out.contiguous().reshape(B, num_classes)


class HierarchicalClassQueryHead(nn.Module):
    """
    Clean Hierarchical Two-Head Architecture for Class-Aware Prototype Queries:
    Level 1 (Super-Class Binary Hate Head):
        Inputs: [z_no; z_implicit; z_explicit] (3 * d_model)
        Preserves all 3 class representations directly without destructive averaging.
        Computes p_hate = sigmoid(W_h [z_no; z_implicit; z_explicit] + b_h)
    Level 2 (Fine-Grained Conditional Head - Implicit vs Explicit):
        Inputs: [z_implicit; z_explicit] (2 * d_model)
        Directly evaluates implicit vs explicit evidence.
        Computes p_implicit = sigmoid(W_f [z_implicit; z_explicit] + b_f)
    
    Compound Tree:
        P(no)          = 1 - p_hate
        P(yes_implicit)= p_hate * p_implicit
        P(yes_explicit)= p_hate * (1 - p_implicit)
    """
    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: Optional[int] = None,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        single_dropout: float = 0.20,
        use_rmsnorm: bool = True
    ):
        super(HierarchicalClassQueryHead, self).__init__()
        self.d_model = d_model
        self.use_msd = use_msd
        hidden_dim = hidden_dim or (d_model // 2)
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        self.dropout_rates = msd_dropout_rates or [0.10, 0.15, 0.20, 0.25, 0.30]

        # 1. Super-Class Binary Hate Head: [z_no; z_imp; z_exp] -> (3 * d_model)
        self.hate_pre_proj = nn.Linear(3 * d_model, hidden_dim)
        self.hate_act = nn.GELU()
        self.hate_norm = NormClass(hidden_dim)
        if self.use_msd:
            self.hate_dropouts = nn.ModuleList([nn.Dropout(p) for p in self.dropout_rates])
        else:
            self.hate_single_dropout = nn.Dropout(single_dropout)
        self.hate_out_proj = nn.Linear(hidden_dim, 1)

        # 2. Fine-Grained Conditional Head: [z_imp; z_exp] -> (2 * d_model)
        self.type_pre_proj = nn.Linear(2 * d_model, hidden_dim)
        self.type_act = nn.GELU()
        self.type_norm = NormClass(hidden_dim)
        if self.use_msd:
            self.type_dropouts = nn.ModuleList([nn.Dropout(p) for p in self.dropout_rates])
        else:
            self.type_single_dropout = nn.Dropout(single_dropout)
        self.type_out_proj = nn.Linear(hidden_dim, 1)

        for mod in [self.hate_pre_proj, self.hate_out_proj, self.type_pre_proj, self.type_out_proj]:
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)

    def forward(
        self,
        z_classes: torch.Tensor, # [B, 3, d_model] -> (z_no, z_implicit, z_explicit)
        return_all_msd_logits: bool = False
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        z_no = z_classes[:, 0, :]       # [B, d_model]
        z_imp = z_classes[:, 1, :]      # [B, d_model]
        z_exp = z_classes[:, 2, :]      # [B, d_model]

        # Natural, clean feature concatenation without redundant subtraction
        feat_hate = torch.cat([z_no, z_imp, z_exp], dim=-1) # [B, 3 * d_model]
        feat_type = torch.cat([z_imp, z_exp], dim=-1)       # [B, 2 * d_model]

        h_hate = self.hate_norm(self.hate_act(self.hate_pre_proj(feat_hate)))
        h_type = self.type_norm(self.type_act(self.type_pre_proj(feat_type)))

        if self.training and self.use_msd:
            outs = []
            for drop_h, drop_t in zip(self.hate_dropouts, self.type_dropouts):
                logit_hate = self.hate_out_proj(drop_h(h_hate)) # [B, 1]
                logit_type = self.type_out_proj(drop_t(h_type)) # [B, 1]

                p_hate = torch.sigmoid(logit_hate)
                p_imp = torch.sigmoid(logit_type)

                p_no = 1.0 - p_hate
                p_implicit = p_hate * p_imp
                p_explicit = p_hate * (1.0 - p_imp)
                probs = torch.cat([p_no, p_implicit, p_explicit], dim=-1) # [B, 3]

                eps = 1e-7
                compound_logits = torch.log(torch.clamp(probs, min=eps, max=1.0 - eps))

                outs.append({
                    'logit_hate': logit_hate,
                    'logit_type': logit_type,
                    'probs': probs,
                    'compound_logits': compound_logits
                })

            if return_all_msd_logits:
                return outs

            avg_logit_hate = torch.mean(torch.stack([o['logit_hate'] for o in outs], dim=0), dim=0)
            avg_logit_type = torch.mean(torch.stack([o['logit_type'] for o in outs], dim=0), dim=0)
            avg_probs = torch.mean(torch.stack([o['probs'] for o in outs], dim=0), dim=0)
            eps = 1e-7
            avg_compound = torch.log(torch.clamp(avg_probs, min=eps, max=1.0 - eps))

            return {
                'logit_hate': avg_logit_hate,
                'logit_type': avg_logit_type,
                'probs': avg_probs,
                'compound_logits': avg_compound
            }

        # Non-MSD training or Eval mode
        if self.training and not self.use_msd:
            logit_hate = self.hate_out_proj(self.hate_single_dropout(h_hate))
            logit_type = self.type_out_proj(self.type_single_dropout(h_type))
        else:
            logit_hate = self.hate_out_proj(h_hate)
            logit_type = self.type_out_proj(h_type)

        p_hate = torch.sigmoid(logit_hate)
        p_imp = torch.sigmoid(logit_type)

        p_no = 1.0 - p_hate
        p_implicit = p_hate * p_imp
        p_explicit = p_hate * (1.0 - p_imp)
        probs = torch.cat([p_no, p_implicit, p_explicit], dim=-1) # [B, 3]

        eps = 1e-7
        compound_logits = torch.log(torch.clamp(probs, min=eps, max=1.0 - eps))

        single_out = {
            'logit_hate': logit_hate,
            'logit_type': logit_type,
            'probs': probs,
            'compound_logits': compound_logits
        }
        if return_all_msd_logits:
            return [single_out]
        return single_out


class TaskBClassAwareAttentionModel(nn.Module):
    """
    Clean & Production-Grade Task B Model with Learnable Class Prototype Queries:
    
    1. Layer 0: Explicit Context Role Injection (PAD=0, Title=1, Desc=2, Comment=3).
    2. Layer 1: Vectorized Class-Aware Cross-Attention with 3 learnable prototype queries.
    3. Layer 2: Optional Inter-Class Query Self-Interaction.
    4. Layer 3: Hierarchical Two-Head / Shared Scoring Head with Multi-Sample Dropout.
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_classes: int = NUM_CLASSES,
        num_slots_per_class: int = 1,
        num_heads: int = 8,
        dropout: float = 0.20,
        use_query_interaction: bool = False,
        use_hierarchical_head: bool = True,
        hidden_dim: Optional[int] = None,
        use_rmsnorm: bool = True,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        **kwargs
    ):
        super(TaskBClassAwareAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_slots_per_class = num_slots_per_class
        self.total_queries = num_classes * num_slots_per_class
        self.num_heads = num_heads
        self.use_query_interaction = use_query_interaction
        self.use_hierarchical_head = use_hierarchical_head
        self.use_rmsnorm = use_rmsnorm
        self.use_msd = use_msd
        hidden_dim = hidden_dim or (d_model // 2)

        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm

        # ---------------------------------------------------------------------
        # LAYER 0: Pre-Backbone Role Embeddings (PAD=0, Title=1, Desc=2, Comment=3, Special=4)
        # E_{input} = E_{token} + E_{role} -> Passed directly into mmBERT inputs_embeds
        # ---------------------------------------------------------------------
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model, padding_idx=ROLE_PAD)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.role_embeddings.weight[ROLE_PAD].zero_()
        self.norm_input = NormClass(d_model)
        self.dropout_input = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 1: 3 Continuous Learnable Class Queries + Cross-Attention
        # ---------------------------------------------------------------------
        self.class_queries = nn.Parameter(torch.empty(self.total_queries, d_model))
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)
        self.query_embeddings = self.class_queries

        self.norm_q = NormClass(d_model)
        self.norm_kv = NormClass(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_cross = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 2: Optional Inter-Class Query Self-Attention Interaction
        # ---------------------------------------------------------------------
        if self.use_query_interaction and self.total_queries > 1:
            self.norm_self = NormClass(d_model)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.dropout_self = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 3: Classification Head (Hierarchical Two-Head vs Standard Pure Class Query Head)
        # ---------------------------------------------------------------------
        if self.use_hierarchical_head:
            self.scoring_head = HierarchicalClassQueryHead(
                d_model=d_model,
                hidden_dim=hidden_dim,
                use_msd=use_msd,
                msd_dropout_rates=msd_dropout_rates,
                single_dropout=dropout,
                use_rmsnorm=use_rmsnorm
            )
        else:
            self.scoring_head = PureClassQueryScoringHead(
                d_model=d_model,
                hidden_dim=hidden_dim,
                use_msd=use_msd,
                msd_dropout_rates=msd_dropout_rates,
                dropout=dropout,
                use_rmsnorm=use_rmsnorm
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        return_attention_map: bool = False,
        return_gates: bool = False,
        detach_bridge: bool = False,
        return_all_msd_logits: bool = False,
        **kwargs
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]],
        Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ]:
        B, S = input_ids.shape

        # 1. Pre-Backbone Role Embedding Injection
        # E_{input, i} = E_{token, i} + E_{role, i}
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        role_trainable = self.role_embeddings.weight.requires_grad
        needs_grad = torch.is_grad_enabled() and (backbone_trainable or role_trainable)

        can_inject_embeds = (
            hasattr(self.mmbert, 'get_input_embeddings')
            and callable(getattr(self.mmbert, 'get_input_embeddings'))
            and self.mmbert.get_input_embeddings() is not None
        )

        if can_inject_embeds:
            if not backbone_trainable:
                with torch.no_grad():
                    word_embeds = self.mmbert.get_input_embeddings()(input_ids) # [B, S, d]
            else:
                word_embeds = self.mmbert.get_input_embeddings()(input_ids)

            e_role = self.role_embeddings(role_ids) # [B, S, d]
            inputs_embeds = self.norm_input(word_embeds + e_role)
            inputs_embeds = self.dropout_input(inputs_embeds)

            if needs_grad:
                h_seq = self.mmbert(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
            else:
                with torch.no_grad():
                    h_seq = self.mmbert(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        else:
            if needs_grad:
                h_seq = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            else:
                with torch.no_grad():
                    h_seq = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 2. Expand 3 Class Prototype Queries to Batch: [B, total_queries, d_model]
        # .contiguous() ensures memory layout compatibility across CUDA devices
        q_raw = self.class_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        if q_raw.dtype != h_seq.dtype:
            q_raw = q_raw.to(h_seq.dtype)

        key_padding_mask = (attention_mask == 0)

        # 3. Layer 1: Vectorized Class-Aware Cross-Attention
        q_normed = self.norm_q(q_raw)
        kv_normed = self.norm_kv(h_seq)

        if return_attention_map:
            z_attn, attn_weights = self.cross_attention(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True
            )
        else:
            z_attn, _ = self.cross_attention(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            attn_weights = None

        z = (q_raw + self.dropout_cross(z_attn)).contiguous() # [B, total_queries, d_model]

        # 4. Layer 2: Optional Inter-Class Query Self-Interaction
        if self.use_query_interaction and self.total_queries > 1:
            z_normed = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_normed,
                key=z_normed,
                value=z_normed,
                need_weights=False
            )
            z = (z + self.dropout_self(z_self)).contiguous()

        # Pool slots per class if num_slots_per_class > 1
        if self.num_slots_per_class == 1:
            z_classes = z.contiguous().reshape(B, self.num_classes, self.d_model)
        else:
            z_classes = z.contiguous().reshape(B, self.num_classes, self.num_slots_per_class, self.d_model).mean(dim=2)

        # 5. Layer 3: Shared Class Query Scoring Head
        out_logits = self.scoring_head(
            z_classes=z_classes,
            return_all_msd_logits=return_all_msd_logits
        )

        # 7. Task C Latent Bridge: Probability-Weighted Blend
        if isinstance(out_logits, dict):
            probs = out_logits['probs']
        elif isinstance(out_logits, list):
            if isinstance(out_logits[0], dict):
                probs = torch.mean(torch.stack([o['probs'] for o in out_logits], dim=0), dim=0)
            else:
                eval_logits = torch.mean(torch.stack(out_logits, dim=0), dim=0)
                probs = F.softmax(eval_logits, dim=-1)
        else:
            probs = F.softmax(out_logits, dim=-1)

        h_B = torch.sum(probs.unsqueeze(-1) * z_classes, dim=1)
        if detach_bridge:
            h_B = h_B.detach()

        if return_gates:
            return out_logits, h_B, None, attn_weights

        return out_logits, h_B, attn_weights

    @property
    def num_queries(self) -> int:
        return self.total_queries

    def init_queries_from_text(self, tokenizer=None, device=None) -> None:
        pass
