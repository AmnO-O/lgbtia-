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
ROLE_HINT = 5     # Optional teacher rationale tokens
NUM_ROLES = 6


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding exclusively to a single multi-head tensor (Keys).
    tensor: [B, num_heads, L, head_dim]
    cos, sin: [1, 1, L, head_dim] or [B, 1, L, head_dim] or [1, L, 1, head_dim] or [B, L, head_dim]
    """
    if cos.dim() == 4 and cos.shape[1] == tensor.shape[2] and cos.shape[2] == 1:
        # [1, L, 1, d_h] -> [1, 1, L, d_h]
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
    elif cos.dim() == 3:
        # [B, L, d_h] or [1, L, d_h] -> [B, 1, L, d_h]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.dim() == 2:
        # [L, d_h] -> [1, 1, L, d_h]
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)

    return (tensor * cos) + (rotate_half(tensor) * sin)


class PretrainedCrossAttentionLayer(nn.Module):
    """
    Asymmetric Cross-Attention whose Q/K/V/O projections are borrowed directly
    from one pretrained ModernBERT attention block.

    Academic Framing:
      "We transplant pretrained Q/K/V/O projections while applying 
       asymmetric RoPE to the context keys."
    """
    def __init__(
        self,
        pretrained_attn: nn.Module,
        rotary_emb: Optional[nn.Module] = None,
        pretrained_norm: Optional[nn.Module] = None,
        use_prenorm: bool = False
    ):
        super(PretrainedCrossAttentionLayer, self).__init__()
        self.attn = pretrained_attn
        self.use_prenorm = use_prenorm

        # Borrow pretrained projection layers directly
        if not hasattr(pretrained_attn, 'Wqkv') or not hasattr(pretrained_attn, 'Wo'):
            raise AttributeError("Target attention block must contain ModernBERT 'Wqkv' and 'Wo' projection matrices!")

        self.Wqkv = pretrained_attn.Wqkv
        self.Wo = pretrained_attn.Wo

        # Optional: Borrow pretrained LayerNorm / RMSNorm for Experiment B
        self.norm = pretrained_norm if use_prenorm else None

        # Wire rotary embedding explicitly from ModernBertModel
        self.rotary_emb = rotary_emb or getattr(pretrained_attn, "rotary_emb", None)

        self.num_heads = getattr(pretrained_attn.config, "num_attention_heads", 12)
        self.head_dim = getattr(pretrained_attn, "head_dim", 64)
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        z: torch.Tensor,
        h_context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            z: [B, K, D] - Query latents (Position-Free, Unrotated)
            h_context: [B, L, D] - Context token representations
            context_mask: [B, L] - Non-padding mask (1 for valid token, 0 for pad)
            position_ids: [B, L] - Positional indices for context tokens (0 to L-1)
            position_embeddings: Optional precomputed tuple of (cos, sin) from ModernBERT rotary_emb
        Returns:
            z_out: [B, K, D] - Updated query latents
            attn_weights: [B, K, L] - Averaged cross-attention alignment weights across heads
        """
        B, K, D = z.shape
        _, L, _ = h_context.shape

        # Optional Pre-Normalization (Experiment B control)
        z_in = self.norm(z) if (self.use_prenorm and self.norm is not None) else z
        h_in = self.norm(h_context) if (self.use_prenorm and self.norm is not None) else h_context

        # 1. Chunk fused ModernBERT Wqkv weight [3D, D] into Q, K, V [D, D]
        q_weight, k_weight, v_weight = self.Wqkv.weight.chunk(3, dim=0)
        q_bias = k_bias = v_bias = None
        if getattr(self.Wqkv, 'bias', None) is not None:
            q_bias, k_bias, v_bias = self.Wqkv.bias.chunk(3, dim=0)

        q = F.linear(z_in, q_weight, q_bias)       # [B, K, D]
        k = F.linear(h_in, k_weight, k_bias)       # [B, L, D]
        v = F.linear(h_in, v_weight, v_bias)       # [B, L, D]

        # 2. Reshape into Multi-Head format [B, num_heads, SeqLen, head_dim]
        q = q.contiguous().view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, d_h]
        k = k.contiguous().view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]
        v = v.contiguous().view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]

        # 3. Intentional Asymmetric RoPE: Apply Native ModernBERT RoPE Exclusively to Keys (K)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            k = apply_rotary_pos_emb_single(k, cos, sin)
        elif self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(L, device=h_context.device).unsqueeze(0).expand(B, -1)
            cos, sin = self.rotary_emb(v, position_ids)
            k = apply_rotary_pos_emb_single(k, cos, sin)

        # 4. Global Asymmetric Attention Matrix (Full Non-PAD Visibility)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # [B, H, K, L]
        if context_mask is not None:
            mask_4d = (1.0 - context_mask.unsqueeze(1).unsqueeze(2).to(scores.dtype)) * -10000.0
            scores = scores + mask_4d

        attn_weights = F.softmax(scores, dim=-1) # [B, H, K, L]
        attn_out = torch.matmul(attn_weights, v) # [B, H, K, d_h]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)

        # 5. Pretrained Wo Linear Projection with Residual Connection
        z_out = z + self.Wo(attn_out)
        
        # Average attention across heads for interpretability: [B, K, L]
        avg_attn = attn_weights.mean(dim=1)
        return z_out, avg_attn


class PureClassQueryScoringHead(nn.Module):
    """
    Shared Class Query Scoring Head.
    Given N class query representations [B, num_classes, d_model],
    maps each query representation z_c to a scalar score for class c via a shared projection f_theta: R^d -> R^1.
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
        z_flat = z_classes.contiguous().reshape(B * num_classes, d)
        z_normed = self.norm(z_flat)

        if self.use_msd:
            msd_outputs = self.msd_head(z_normed, return_all_msd_logits=True)
            branch_logits = [out.contiguous().reshape(B, num_classes) for out in msd_outputs]
            if return_all_msd_logits and self.training:
                return branch_logits
            return torch.mean(torch.stack(branch_logits, dim=0), dim=0)
        else:
            out = self.linear(z_normed)
            return out.contiguous().reshape(B, num_classes)


class HierarchicalClassQueryHead(nn.Module):
    """
    Clean Hierarchical Two-Head Architecture for Class-Aware Prototype Queries:
    Level 1 (Super-Class Binary Hate Head):
        Inputs: [z_no; z_implicit; z_explicit] (3 * d_model)
        Computes p_hate = sigmoid(W_h [z_no; z_implicit; z_explicit] + b_h)
    Level 2 (Fine-Grained Conditional Head - Implicit vs Explicit):
        Inputs: [z_implicit; z_explicit] (2 * d_model)
        Computes p_implicit = sigmoid(W_f [z_implicit; z_explicit] + b_f)
    
    Compound Tree Decomposition:
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

        feat_hate = torch.cat([z_no, z_imp, z_exp], dim=-1) # [B, 3 * d_model]
        feat_type = torch.cat([z_imp, z_exp], dim=-1)       # [B, 2 * d_model]

        h_hate = self.hate_norm(self.hate_act(self.hate_pre_proj(feat_hate)))
        h_type = self.type_norm(self.type_act(self.type_pre_proj(feat_type)))

        if self.training and self.use_msd:
            outs = []
            for drop_h, drop_t in zip(self.hate_dropouts, self.type_dropouts):
                logit_hate = self.hate_out_proj(drop_h(h_hate)) # [B, 1]
                logit_type = self.type_out_proj(drop_t(h_type)) # [B, 1]

                log_p_hate = F.logsigmoid(logit_hate)
                log_p_no = F.logsigmoid(-logit_hate)

                log_p_imp_given_hate = F.logsigmoid(logit_type)
                log_p_exp_given_hate = F.logsigmoid(-logit_type)

                log_p_implicit = log_p_hate + log_p_imp_given_hate
                log_p_explicit = log_p_hate + log_p_exp_given_hate

                log_probs = torch.cat([log_p_no, log_p_implicit, log_p_explicit], dim=-1) # [B, 3]
                probs = torch.exp(log_probs) # [B, 3]

                outs.append({
                    'logit_hate': logit_hate,
                    'logit_type': logit_type,
                    'probs': probs,
                    'log_probs': log_probs,
                    'compound_logits': log_probs
                })

            if return_all_msd_logits:
                return outs

            avg_logit_hate = torch.mean(torch.stack([o['logit_hate'] for o in outs], dim=0), dim=0)
            avg_logit_type = torch.mean(torch.stack([o['logit_type'] for o in outs], dim=0), dim=0)
            avg_probs = torch.mean(torch.stack([o['probs'] for o in outs], dim=0), dim=0)
            eps = 1e-7
            avg_log_probs = torch.log(torch.clamp(avg_probs, min=eps, max=1.0 - eps))

            return {
                'logit_hate': avg_logit_hate,
                'logit_type': avg_logit_type,
                'probs': avg_probs,
                'log_probs': avg_log_probs,
                'compound_logits': avg_log_probs
            }

        # Non-MSD training or Eval mode
        if self.training and not self.use_msd:
            logit_hate = self.hate_out_proj(self.hate_single_dropout(h_hate))
            logit_type = self.type_out_proj(self.type_single_dropout(h_type))
        else:
            logit_hate = self.hate_out_proj(h_hate)
            logit_type = self.type_out_proj(h_type)

        log_p_hate = F.logsigmoid(logit_hate)
        log_p_no = F.logsigmoid(-logit_hate)

        log_p_imp_given_hate = F.logsigmoid(logit_type)
        log_p_exp_given_hate = F.logsigmoid(-logit_type)

        log_p_implicit = log_p_hate + log_p_imp_given_hate
        log_p_explicit = log_p_hate + log_p_exp_given_hate

        log_probs = torch.cat([log_p_no, log_p_implicit, log_p_explicit], dim=-1) # [B, 3]
        probs = torch.exp(log_probs) # [B, 3]

        single_out = {
            'logit_hate': logit_hate,
            'logit_type': logit_type,
            'probs': probs,
            'log_probs': log_probs,
            'compound_logits': log_probs
        }
        if return_all_msd_logits:
            return [single_out]
        return single_out


class TaskBClassAwareAttentionModel(nn.Module):
    """
    Task B Specialized Architecture with Pretrained Attention Weight Borrowing:
    
    1. Layer 0: Context Role Injection (PAD=0, Title=1, Desc=2, Comment=3, Special=4).
    2. Strict Layer Resolution:
       - Directly inspects `.model.layers` (AutoModelForMaskedLM / ModernBertModel) or `.layers`.
       - Enforces exactly 22 pretrained layers.
       - Discovers `rotary_emb` on ModernBertModel.
    3. Parameter Freeze Enforcement (Scientific Control):
       - If freeze_backbone=True: Freezes entire mmBERT backbone.
       - If freeze_cross_projections=True: Explicitly freezes borrowed Wqkv and Wo in cross_layers.
       - Explicitly enables gradients ONLY on class_queries and scoring_head (and role_embeddings).
    4. Genuine H_17 Extraction:
       - Computes position_embeddings via self.rotary_emb.
       - Runs only Layers 1 to 17 (0:17) passing position_embeddings and attention_mask.
    5. Pretrained Attention Borrowing Stack:
       - Exactly 5 layers (18 to 22, indices 17:22) applied sequentially to query latents Z.
       - Asymmetric RoPE: Unrotated Q + Rotated K via the precomputed/callable rotary_emb.
    6. Hierarchical LogSigmoid Tree Head.
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_classes: int = NUM_CLASSES,
        num_slots_per_class: int = 1,
        num_heads: int = 12,
        dropout: float = 0.20,
        use_query_interaction: bool = False,
        use_hierarchical_head: bool = True,
        hidden_dim: Optional[int] = None,
        use_rmsnorm: bool = True,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        cross_layer_start: int = 17, # Zero-based: index 17 corresponds to Layer 18
        use_prenorm: bool = False,
        freeze_cross_projections: bool = True, # Enforce frozen borrowed Wqkv, Wo
        freeze_backbone: bool = True,           # Enforce frozen backbone for Stage 1
        train_role_embeddings: bool = True,     # Role embedding trainability control
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
        self.cross_layer_start = cross_layer_start
        self.use_prenorm = use_prenorm
        self.freeze_cross_projections = freeze_cross_projections
        self.freeze_backbone = freeze_backbone
        self.train_role_embeddings = train_role_embeddings
        hidden_dim = hidden_dim or (d_model // 2)

        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm

        # ---------------------------------------------------------------------
        # 1. STRICT LAYER RESOLUTION & ROTARY EMBEDDING DISCOVERY
        # ---------------------------------------------------------------------
        base_model = mmbert_model
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            base_model = base_model.model
        elif hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layers"):
            base_model = base_model.encoder
        elif hasattr(base_model, "layers"):
            pass
        else:
            raise TypeError(
                "Scientific Integrity Check Failed: Expected a ModernBERT/mmBERT backbone "
                "exposing `.model.layers`, `.encoder.layers`, or `.layers`."
            )

        self.encoder_layers = base_model.layers
        total_layers = len(self.encoder_layers)
        if total_layers != 22:
            raise ValueError(
                f"Scientific Integrity Check Failed: Expected exactly 22 ModernBERT layers, got {total_layers}."
            )

        # Locate rotary_emb on parent ModernBertModel / mmBERT backbone
        rotary_emb = None
        for candidate in [base_model, mmbert_model, getattr(mmbert_model, "model", None)]:
            if candidate is not None and hasattr(candidate, "rotary_emb") and getattr(candidate, "rotary_emb") is not None:
                rotary_emb = getattr(candidate, "rotary_emb")
                break

        # Fallback inspection on first attention layer if model-level attribute is absent
        if rotary_emb is None and len(self.encoder_layers) > 0:
            first_attn = getattr(self.encoder_layers[0], 'attn', getattr(self.encoder_layers[0], 'attention', None))
            rotary_emb = getattr(first_attn, 'rotary_emb', None)

        if rotary_emb is None:
            raise AttributeError(
                "Scientific Integrity Check Failed: Cannot locate ModernBERT 'rotary_emb' module "
                "on backbone. Keys (K) require genuine RoPE to preserve contextual token geometry!"
            )
        self.rotary_emb = rotary_emb

        # ---------------------------------------------------------------------
        # LAYER 0: Context Role Embeddings (PAD=0, Title=1, Desc=2, Comment=3, Special=4)
        # ---------------------------------------------------------------------
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model, padding_idx=ROLE_PAD)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.role_embeddings.weight[ROLE_PAD].zero_()
        self.norm_input = NormClass(d_model)
        self.dropout_input = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 1: Trainable Class Prototype Queries Z_0 [K, D]
        # ---------------------------------------------------------------------
        self.class_queries = nn.Parameter(torch.empty(self.total_queries, d_model))
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)
        self.query_embeddings = self.class_queries

        # ---------------------------------------------------------------------
        # LAYER 2: Extract Exactly Layers 18 to 22 (Indices 17 to 21)
        # ---------------------------------------------------------------------
        target_layers = self.encoder_layers[cross_layer_start:22]
        if len(target_layers) != 5:
            raise ValueError(f"Expected exactly 5 target borrowed layers, got {len(target_layers)}")

        self.cross_layers = nn.ModuleList()
        for layer in target_layers:
            attn_mod = getattr(layer, 'attn', getattr(layer, 'attention', None))
            norm_mod = getattr(layer, 'attn_norm', getattr(layer, 'input_layernorm', None))
            if attn_mod is None or not hasattr(attn_mod, 'Wqkv') or not hasattr(attn_mod, 'Wo'):
                raise AttributeError("Target layer missing required ModernBERT 'Wqkv' or 'Wo' attention weights!")

            cross_mod = PretrainedCrossAttentionLayer(
                pretrained_attn=attn_mod,
                rotary_emb=self.rotary_emb,
                pretrained_norm=norm_mod,
                use_prenorm=use_prenorm
            )
            self.cross_layers.append(cross_mod)

        # ---------------------------------------------------------------------
        # LAYER 3: Optional Inter-Class Query Self-Interaction
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
        # LAYER 4: Classification Head (Hierarchical Two-Head vs Standard)
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

        # ---------------------------------------------------------------------
        # 2. ENFORCE SCIENTIFIC FREEZING RULES
        # ---------------------------------------------------------------------
        self.enforce_freeze_rules()

    def enforce_freeze_rules(self):
        """
        Enforces clean separation between frozen pretrained projections
        and trainable query directions.
        """
        if self.freeze_backbone:
            for p in self.mmbert.parameters():
                p.requires_grad = False

        if self.freeze_cross_projections:
            for layer in self.cross_layers:
                for p in layer.Wqkv.parameters():
                    p.requires_grad = False
                for p in layer.Wo.parameters():
                    p.requires_grad = False
                if layer.norm is not None:
                    for p in layer.norm.parameters():
                        p.requires_grad = False

        # Explicitly enable gradients for trainable components
        self.class_queries.requires_grad = True
        for p in self.scoring_head.parameters():
            p.requires_grad = True

        if hasattr(self, 'self_attention'):
            for p in self.self_attention.parameters():
                p.requires_grad = True

        self.role_embeddings.weight.requires_grad = self.train_role_embeddings
        self.norm_input.weight.requires_grad = self.train_role_embeddings
        if hasattr(self.norm_input, 'bias') and self.norm_input.bias is not None:
            self.norm_input.bias.requires_grad = self.train_role_embeddings

    def extract_h17(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Genuinely executes only layers 0 to cross_layer_start-1 (Layers 1 to 17)
        to extract H_17. Completely omits layers 18 to 22 and final_norm.
        
        Properly computes position_embeddings via self.rotary_emb and passes them
        into ModernBertEncoderLayer.forward(...) to avoid TypeError unpacking None.
        """
        B, L, _ = inputs_embeds.shape
        position_ids = torch.arange(L, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
        
        # Compute position embeddings required by ModernBert layers: (cos, sin)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for i in range(self.cross_layer_start):
            layer_module = self.encoder_layers[i]
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        return hidden_states, position_embeddings

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
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        role_trainable = self.role_embeddings.weight.requires_grad
        needs_grad = torch.is_grad_enabled() and (backbone_trainable or role_trainable)

        # Retrieve embedding module
        embed_fn = None
        if hasattr(self.mmbert, 'get_input_embeddings') and callable(self.mmbert.get_input_embeddings()):
            embed_fn = self.mmbert.get_input_embeddings()
        elif hasattr(self.mmbert, 'model') and hasattr(self.mmbert.model, 'embeddings'):
            embed_fn = self.mmbert.model.embeddings
        elif hasattr(self.mmbert, 'embeddings'):
            embed_fn = self.mmbert.embeddings

        if embed_fn is None:
            raise AttributeError("Cannot locate input embeddings in backbone model!")

        if not backbone_trainable:
            with torch.no_grad():
                word_embeds = embed_fn(input_ids)
        else:
            word_embeds = embed_fn(input_ids)

        e_role = self.role_embeddings(role_ids)
        inputs_embeds = self.norm_input(word_embeds + e_role)
        inputs_embeds = self.dropout_input(inputs_embeds)

        # 2. Genuine H_17 Extraction (Pass through layers 1 to 17 only with position_embeddings)
        if needs_grad:
            h_seq, pos_emb = self.extract_h17(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                h_seq, pos_emb = self.extract_h17(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # 3. Expand Class Prototype Queries: Z_0 [B, total_queries, d_model]
        z = self.class_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        if z.dtype != h_seq.dtype:
            z = z.to(h_seq.dtype)

        # 4. Cross-Attention Sequential Processing (Layers 18 to 22)
        last_attn_weights = None
        for cross_layer in self.cross_layers:
            z, last_attn_weights = cross_layer(
                z=z,
                h_context=h_seq,
                context_mask=attention_mask,
                position_embeddings=pos_emb
            )

        # 5. Optional Inter-Class Query Self-Interaction
        if self.use_query_interaction and self.total_queries > 1:
            z_normed = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_normed,
                key=z_normed,
                value=z_normed,
                need_weights=False
            )
            z = (z + self.dropout_self(z_self)).contiguous()

        # 6. Slot Aggregation (if num_slots_per_class > 1)
        if self.num_slots_per_class == 1:
            z_classes = z.contiguous().reshape(B, self.num_classes, self.d_model)
        else:
            z_classes = z.contiguous().reshape(B, self.num_classes, self.num_slots_per_class, self.d_model).mean(dim=2)

        # 7. Classification Head Readout
        out_logits = self.scoring_head(
            z_classes=z_classes,
            return_all_msd_logits=return_all_msd_logits
        )

        # 8. Task C Latent Bridge (Probability-Weighted Blend)
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
            return out_logits, h_B, None, last_attn_weights

        return out_logits, h_B, last_attn_weights

    @property
    def num_queries(self) -> int:
        return self.total_queries

    def init_queries_from_text(self, tokenizer=None, device=None) -> None:
        pass
