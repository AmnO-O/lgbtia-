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
    Applies Rotary Position Embedding exclusively to a single multi-head tensor (e.g. Keys).
    tensor: [B, num_heads, L, head_dim]
    cos, sin: [1, 1, L, head_dim] or [B, 1, L, head_dim]
    """
    return (tensor * cos) + (rotate_half(tensor) * sin)


class PretrainedCrossAttentionLayer(nn.Module):
    """
    Asymmetric Cross-Attention whose Q/K/V/O projections are borrowed directly
    from one pretrained ModernBERT attention block.

    Scientific Framing (Parameter-Transplanted Cross-Attention):
      - We transplant the linear projection parameters (Wqkv, Wo) while replacing
        the pretrained self-attention topology with global latent-to-token cross-attention.
      - Q is unrotated: Class queries represent abstract, position-free conceptual probes.
      - K preserves full ModernBERT RoPE: Maintains contextual token order geometry.
      - Global non-padding visibility: Cross-attention mask only removes PAD tokens,
        unconstrained by local sliding-window restrictions.
      - Configurable Pre-Normalization (Exp A: False, Exp B: True).
    """
    def __init__(
        self,
        pretrained_attn: nn.Module,
        pretrained_norm: Optional[nn.Module] = None,
        use_prenorm: bool = False
    ):
        super(PretrainedCrossAttentionLayer, self).__init__()
        self.attn = pretrained_attn
        self.use_prenorm = use_prenorm

        # Borrow pretrained projection layers directly (no new parameters initialized)
        self.Wqkv = pretrained_attn.Wqkv
        self.Wo = pretrained_attn.Wo

        # Optional: Borrow pretrained LayerNorm / RMSNorm for Experiment B
        self.norm = pretrained_norm if use_prenorm else None

        # Borrow rotary embedding module from base ModernBERT attention
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        self.num_heads = getattr(pretrained_attn.config, "num_attention_heads", 12)
        self.head_dim = getattr(pretrained_attn, "head_dim", 64)
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        z: torch.Tensor,
        h_context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            z: [B, K, D] - Query latents (Position-Free, Unrotated)
            h_context: [B, L, D] - Context token representations
            context_mask: [B, L] - Non-padding mask (1 for valid token, 0 for pad)
            position_ids: [B, L] - Positional indices for context tokens (0 to L-1)
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
        if hasattr(self.Wqkv, 'weight'):
            q_weight, k_weight, v_weight = self.Wqkv.weight.chunk(3, dim=0)
            q_bias = k_bias = v_bias = None
            if getattr(self.Wqkv, 'bias', None) is not None:
                q_bias, k_bias, v_bias = self.Wqkv.bias.chunk(3, dim=0)

            q = F.linear(z_in, q_weight, q_bias)       # [B, K, D]
            k = F.linear(h_in, k_weight, k_bias)       # [B, L, D]
            v = F.linear(h_in, v_weight, v_bias)       # [B, L, D]
        else:
            # Fallback if Wqkv is already split or callable
            qkv = self.Wqkv(h_in)
            q = F.linear(z_in, self.Wqkv.weight[:D])
            k = qkv[..., D:2*D]
            v = qkv[..., 2*D:]

        # 2. Reshape into Multi-Head format [B, num_heads, SeqLen, head_dim]
        q = q.contiguous().view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, d_h]
        k = k.contiguous().view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]
        v = v.contiguous().view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]

        # 3. Apply Native ModernBERT RoPE Exclusively to Keys (K)
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(L, device=h_context.device).unsqueeze(0).expand(B, -1)
            try:
                # ModernBERT signature: rotary_emb(value_states, position_ids)
                cos, sin = self.rotary_emb(v, position_ids)
                if cos.dim() == 3:
                    cos = cos.unsqueeze(1)
                    sin = sin.unsqueeze(1)
                elif cos.dim() == 2:
                    cos = cos.unsqueeze(0).unsqueeze(1)
                    sin = sin.unsqueeze(0).unsqueeze(1)
                k = apply_rotary_pos_emb_single(k, cos, sin)
            except Exception:
                # Fallback if custom signature
                pass

        # 4. Global Asymmetric Attention Matrix (Full Non-PAD Visibility)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # [B, H, K, L]
        if context_mask is not None:
            # Mask out PAD tokens: shape [B, 1, 1, L]
            mask_4d = (1.0 - context_mask.unsqueeze(1).unsqueeze(2).to(scores.dtype)) * -10000.0
            scores = scores + mask_4d

        attn_weights = F.softmax(scores, dim=-1) # [B, H, K, L]
        attn_out = torch.matmul(attn_weights, v) # [B, H, K, d_h]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)

        # 5. Pretrained Wo Linear Projection with Residual Connection
        z_out = z + self.Wo(attn_out)
        
        # Average attention across heads for interpretability / inspection: [B, K, L]
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
    2. Backbone Pass: ModernBERT Layers 1 to 17 extract deep contextual features H_17.
    3. Trainable Query Probes: Z_0 = [q_no, q_implicit, q_explicit] (Position-Free).
    4. Pretrained Cross-Attention Stack (Layers 18 to 22):
       - Repurposes frozen/trainable pretrained projection matrices W_qkv and W_o.
       - Asymmetric RoPE: Unrotated Q + Rotated K.
       - Full Non-PAD visibility across context.
    5. Hierarchical Two-Head LogSigmoid Classifier.
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
        cross_layer_start: int = 17, # Zero-based: layer index 17 is Layer 18
        use_prenorm: bool = False,
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
        hidden_dim = hidden_dim or (d_model // 2)

        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm

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
        # LAYER 2: Pretrained Attention Weight Borrowing Stack (Layers 18 to 22)
        # ---------------------------------------------------------------------
        self.cross_layers = nn.ModuleList()
        
        # Locate backbone encoder layers (ModernBERT 'layers' or BERT 'layer')
        encoder_layers = None
        if hasattr(self.mmbert, 'encoder') and hasattr(self.mmbert.encoder, 'layers'):
            encoder_layers = self.mmbert.encoder.layers
        elif hasattr(self.mmbert, 'layers'):
            encoder_layers = self.mmbert.layers
        elif hasattr(self.mmbert, 'encoder') and hasattr(self.mmbert.encoder, 'layer'):
            encoder_layers = self.mmbert.encoder.layer

        self.has_borrowed_layers = False
        if encoder_layers is not None and len(encoder_layers) > cross_layer_start:
            target_layers = encoder_layers[cross_layer_start:]
            for layer in target_layers:
                attn_mod = getattr(layer, 'attn', getattr(layer, 'attention', None))
                norm_mod = getattr(layer, 'attn_norm', getattr(layer, 'input_layernorm', None))
                if attn_mod is not None and hasattr(attn_mod, 'Wqkv'):
                    cross_mod = PretrainedCrossAttentionLayer(
                        pretrained_attn=attn_mod,
                        pretrained_norm=norm_mod,
                        use_prenorm=use_prenorm
                    )
                    self.cross_layers.append(cross_mod)
            if len(self.cross_layers) > 0:
                self.has_borrowed_layers = True

        # Fallback multihead attention if backbone does not expose Wqkv directly
        if not self.has_borrowed_layers:
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

        can_inject_embeds = (
            hasattr(self.mmbert, 'get_input_embeddings')
            and callable(getattr(self.mmbert, 'get_input_embeddings'))
            and self.mmbert.get_input_embeddings() is not None
        )

        if can_inject_embeds:
            if not backbone_trainable:
                with torch.no_grad():
                    word_embeds = self.mmbert.get_input_embeddings()(input_ids)
            else:
                word_embeds = self.mmbert.get_input_embeddings()(input_ids)

            e_role = self.role_embeddings(role_ids)
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

        # 2. Expand Class Prototype Queries: Z_0 [B, total_queries, d_model]
        z = self.class_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        if z.dtype != h_seq.dtype:
            z = z.to(h_seq.dtype)

        # 3. Cross-Attention Processing (Borrowed Stack or Fallback)
        last_attn_weights = None
        if self.has_borrowed_layers and len(self.cross_layers) > 0:
            for cross_layer in self.cross_layers:
                z, last_attn_weights = cross_layer(
                    z=z,
                    h_context=h_seq,
                    context_mask=attention_mask
                )
        else:
            # Fallback standard cross-attention
            q_normed = self.norm_q(z)
            kv_normed = self.norm_kv(h_seq)
            key_padding_mask = (attention_mask == 0)

            if return_attention_map:
                z_attn, last_attn_weights = self.cross_attention(
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
                last_attn_weights = None
            z = (z + self.dropout_cross(z_attn)).contiguous()

        # 4. Optional Inter-Class Query Self-Interaction
        if self.use_query_interaction and self.total_queries > 1:
            z_normed = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_normed,
                key=z_normed,
                value=z_normed,
                need_weights=False
            )
            z = (z + self.dropout_self(z_self)).contiguous()

        # 5. Slot Aggregation (if num_slots_per_class > 1)
        if self.num_slots_per_class == 1:
            z_classes = z.contiguous().reshape(B, self.num_classes, self.d_model)
        else:
            z_classes = z.contiguous().reshape(B, self.num_classes, self.num_slots_per_class, self.d_model).mean(dim=2)

        # 6. Classification Head Readout
        out_logits = self.scoring_head(
            z_classes=z_classes,
            return_all_msd_logits=return_all_msd_logits
        )

        # 7. Task C Latent Bridge (Probability-Weighted Blend)
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
