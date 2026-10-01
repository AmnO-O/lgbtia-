"""
Task B Specialized Cross-Context Attention Classifier.

Key Structural Advancements:
1. Structural Pre-Backbone Role Embeddings (sans_pos with RoPE):
   - mmBERT (ModernBERT architecture, 22 layers, d=768, 12 heads, 307M params) uses `position_embedding_type="sans_pos"`
     where positional information is injected via Rotary Position Embeddings (RoPE) inside attention heads.
   - Therefore, input token representations are constructed as:
         E_{input, i} = E_{token, i} + E_{role, i}
     (where role ID: 0 = Pad, 1 = Title, 2 = Comment), followed by native mmBERT embedding normalization and RoPE.
   - Injected directly at the input embedding level (`inputs_embeds`) so the entire mmBERT encoder is natively role-aware.
   - P0 Gradient Flow Guarantee (Review #2): If mmBERT parameters are frozen during Phase 1
     linear probing but role_embedding is trainable, mmBERT forward pass runs WITH autograd enabled
     (no torch.no_grad()) so that gradients correctly propagate backwards to role_embedding (dH/dE_role != 0).
2. Asymmetric Cross-Context Attention (Q = H_Comment ONLY):
   - Pure Comment tokens H_C act as Query. Title tokens are strictly excluded from Q, eliminating spurious title-description cross-attention.
   - Vectorized Zero-Sync GPU Execution (Review #5): Query projection is computed directly on the full sequence tensor without host-device synchronization, and masked mean pooling isolates H_C' in O(1) vectorized GPU kernels.
   - Context = MultiHeadAttention(Q=H_C, K=H_D, V=H_D).
3. Element-wise Gated Residual Highway:
   - g = sigmoid(W_g [H_C ; Context] + b_g) in (0, 1)^d.
   - H_C' = RMSNorm(H_C + Dropout(g * Context)).
4. Dual-Stream Pooling:
   - h_T = MeanPool(H_T) (Pure Title context).
   - h_C = MeanPool(H_C') (Context-infused Comment representation).
   - Final representation: z = RMSNorm([h_T ; h_C']) in R^{2d}.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_CLASSES, ROLE_PAD, ROLE_TITLE, ROLE_COMMENT


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for improved numerical stability and speed."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.scale


class MultiSampleDropoutClassifier(nn.Module):
    """
    Multi-Sample Dropout (MSD) Classification Head.
    Passes features through multiple parallel dropout masks with distinct rates
    and averages their linear projections for stronger regularization.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int = NUM_CLASSES,
        dropout_rates: Optional[List[float]] = None,
        use_rmsnorm: bool = True
    ):
        super(MultiSampleDropoutClassifier, self).__init__()
        self.dropout_rates = dropout_rates or [0.10, 0.15, 0.20, 0.25, 0.30]
        
        self.pre_proj = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm = NormClass(hidden_dim)
        
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in self.dropout_rates])
        self.out_proj = nn.Linear(hidden_dim, num_classes)

        # Initialize projection layers
        nn.init.xavier_uniform_(self.pre_proj.weight)
        nn.init.zeros_(self.pre_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, return_all_msd_logits: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        h = self.norm(self.act(self.pre_proj(x)))
        if self.training:
            logits_list = [self.out_proj(drop(h)) for drop in self.dropouts]
            if return_all_msd_logits:
                return logits_list
            return torch.mean(torch.stack(logits_list, dim=0), dim=0)
        return self.out_proj(h)


class GatedCrossAttentionContextBlock(nn.Module):
    """
    Asymmetric Cross-Attention Block with Element-wise Gated Residual Highway.
    
    Q = H_Comment (Comment-only tokens), K = H_Desc, V = H_Desc
    H_Comment' = RMSNorm(H_Comment + Dropout(g * Context))
    """
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.10,
        gate_bias_init: float = -1.50,
        gate_gain_init: float = 0.05,
        use_rmsnorm: bool = True
    ):
        super(GatedCrossAttentionContextBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-Head Cross Attention (Q: Pure Comment, K, V: Description)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Element-wise Gate: Linear(2*d -> d) with negative bias init (-1.5 -> sigmoid ~ 0.18)
        # gain=0.05 allows dynamic input-dependence from the start while preserving conservative prior
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=gate_gain_init)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)
        
        # Normalization layer
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm = NormClass(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_comment: torch.Tensor,
        h_desc: torch.Tensor,
        desc_key_padding_mask: Optional[torch.Tensor] = None,
        return_gate_values: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h_comment: [B, S_c, d_model] - Comment query representations
            h_desc: [B, S_d, d_model] - Description token representations (Key, Value)
            desc_key_padding_mask: [B, S_d] - True for Pad tokens to ignore in attention
            return_gate_values: bool - Whether to return gate activations
        Returns:
            h_fused: [B, S_c, d_model]
            gate: [B, S_c, d_model] or None
        """
        # Cross-Attention: Q = h_comment, K = h_desc, V = h_desc
        context, _ = self.cross_attn(
            query=h_comment,
            key=h_desc,
            value=h_desc,
            key_padding_mask=desc_key_padding_mask,
            need_weights=False
        ) # [B, S_c, d_model]

        # Compute Gate g in (0, 1)^d
        gate_input = torch.cat([h_comment, context], dim=-1) # [B, S_c, 2*d_model]
        gate = torch.sigmoid(self.gate_proj(gate_input))      # [B, S_c, d_model]

        # Gated Residual Highway: H' = Norm(H + Dropout(g * Context))
        h_fused = self.norm(h_comment + self.dropout(gate * context))

        if return_gate_values:
            return h_fused, gate
        return h_fused, None


class TaskBCrossContextAttentionModel(nn.Module):
    """
    Task B Dual-Stream Cross-Context Attention Classifier with Pre-Backbone Role Embeddings
    and Pure Comment Query Cross-Attention.
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_classes: int = NUM_CLASSES,
        num_heads: int = 8,
        hidden_dim: int = 384,
        dropout: float = 0.30,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        gate_bias_init: float = -1.50,
        gate_gain_init: float = 0.05,
        use_rmsnorm: bool = True
    ):
        super(TaskBCrossContextAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_msd = use_msd
        
        # 1. Structural Pre-Backbone Role Embedding for Title vs Comment tokens (0: Pad, 1: Title, 2: Comment)
        NUM_ROLES = 3
        self.role_embedding = nn.Embedding(NUM_ROLES, d_model, padding_idx=ROLE_PAD)
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.role_embedding.weight[ROLE_PAD].zero_()

        # 2. Asymmetric Gated Cross-Attention Block
        self.cross_context = GatedCrossAttentionContextBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout=min(dropout, 0.30),
            gate_bias_init=gate_bias_init,
            gate_gain_init=gate_gain_init,
            use_rmsnorm=use_rmsnorm
        )

        # 3. Post-backbone Normalizations
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.tc_norm = NormClass(d_model)
        self.desc_norm = NormClass(d_model)
        self.fusion_norm = NormClass(2 * d_model)

        # 4. Classification Head (Dual Stream: Title (768) + Context-infused Comment (768) -> 1536)
        if use_msd:
            self.classifier = MultiSampleDropoutClassifier(
                in_features=2 * d_model,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout_rates=msd_dropout_rates,
                use_rmsnorm=use_rmsnorm
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(2 * d_model, hidden_dim),
                nn.GELU(),
                NormClass(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )

    def extract_backbone_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Runs the transformer backbone.
        If role_ids is provided, injects role embeddings at the embedding layer (inputs_embeds)
        so that all transformer layers are natively role-aware:
            E_{input, i} = E_{token, i} + E_{role, i}

        P0 Gradient Flow Guarantee (Review #2):
        - When role_ids is provided and role_embedding requires gradients, the forward pass
          through mmBERT MUST execute with autograd enabled so that dH / dE_role can be computed,
          even if the backbone parameters themselves are frozen!
        - When no gradient is needed by either mmBERT parameters OR role_embedding (e.g. inference
          or frozen stream without role embeddings), torch.no_grad() is safely applied to save memory.
        """
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        role_trainable = (role_ids is not None) and self.role_embedding.weight.requires_grad
        needs_grad = torch.is_grad_enabled() and (backbone_trainable or role_trainable)
        
        # Check if backbone supports inputs_embeds (HuggingFace transformers: BERT, RoBERTa, ModernBERT, etc.)
        can_inject_embeds = (
            role_ids is not None
            and hasattr(self.mmbert, 'get_input_embeddings')
            and callable(getattr(self.mmbert, 'get_input_embeddings'))
            and self.mmbert.get_input_embeddings() is not None
        )

        if can_inject_embeds:
            # Token embeddings from backbone (word_embeds does not need grad if backbone is frozen)
            if not backbone_trainable:
                with torch.no_grad():
                    word_embeds = self.mmbert.get_input_embeddings()(input_ids) # [B, S, d]
            else:
                word_embeds = self.mmbert.get_input_embeddings()(input_ids)

            # Trainable Role Embeddings
            role_embeds = self.role_embedding(role_ids) # [B, S, d] (requires_grad = True during Phase 1)
            inputs_embeds = word_embeds + role_embeds

            # Forward pass through mmBERT:
            # If needs_grad is True (due to role_embeds being trainable),
            # DO NOT wrap in torch.no_grad(), allowing dH / dE_role gradient backprop!
            if needs_grad:
                outputs = self.mmbert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            else:
                with torch.no_grad():
                    outputs = self.mmbert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            # Standard input_ids path (e.g. Description stream)
            if needs_grad:
                outputs = self.mmbert(input_ids=input_ids, attention_mask=attention_mask)
            else:
                with torch.no_grad():
                    outputs = self.mmbert(input_ids=input_ids, attention_mask=attention_mask)

            # Fallback for models without get_input_embeddings
            if role_ids is not None and not can_inject_embeds:
                if hasattr(outputs, 'last_hidden_state'):
                    raw_h = outputs.last_hidden_state
                elif isinstance(outputs, (tuple, list)):
                    raw_h = outputs[0]
                elif isinstance(outputs, torch.Tensor):
                    raw_h = outputs
                else:
                    raise ValueError(f"Unsupported backbone output type: {type(outputs)}")
                return raw_h + self.role_embedding(role_ids)

        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        raise ValueError(f"Unsupported backbone output type: {type(outputs)}")

    @staticmethod
    def masked_mean_pool(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Vectorized mean pooling across the sequence dimension only over valid tokens.
        Executed entirely in PyTorch GPU memory with zero CPU-GPU sync.
        Args:
            tensor: [B, S, d]
            mask: [B, S] (1/True for valid tokens, 0/False for excluded/pad)
        Returns:
            [B, d]
        """
        if mask.dtype == torch.bool:
            mask_float = mask.float().unsqueeze(-1)
        else:
            mask_float = mask.unsqueeze(-1).float()
            
        sum_embeddings = torch.sum(tensor * mask_float, dim=1) # [B, d]
        sum_mask = torch.clamp(mask_float.sum(dim=1), min=1e-9) # [B, 1]
        return sum_embeddings / sum_mask

    def forward(
        self,
        tc_input_ids: torch.Tensor,
        tc_attention_mask: torch.Tensor,
        tc_role_ids: torch.Tensor,
        desc_input_ids: torch.Tensor,
        desc_attention_mask: torch.Tensor,
        return_all_msd_logits: bool = False,
        return_gate_values: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Dual-Stream Forward Pass (Zero Host-Device Sync):
        Stream 1: Title + Comment with Pre-Backbone Role Embeddings (Inputs Embeds)
        Stream 2: Description alone (Pretrained Language Stream)
        Fusion: Fully Vectorized Gated Cross-Attention (Q = H_TC, K, V = H_Desc) with Comment Token Masking
        """
        # ==========================================================
        # 1. STREAM 1: TITLE + COMMENT WITH ROLE EMBEDDING INJECTION
        # ==========================================================
        h_tc = self.extract_backbone_features(
            input_ids=tc_input_ids,
            attention_mask=tc_attention_mask,
            role_ids=tc_role_ids
        ) # [B, S_tc, d]
        h_tc = self.tc_norm(h_tc)

        # ==========================================================
        # 2. STREAM 2: DESCRIPTION STREAM
        # ==========================================================
        h_desc = self.extract_backbone_features(
            input_ids=desc_input_ids,
            attention_mask=desc_attention_mask,
            role_ids=None
        ) # [B, S_d, d]
        h_desc = self.desc_norm(h_desc)

        # ==========================================================
        # 3. VECTORIZED ASYMMETRIC CROSS-ATTENTION & GATING (ZERO GPU->CPU SYNC)
        # ==========================================================
        # Boolean key padding mask for PyTorch MultiheadAttention (True = Pad token to ignore)
        desc_pad_mask = (desc_attention_mask == 0) # [B, S_d]
        if desc_pad_mask.all(dim=-1).any():
            desc_pad_mask[:, 0] = False # Safety unmask [CLS]

        comment_mask = (tc_role_ids == ROLE_COMMENT) # [B, S_tc] (Boolean tensor)
        title_mask = (tc_role_ids == ROLE_TITLE)     # [B, S_tc] (Boolean tensor)

        # Vectorized Cross-Attention: Q = h_tc, K = h_desc, V = h_desc
        # Runs fully in parallel on GPU without Python per-sample loops or .item() syncs
        h_fused_all, gate_all = self.cross_context(
            h_comment=h_tc,
            h_desc=h_desc,
            desc_key_padding_mask=desc_pad_mask,
            return_gate_values=return_gate_values
        ) # [B, S_tc, d]

        # ==========================================================
        # 4. DUAL TOKEN POOLING & CONCATENATION
        # ==========================================================
        # 1. Pure Title representation: pooled only from Title tokens in h_tc (no description cross-talk)
        h_title_pooled = self.masked_mean_pool(h_tc, title_mask) # [B, d]

        # 2. Context-infused Comment representation: pooled only from Comment tokens in h_fused_all
        h_comment_pooled = self.masked_mean_pool(h_fused_all, comment_mask) # [B, d]

        # Joint Multi-Aspect Representation: z = RMSNorm([h_T; h_C']) [B, 2*d]
        z_fusion = torch.cat([h_title_pooled, h_comment_pooled], dim=-1)
        z_fusion = self.fusion_norm(z_fusion)

        # ==========================================================
        # 5. CLASSIFICATION HEAD (1536 -> 384 -> 3)
        # ==========================================================
        logits_output = self.classifier(z_fusion, return_all_msd_logits=return_all_msd_logits)

        if return_all_msd_logits:
            out_logits = logits_output if isinstance(logits_output, list) else [logits_output]
        else:
            if isinstance(logits_output, list):
                out_logits = torch.mean(torch.stack(logits_output, dim=0), dim=0)
            else:
                out_logits = logits_output

        if return_gate_values:
            # If gate values are requested for logging, mask only comment tokens
            gate_masked = gate_all * comment_mask.unsqueeze(-1).float()
            return out_logits, gate_masked
        return out_logits
