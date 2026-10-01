"""
Task B Dual-Stream Cross-Context Attention Model.

Architecture: Dual-Stream Asymmetric Context Retrieval with Gated Comment Fusion
- Stream 1: [CLS] Title [SEP] Comment [SEP] -> mmBERT -> Post-Backbone Role Embeddings + Normalization
- Stream 2: [CLS] Description [SEP] -> mmBERT (shared weights) -> Normalization
- Asymmetric Context Cross-Attention:
    Q = H_Comment (Clean comment tokens)
    K = H_Desc, V = H_Desc
- Element-wise/Scalar Gated Residual Highway:
    g = sigmoid(W_g [H_Comment; Context] + b_g) with b_g_init = -1.50 (conservative ~0.18)
    H_Comment' = RMSNorm(H_Comment + g * Context)
- Dual Token Pooling:
    h_Title = MaskedMeanPool(H_Title)
    h_Comment = MaskedMeanPool(H_Comment')
    z = RMSNorm([h_Title; h_Comment'])  [B, 2*d_model]
- Classification Head: Multi-Sample Dropout (or Single Dropout) with hidden_dim=384 -> num_classes.
"""

from typing import List, Tuple, Dict, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES
from .task_b_cross_data import ROLE_PAD, ROLE_TITLE, ROLE_COMMENT


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


class GatedCrossAttentionContextBlock(nn.Module):
    """
    Asymmetric Cross-Attention Block with Element-wise Gated Residual Highway.
    
    Q = H_Comment, K = H_Desc, V = H_Desc
    H_Comment' = RMSNorm(H_Comment + Dropout(g * Context))
    """
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.10,
        gate_bias_init: float = -1.50,
        use_rmsnorm: bool = True
    ):
        super(GatedCrossAttentionContextBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-Head Cross Attention (Q: Comment, K, V: Description)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Element-wise Gate: Linear(2*d -> d) with negative bias init (-1.5 -> sigmoid ~ 0.18)
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.01)
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
            h_comment: [B, S_c, d_model] - Comment token representations (Query)
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


class MultiSampleDropoutHead(nn.Module):
    """
    Multi-Sample Dropout (MSD) or Single High-Dropout Classification Head.
    Architecture: [2*d_model] -> Linear(2*d_model -> hidden_dim) -> Mish/GELU -> Dropout -> Linear(hidden_dim -> num_classes)
    """
    def __init__(
        self,
        in_dim: int = 1536,
        hidden_dim: int = 384,
        num_classes: int = 3,
        use_msd: bool = True,
        msd_dropout_rates: Optional[List[float]] = None,
        dropout: float = 0.50,
        use_rmsnorm: bool = True
    ):
        super(MultiSampleDropoutHead, self).__init__()
        self.use_msd = use_msd
        self.msd_rates = msd_dropout_rates or [0.1, 0.2, 0.3, 0.4, 0.5]
        
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm = NormClass(in_dim)
        
        self.dense = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        
        if self.use_msd:
            self.dropouts = nn.ModuleList([nn.Dropout(p) for p in self.msd_rates])
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_all_msd_logits: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        h = self.norm(x)
        h = self.act(self.dense(h))
        
        if self.use_msd:
            logits_list = [self.classifier(drop(h)) for drop in self.dropouts]
            if return_all_msd_logits:
                return logits_list
            return torch.mean(torch.stack(logits_list, dim=0), dim=0)
        else:
            out = self.classifier(self.dropout(h))
            if return_all_msd_logits:
                return [out]
            return out


class TaskBCrossContextAttentionModel(nn.Module):
    """
    Task B Dual-Stream Cross-Context Attention Classifier for Hate Speech Detection.
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
        use_rmsnorm: bool = True
    ):
        super(TaskBCrossContextAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_msd = use_msd
        
        # 1. Structural Post-Backbone Role Embedding for Title vs Comment tokens
        NUM_ROLES = 3 # 0: Pad, 1: Title, 2: Comment
        self.role_embedding = nn.Embedding(NUM_ROLES, d_model, padding_idx=ROLE_PAD)
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.02)
        # Explicit zero initialization for padding_idx row (Review Fix #1)
        with torch.no_grad():
            self.role_embedding.weight[ROLE_PAD].zero_()

        # 2. Asymmetric Gated Cross-Attention Block
        self.cross_context = GatedCrossAttentionContextBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout=min(dropout, 0.30),
            gate_bias_init=gate_bias_init,
            use_rmsnorm=use_rmsnorm
        )

        # 3. Post-backbone Normalizations
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.tc_norm = NormClass(d_model)
        self.desc_norm = NormClass(d_model)
        self.fusion_norm = NormClass(2 * d_model)

        # 4. Multi-Sample Dropout (MSD) or High Single-Dropout Classification Head (Review Fix #4: hidden_dim=384)
        self.msd_rates = msd_dropout_rates or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.classifier = MultiSampleDropoutHead(
            in_dim=2 * d_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_msd=use_msd,
            msd_dropout_rates=self.msd_rates,
            dropout=dropout,
            use_rmsnorm=use_rmsnorm
        )

    def extract_backbone_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Helper to run the backbone and return sequence tensor [B, S, d_model]."""
        # Fast path: If backbone parameters don't require grad (e.g. Phase 1), execute with no_grad
        is_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        if not is_trainable:
            with torch.no_grad():
                outputs = self.mmbert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.mmbert(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        raise ValueError(f"Unsupported backbone output type: {type(outputs)}")

    @staticmethod
    def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Pools tokens using a boolean or binary mask [B, S].
        Returns [B, d_model].
        """
        expanded_mask = mask.unsqueeze(-1).float() # [B, S, 1]
        sum_tokens = torch.sum(tokens * expanded_mask, dim=1) # [B, d_model]
        sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-6) # [B, 1]
        return sum_tokens / sum_mask

    def forward(
        self,
        tc_input_ids: torch.Tensor,
        tc_attention_mask: torch.Tensor,
        tc_role_ids: torch.Tensor,
        desc_input_ids: torch.Tensor,
        desc_attention_mask: torch.Tensor,
        return_all_msd_logits: bool = False,
        return_gate_values: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]]:
        """
        Forward pass executing the Dual-Stream Asymmetric Cross-Context Architecture.
        
        Args:
            tc_input_ids: [B, S_tc] - Title + Comment token IDs
            tc_attention_mask: [B, S_tc] - Title + Comment attention mask
            tc_role_ids: [B, S_tc] - 1 for Title, 2 for Comment, 0 for Pad
            desc_input_ids: [B, S_d] - Description token IDs
            desc_attention_mask: [B, S_d] - Description attention mask
            return_all_msd_logits: if True, returns List of logits from each MSD branch
            return_gate_values: if True, returns average gate activation
        """
        # ==========================================================
        # 1. STREAM 1: Title + Comment with Post-Backbone Role Embeddings
        # ==========================================================
        h_tc = self.extract_backbone_features(tc_input_ids, tc_attention_mask) # [B, S_tc, d]
        role_emb = self.role_embedding(tc_role_ids) # [B, S_tc, d]
        h_tc = self.tc_norm(h_tc + role_emb)

        # ==========================================================
        # 2. STREAM 2: Description Context
        # ==========================================================
        h_desc = self.extract_backbone_features(desc_input_ids, desc_attention_mask) # [B, S_d, d]
        h_desc = self.desc_norm(h_desc)

        # ==========================================================
        # 3. ASYMMETRIC CROSS-ATTENTION & GATED FUSION
        # ==========================================================
        # Boolean key padding mask for PyTorch MultiheadAttention (True = Pad token to ignore)
        # Safeguard: ensure at least one token is unmasked to prevent NaN in MultiheadAttention
        desc_pad_mask = (desc_attention_mask == 0) # [B, S_d]
        if desc_pad_mask.all(dim=-1).any():
            desc_pad_mask[:, 0] = False # Unmask [CLS] as safety fallback

        comment_mask = (tc_role_ids == ROLE_COMMENT) # [B, S_tc]
        title_mask = (tc_role_ids == ROLE_TITLE)     # [B, S_tc]

        # Pass h_tc through cross_context:
        # Cross-attention computes context retrieval for the sequence.
        h_cross_infused, gate = self.cross_context(
            h_comment=h_tc,
            h_desc=h_desc,
            desc_key_padding_mask=desc_pad_mask,
            return_gate_values=return_gate_values
        )

        # Selective Gated Comment Fusion:
        # Title tokens remain pure unmodified h_tc (no description noise)
        # Comment tokens receive context-infused representations
        comment_mask_expanded = comment_mask.unsqueeze(-1).float() # [B, S_tc, 1]
        h_fused_tc = (1.0 - comment_mask_expanded) * h_tc + comment_mask_expanded * h_cross_infused

        # ==========================================================
        # 4. DUAL TOKEN POOLING & CONCATENATION
        # ==========================================================
        # 1. Pure Title representation pooled only from Title tokens: h_T = MeanPool(H_T)
        h_title_pooled = self.masked_mean_pool(h_tc, title_mask) # [B, d]

        # 2. Context-infused Comment representation pooled only from Comment tokens: h_C = MeanPool(H_C')
        h_comment_pooled = self.masked_mean_pool(h_fused_tc, comment_mask) # [B, d]

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
            return out_logits, gate
        return out_logits
