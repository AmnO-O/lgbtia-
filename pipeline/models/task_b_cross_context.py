"""
Task B Dual-Stream Cross-Context Attention Model.

Architecture:
1. Shared mmBERT Backbone:
   - Branch 1 (TC Stream): [CLS] Title [SEP] Comment [SEP] -> H_TC [B, S_tc, d]
   - Branch 2 (D Stream):  [CLS] Description [SEP]         -> H_D  [B, S_d, d]

2. Selective Cross-Attention & Gated Residual Highway:
   - Q = H_comment (token representations of comment)
   - K, V = H_D (token representations of description)
   - Cross Attention Context C = MultiHeadAttention(Q=H_comment, K=H_D, V=H_D)
   - Element-wise Gate g = sigmoid(W_g [H_comment; C] + b_g) with negative bias initialization (b_g = -1.5)
   - Fused Comment Tokens: H'_comment = RMSNorm(H_comment + g * C)

3. Representation Pooling:
   - Masked Average Pooling over Title Tokens: H_title [B, d]
   - Masked Average Pooling over Fused Comment Tokens: H'_comment [B, d]
   - Joint Representation: Z = [H_title; H'_comment] [B, 2*d]

4. Classification Head:
   - Multi-Sample Dropout (MSD) for low-variance generalization across 3 classes (0: No, 1: Implicit, 2: Explicit).
"""

import math
from typing import Optional, Tuple, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .head import MultiSampleDropoutHead

ROLE_PAD = 0
ROLE_TITLE = 1
ROLE_COMMENT = 2
NUM_ROLES = 3
NUM_CLASSES = 3


class GatedCrossAttentionContextBlock(nn.Module):
    """
    Asymmetric Cross-Attention Block with Element-wise Gated Residual Highway.
    
    Extracts relevant background context from Description conditioned on Comment tokens,
    and fuses them using an element-wise learned gate with negative bias initialization.
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
        
        # Element-wise Gate: Linear(2*d -> d) with negative bias init
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
            h_comment: [B, S_tc, d_model] - representations containing comment tokens.
            h_desc: [B, S_d, d_model] - representations of description tokens.
            desc_key_padding_mask: [B, S_d] - True indicates key tokens that should be ignored in attention.
            return_gate_values: whether to return gate activations for analysis/logging.
        Returns:
            h_fused_comment: [B, S_tc, d_model] - Context-infused comment token representations.
            gate_values: [B, S_tc, d_model] or None.
        """
        # Cross-Attention: Q = h_comment, K = h_desc, V = h_desc
        # Note: key_padding_mask in nn.MultiheadAttention expects True for PAD tokens
        context, _ = self.cross_attn(
            query=h_comment,
            key=h_desc,
            value=h_desc,
            key_padding_mask=desc_key_padding_mask,
            need_weights=False
        ) # [B, S_tc, d_model]

        # Compute Element-wise Gate g in (0, 1)^d
        gate_input = torch.cat([h_comment, context], dim=-1) # [B, S_tc, 2*d_model]
        gate = torch.sigmoid(self.gate_proj(gate_input))      # [B, S_tc, d_model]

        # Gated Residual Highway: H' = Norm(H + g * Context)
        h_fused = self.norm(h_comment + self.dropout(gate * context))

        if return_gate_values:
            return h_fused, gate
        return h_fused, None


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
        dropout: float = 0.50,
        use_msd: bool = False,
        msd_dropout_rates: Optional[List[float]] = None,
        gate_bias_init: float = -1.50,
        use_rmsnorm: bool = True
    ):
        super(TaskBCrossContextAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_msd = use_msd
        
        # 1. Structural Role Embedding for Title vs Comment tokens
        self.role_embedding = nn.Embedding(NUM_ROLES, d_model, padding_idx=ROLE_PAD)
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.02)

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

        # 4. Multi-Sample Dropout (MSD) or High Single-Dropout Classification Head
        self.msd_rates = msd_dropout_rates or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.classifier = MultiSampleDropoutHead(
            in_dim=2 * d_model,
            hidden_dim=d_model,
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
        Computes masked average pooling over a sequence.
        Args:
            tokens: [B, S, d_model]
            mask: [B, S] (1 for valid tokens, 0 for pad/irrelevant)
        Returns:
            pooled: [B, d_model]
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
            return_all_msd_logits: if True, returns List of logits from each MSD branch for multi-sample loss
            return_gate_values: if True, returns average gate activation
        """
        # ==========================================================
        # 1. STREAM 1: Title + Comment with Role Embeddings
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
        # Create boolean key padding mask for PyTorch MultiheadAttention (True = Pad token to ignore)
        desc_pad_mask = (desc_attention_mask == 0) # [B, S_d]

        # Extract only comment tokens for cross-attention
        comment_mask = (tc_role_ids == ROLE_COMMENT) # [B, S_tc]
        title_mask = (tc_role_ids == ROLE_TITLE)     # [B, S_tc]

        # Zero out non-comment token representations to ensure clean Query projection
        h_comment_pure = h_tc * comment_mask.unsqueeze(-1).float()

        # Run Gated Cross-Attention
        h_fused_comment, gate = self.cross_context(
            h_comment=h_comment_pure,
            h_desc=h_desc,
            desc_key_padding_mask=desc_pad_mask,
            return_gate_values=return_gate_values
        )

        # ==========================================================
        # 4. DUAL TOKEN POOLING & CONCATENATION
        # ==========================================================
        # Pool title representation
        h_title_pooled = self.masked_mean_pool(h_tc, title_mask) # [B, d]

        # Pool context-infused comment representation
        h_comment_pooled = self.masked_mean_pool(h_fused_comment, comment_mask) # [B, d]

        # Joint Multi-Aspect Representation [B, 2*d]
        z_fusion = torch.cat([h_title_pooled, h_comment_pooled], dim=-1)
        z_fusion = self.fusion_norm(z_fusion)

        # ==========================================================
        # 5. MULTI-SAMPLE DROPOUT CLASSIFICATION HEAD
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
