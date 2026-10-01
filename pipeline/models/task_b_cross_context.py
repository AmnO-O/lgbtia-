"""
Task B Dual-Stream Cross-Context Attention Model.

Architecture: Dual-Stream Asymmetric Context Retrieval with Gated Comment Fusion
- Stream 1 (True Pre-Backbone Role-Aware Input):
    E_i = E_token,i + E_position,i + E_role,i
    X_TC -> mmBERT (All layers are role-aware) -> H_TC
- Stream 2 (Description Context Memory):
    X_Desc -> mmBERT (shared weights) -> H_Desc
- Pure Comment Query Extraction:
    H_Comment = Extract Comment Tokens from H_TC [B, S_c, d]
- Asymmetric Context Cross-Attention:
    Q = H_Comment (Clean comment tokens ONLY, no Title / Pad tokens)
    K = H_Desc, V = H_Desc
- Selective Gated Residual Highway:
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

from ..config import NUM_CLASSES
from ..task_b_cross_data import ROLE_PAD, ROLE_TITLE, ROLE_COMMENT


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


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
            h_comment: [B, S_c, d_model] - Pure Comment token representations (Query)
            h_desc: [B, S_d, d_model] - Description token representations (Key, Value)
            desc_key_padding_mask: [B, S_d] - True for Pad tokens to ignore in attention
            return_gate_values: bool - Whether to return gate activations
        Returns:
            h_fused: [B, S_c, d_model]
            gate: [B, S_c, d_model] or None
        """
        # Cross-Attention: Q = h_comment (pure comment query), K = h_desc, V = h_desc
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
    Architecture: [2*d_model] -> Linear(2*d_model -> hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim -> num_classes)
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

        # 4. Multi-Sample Dropout (MSD) Classification Head (hidden_dim=384)
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
        attention_mask: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Runs the transformer backbone.
        If role_ids is provided, injects role embeddings at the embedding layer (inputs_embeds)
        so that all transformer layers are natively role-aware:
            E_total = E_token + E_role
        """
        is_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        
        # Check if backbone supports inputs_embeds (HuggingFace transformers: BERT, RoBERTa, ModernBERT, etc.)
        can_inject_embeds = (
            role_ids is not None
            and hasattr(self.mmbert, 'get_input_embeddings')
            and callable(getattr(self.mmbert, 'get_input_embeddings'))
            and self.mmbert.get_input_embeddings() is not None
        )

        if can_inject_embeds:
            # Pre-Backbone Injection: E_total = E_token + E_role
            word_embeds = self.mmbert.get_input_embeddings()(input_ids) # [B, S, d]
            role_embeds = self.role_embedding(role_ids)                  # [B, S, d]
            inputs_embeds = word_embeds + role_embeds

            if not is_trainable:
                with torch.no_grad():
                    outputs = self.mmbert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            else:
                outputs = self.mmbert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            # Standard input_ids path
            if not is_trainable:
                with torch.no_grad():
                    outputs = self.mmbert(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = self.mmbert(input_ids=input_ids, attention_mask=attention_mask)

            # Fallback for dummy models/backbones without get_input_embeddings: add post-hoc if role_ids given
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
    def extract_pure_comment_tensor(
        h_tc: torch.Tensor,
        comment_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts only comment tokens into a compact, padded tensor [B, max_c_len, d]
        along with its corresponding comment_key_padding_mask [B, max_c_len].
        
        This guarantees:
        1. Q in Cross-Attention strictly contains ONLY comment tokens.
        2. Title and special tokens are never fed as Queries into the Cross-Attention layer.
        """
        batch_size, seq_len, d_model = h_tc.shape
        device = h_tc.device
        
        # Calculate lengths of comment tokens per sequence in batch
        comment_lens = comment_mask.sum(dim=-1) # [B]
        max_c_len = max(int(comment_lens.max().item()), 1)
        
        # Pre-allocate dense comment tensor and mask
        h_comment_dense = torch.zeros(batch_size, max_c_len, d_model, device=device, dtype=h_tc.dtype)
        comment_pad_mask = torch.ones(batch_size, max_c_len, device=device, dtype=torch.bool) # True = PAD
        
        for b in range(batch_size):
            c_idx = torch.where(comment_mask[b])[0]
            if len(c_idx) > 0:
                h_comment_dense[b, :len(c_idx)] = h_tc[b, c_idx]
                comment_pad_mask[b, :len(c_idx)] = False
            else:
                # Edge case: fallback to first token if no comment tokens found
                h_comment_dense[b, 0] = h_tc[b, 0]
                comment_pad_mask[b, 0] = False
                
        return h_comment_dense, comment_pad_mask

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
        # 1. STREAM 1: Title + Comment with True Pre-Backbone Role Embeddings
        # ==========================================================
        # Injects role embeddings directly at the input embedding layer before all transformer layers
        h_tc = self.extract_backbone_features(
            input_ids=tc_input_ids,
            attention_mask=tc_attention_mask,
            role_ids=tc_role_ids
        ) # [B, S_tc, d]
        h_tc = self.tc_norm(h_tc)

        # ==========================================================
        # 2. STREAM 2: Description Context
        # ==========================================================
        h_desc = self.extract_backbone_features(
            input_ids=desc_input_ids,
            attention_mask=desc_attention_mask,
            role_ids=None
        ) # [B, S_d, d]
        h_desc = self.desc_norm(h_desc)

        # ==========================================================
        # 3. PURE COMMENT EXTRACTION & ASYMMETRIC CROSS-ATTENTION (Q = H_Comment)
        # ==========================================================
        # Boolean key padding mask for PyTorch MultiheadAttention (True = Pad token to ignore)
        desc_pad_mask = (desc_attention_mask == 0) # [B, S_d]
        if desc_pad_mask.all(dim=-1).any():
            desc_pad_mask[:, 0] = False # Unmask [CLS] as safety fallback

        comment_mask = (tc_role_ids == ROLE_COMMENT) # [B, S_tc]
        title_mask = (tc_role_ids == ROLE_TITLE)     # [B, S_tc]

        # Extract PURE Comment Tokens: H_C in [B, max_c_len, d]
        # Title tokens are completely excluded from Query projection
        h_comment_pure, c_pad_mask = self.extract_pure_comment_tensor(h_tc, comment_mask)

        # Cross-Attention: Q = H_Comment ONLY, K = H_Desc, V = H_Desc
        h_fused_comment, gate = self.cross_context(
            h_comment=h_comment_pure,
            h_desc=h_desc,
            desc_key_padding_mask=desc_pad_mask,
            return_gate_values=return_gate_values
        )

        # ==========================================================
        # 4. DUAL TOKEN POOLING & CONCATENATION
        # ==========================================================
        # 1. Pure Title representation pooled only from Title tokens: h_T = MeanPool(H_T)
        h_title_pooled = self.masked_mean_pool(h_tc, title_mask) # [B, d]

        # 2. Context-infused Comment representation pooled only from fused Comment tokens: h_C = MeanPool(H_C')
        comment_valid_mask = ~c_pad_mask # [B, max_c_len]
        h_comment_pooled = self.masked_mean_pool(h_fused_comment, comment_valid_mask) # [B, d]

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
