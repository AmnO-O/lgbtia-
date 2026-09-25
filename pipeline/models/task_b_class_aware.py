import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union

from .task_b_questions import (
    QueryProbe,
    DEFAULT_TASK_B_PROBES,
    get_default_probes,
    get_queries_per_class,
    CLASS_NO_HATE,
    CLASS_IMPLICIT_HATE,
    CLASS_EXPLICIT_HATE,
    NUM_CLASSES,
)

# Role IDs for explicit role injection
# 0: Pad / Special tokens
# 1: Title (<T>...</T>)
# 2: Description (<D>...</D>)
# 3: Comment (<C>...</C>)
ROLE_PAD = 0
ROLE_TITLE = 1
ROLE_DESC = 2
ROLE_COMMENT = 3
NUM_ROLES = 4


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) - Zhang & Sennrich (2019).
    Scales inputs by the root mean square of activation values, providing superior
    gradient stabilization, faster CUDA execution, and strict invariance to activation scaling.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class TaskBClassAwareAttentionModel(nn.Module):
    """
    Enhanced Multi-Query Class-Aware Cross-Attention (MHCA) Architecture for Task B.
    
    Architectural Highlights:
      1. Pre-LN / Pre-RMSNorm Residual Connections:
         Normalizes inputs *before* Attention and Interaction sub-layers (x + Sublayer(Norm(x))).
         This provides an unobstructed residual highway, preventing gradient vanishing or exploding
         during backbone fine-tuning.
      2. RMSNorm vs LayerNorm:
         Uses RMSNorm by default for tighter gradient variance bounds and faster GPU computation.
      3. PyTorch SDPA / FlashAttention Enablement:
         `need_weights=return_attention_map` allows PyTorch to execute fast FlashAttention kernels.
      4. Dynamic Multi-Query Probing with Aspect Pooling:
         Probes fine-grained semantic angles (sarcasm, dogwhistle, slurs, support) and aggregates
         them into the 3 target classes using learnable attention pooling.
      5. Task C Gradient Isolation (`detach_bridge=True`).
    """
    def __init__(
        self,
        mmbert_model: nn.Module,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.20,
        use_query_interaction: bool = True,
        hidden_dim: Optional[int] = None,
        probes: Optional[List[QueryProbe]] = None,
        num_queries_per_class: Optional[int] = None,
        pooling_mode: str = "attention", # 'attention', 'mean', or 'max'
        use_rmsnorm: bool = True,        # Toggle RMSNorm vs standard LayerNorm
    ):
        super(TaskBClassAwareAttentionModel, self).__init__()
        self.mmbert = mmbert_model
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_query_interaction = use_query_interaction
        self.pooling_mode = pooling_mode
        self.use_rmsnorm = use_rmsnorm
        hidden_dim = hidden_dim or d_model // 2

        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm

        # ---------------------------------------------------------------------
        # 1. Setup Query Probes & Mapping
        # ---------------------------------------------------------------------
        if probes is not None:
            self.probes = list(probes)
        elif num_queries_per_class is not None:
            self.probes = []
            for c_idx in range(NUM_CLASSES):
                for q_i in range(num_queries_per_class):
                    self.probes.append(QueryProbe(
                        id=f"class_{c_idx}_slot_{q_i}",
                        class_idx=c_idx,
                        aspect=f"slot_{q_i}",
                        question_vi="",
                        question_en=""
                    ))
        else:
            self.probes = get_default_probes()

        self.num_queries = len(self.probes)
        
        self.class_to_query_indices: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        for idx, probe in enumerate(self.probes):
            if probe.class_idx in self.class_to_query_indices:
                self.class_to_query_indices[probe.class_idx].append(idx)

        for c in range(NUM_CLASSES):
            if len(self.class_to_query_indices[c]) == 0:
                raise ValueError(f"Task B Class {c} has no query probes assigned! Ensure at least 1 probe per class.")

        # ---------------------------------------------------------------------
        # LAYER 0: Role Embeddings (Title vs Description vs Comment)
        # ---------------------------------------------------------------------
        self.role_embeddings = nn.Embedding(NUM_ROLES, d_model)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        self.norm_input = NormClass(d_model)
        self.dropout_input = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 1: Multi-Query Cross-Attention (Pre-Norm MHCA)
        # ---------------------------------------------------------------------
        self.query_embeddings = nn.Parameter(torch.empty(self.num_queries, d_model))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)

        # Pre-Norm for Queries & Context
        self.norm_q_cross = NormClass(d_model)
        self.norm_kv_cross = NormClass(d_model)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_cross = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # LAYER 2: Inter-Query Interaction Layer (Pre-Norm MHSA)
        # ---------------------------------------------------------------------
        if self.use_query_interaction:
            self.norm_self = NormClass(d_model)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.dropout_self = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # Aspect-to-Class Aggregation (Soft-Attention Pooling Heads)
        # ---------------------------------------------------------------------
        if self.pooling_mode == "attention":
            self.query_att_scorers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 4),
                    nn.Tanh(),
                    nn.Linear(d_model // 4, 1)
                ) for _ in range(NUM_CLASSES)
            ])

        # ---------------------------------------------------------------------
        # LAYER 3: Joint Comparative Classification Head
        # ---------------------------------------------------------------------
        self.norm_head_in = NormClass(NUM_CLASSES * d_model)
        self.classifier = nn.Sequential(
            self.norm_head_in,
            nn.Linear(NUM_CLASSES * d_model, hidden_dim),
            NormClass(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES)
        )

    def init_queries_from_text(self, tokenizer: Any, device: Optional[torch.device] = None):
        """
        Semantically initialize query_embeddings by encoding
        the natural language question strings using mmBERT's own embedding layer.
        Safely restores model training mode upon completion.
        """
        was_training = self.training
        self.eval()
        dev = device or self.query_embeddings.device
        
        try:
            with torch.no_grad():
                for i, probe in enumerate(self.probes):
                    text = probe.question_vi if probe.question_vi.strip() else probe.question_en
                    if not text.strip():
                        continue
                    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                    tokens = {k: v.to(dev) for k, v in tokens.items()}
                    
                    outputs = self.mmbert(**tokens)
                    mask = tokens["attention_mask"].unsqueeze(-1)
                    emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    self.query_embeddings.data[i].copy_(emb.squeeze(0).to(self.query_embeddings.dtype))
            print(f"[TaskBClassAwareAttentionModel] Initialized {self.num_queries} queries using mmBERT semantic embeddings.")
        finally:
            self.train(was_training)

    def pool_class_queries(self, z_queries: torch.Tensor) -> torch.Tensor:
        """
        Pools [B, K, d_model] query representations into [B, 3, d_model] class representations.
        """
        B, K, D = z_queries.shape
        class_representations = []

        for c in range(NUM_CLASSES):
            indices = self.class_to_query_indices[c]
            sub_z = z_queries[:, indices, :]

            if len(indices) == 1:
                class_representations.append(sub_z.squeeze(1))
            elif self.pooling_mode == "attention":
                scores = self.query_att_scorers[c](sub_z)       # [B, num_sub_queries, 1]
                weights = F.softmax(scores, dim=1)              # [B, num_sub_queries, 1]
                pooled_c = (weights * sub_z).sum(dim=1)         # [B, d_model]
                class_representations.append(pooled_c)
            elif self.pooling_mode == "max":
                pooled_c, _ = torch.max(sub_z, dim=1)           # [B, d_model]
                class_representations.append(pooled_c)
            else: # 'mean'
                pooled_c = torch.mean(sub_z, dim=1)             # [B, d_model]
                class_representations.append(pooled_c)

        return torch.stack(class_representations, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        return_attention_map: bool = False,
        detach_bridge: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            role_ids: [B, S]
            return_attention_map: Whether to compute & return the attention map [B, K, S]
            detach_bridge: Whether to detach h_B gradient to prevent Task C backward interference
        Returns:
            logits: [B, 3] (Classification scores for [no, yes_implicit, yes_explicit])
            h_B: [B, d_model] (Hate-Type-Aware Representation for Task C bridge)
            attn_weights: [B, K, S] if return_attention_map else None
        """
        B, S = input_ids.shape

        # ---------------------------------------------------------------------
        # LAYER 0: Encoder & Role Injection
        # ---------------------------------------------------------------------
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        with torch.set_grad_enabled(backbone_trainable):
            h_mmbert = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        e_role = self.role_embeddings(role_ids)
        if e_role.dtype != h_mmbert.dtype:
            e_role = e_role.to(h_mmbert.dtype)
            
        h_final = self.norm_input(h_mmbert + e_role)
        h_final = self.dropout_input(h_final) # [B, S, d_model]

        # ---------------------------------------------------------------------
        # LAYER 1: Multi-Query Cross-Attention (Pre-Norm Architecture)
        # Residual Highway: z = q + CrossAttn(Norm(q), Norm(h_final), Norm(h_final))
        # ---------------------------------------------------------------------
        q_raw = self.query_embeddings.unsqueeze(0).expand(B, -1, -1) # [B, K, d_model]
        if q_raw.dtype != h_final.dtype:
            q_raw = q_raw.to(h_final.dtype)
            
        key_padding_mask = (attention_mask == 0)

        # Pre-normalization on query and context inputs
        q_normed = self.norm_q_cross(q_raw)
        kv_normed = self.norm_kv_cross(h_final)

        if return_attention_map:
            z_attn, attn_weights = self.cross_attention(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True # [B, K, S]
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

        # Clean Pre-LN residual addition
        z = q_raw + self.dropout_cross(z_attn) # [B, K, d_model]

        # ---------------------------------------------------------------------
        # LAYER 2: Inter-Query Interaction Layer (Pre-Norm MHSA)
        # Residual Highway: z' = z + SelfAttn(Norm(z))
        # ---------------------------------------------------------------------
        if self.use_query_interaction:
            z_normed = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_normed,
                key=z_normed,
                value=z_normed,
                need_weights=False
            )
            z_prime = z + self.dropout_self(z_self) # [B, K, d_model]
        else:
            z_prime = z # [B, K, d_model]

        # ---------------------------------------------------------------------
        # Query-to-Class Aggregation: [B, K, d_model] -> [B, 3, d_model]
        # ---------------------------------------------------------------------
        z_classes = self.pool_class_queries(z_prime) # [B, 3, d_model]

        # ---------------------------------------------------------------------
        # LAYER 3: Joint Comparative Classification Head
        # ---------------------------------------------------------------------
        z_flat = z_classes.reshape(B, NUM_CLASSES * self.d_model) # [B, 3 * d_model]
        s = self.classifier(z_flat)                              # [B, 3]

        # ---------------------------------------------------------------------
        # TASK C BRIDGE: Hate-Type-Aware Representation h_B
        # ---------------------------------------------------------------------
        probs = F.softmax(s, dim=-1).unsqueeze(-1) # [B, 3, 1]
        h_B = (probs * z_classes).sum(dim=1)       # [B, d_model]

        if detach_bridge:
            h_B = h_B.detach()

        return s, h_B, attn_weights
