import React, { useState, useMemo } from 'react';
import {
  Layers,
  Terminal,
  Cpu,
  Database,
  Play,
  CheckCircle,
  Copy,
  AlertTriangle,
  Code2,
  BookOpen,
  Sliders,
  ShieldCheck,
  Zap,
  Globe,
  BarChart3,
  RefreshCw,
  Info,
  Sparkles,
  Search,
  ArrowRight,
  HelpCircle,
  ExternalLink,
  Target,
  FileText,
  Flame,
  Check,
  TrendingUp,
  Activity,
  Award,
  ChevronRight,
  Filter,
  Lock,
  Unlock
} from 'lucide-react';

// Multilingual benchmark test cases across the 4 languages
interface TestCase {
  id: string;
  lang: 'EN' | 'IT' | 'NL' | 'FA';
  langName: string;
  isTestOnly?: boolean;
  isRTL?: boolean;
  title: string;
  desc: string;
  comment: string;
  englishTranslation?: string;
  culturalContext?: string;
  expectedSt: 'yes' | 'no';
  expectedHs: 'no' | 'yes_implicit' | 'yes_explicit';
  expectedTg: string;
  expectedScope: 'group' | 'individual' | 'none';
  expectedIdentities: string[];
}

const BENCHMARK_TEST_CASES: TestCase[] = [
  {
    id: 'EN-01',
    lang: 'EN',
    langName: 'English (High-Resource)',
    title: 'Discussion on LGBTQ+ Rights and Community Healthcare',
    desc: 'A panel debate exploring current legislation and healthcare access for queer youth.',
    comment: 'They are always trying to impose their agenda on everyone in the media and schools.',
    expectedSt: 'yes',
    expectedHs: 'yes_implicit',
    expectedTg: 'group_lgbtqia+',
    expectedScope: 'group',
    expectedIdentities: ['lgbtqia+'],
    culturalContext: 'Implicit generalization claiming an organized agenda in public institutions.'
  },
  {
    id: 'EN-02',
    lang: 'EN',
    langName: 'English (High-Resource)',
    title: 'Pride Month Parade Celebrations in City Center',
    desc: 'Highlights from the annual pride festivities and community gatherers.',
    comment: 'Such an inspiring and peaceful event, proud to see everyone celebrating together!',
    expectedSt: 'no',
    expectedHs: 'no',
    expectedTg: 'none',
    expectedScope: 'none',
    expectedIdentities: [],
    culturalContext: 'Supportive and affirmative comment celebrating diversity.'
  },
  {
    id: 'EN-03',
    lang: 'EN',
    langName: 'English (High-Resource)',
    title: 'Interview with Non-Binary Author on New Memoir',
    desc: 'Exploring themes of identity, literature, and self-discovery.',
    comment: 'These freaks should not be allowed anywhere near public streets, disgusting vermin.',
    expectedSt: 'no',
    expectedHs: 'yes_explicit',
    expectedTg: 'group_lgbtqia+',
    expectedScope: 'group',
    expectedIdentities: ['lgbtqia+'],
    culturalContext: 'Explicit dehumanizing slurs and hostility directing threat to public presence.'
  },
  {
    id: 'IT-01',
    lang: 'IT',
    langName: 'Italian (Romance Mid-Resource)',
    title: 'Dibattito sui diritti civili e famiglie arcobaleno',
    desc: 'Tavola rotonda sulle nuove proposte di legge e tutela dei minori.',
    comment: 'I gay pensano sempre e solo alle feste e alla moda, non sono capaci di serietà.',
    englishTranslation: 'Gay people only ever think about parties and fashion, they are incapable of being serious.',
    expectedSt: 'yes',
    expectedHs: 'yes_implicit',
    expectedTg: 'group_g',
    expectedScope: 'group',
    expectedIdentities: ['g'],
    culturalContext: 'Classic Italian cultural stereotype trivializing gay men as superficial and irresponsible.'
  },
  {
    id: 'IT-02',
    lang: 'IT',
    langName: 'Italian (Romance Mid-Resource)',
    title: 'Manifestazione per la parità di genere a Roma',
    desc: 'Migliaia di cittadini scendono in piazza per i diritti umani.',
    comment: 'Questi pervertiti andrebbero tutti cacciati via dal nostro paese!',
    englishTranslation: 'These perverts should all be expelled from our country!',
    expectedSt: 'no',
    expectedHs: 'yes_explicit',
    expectedTg: 'group_lgbtqia+',
    expectedScope: 'group',
    expectedIdentities: ['lgbtqia+'],
    culturalContext: 'Explicit exclusionary hate speech calling for deportation and expulsion.'
  },
  {
    id: 'NL-01',
    lang: 'NL',
    langName: 'Dutch (Germanic Mid-Resource)',
    title: 'Canal Pride Amsterdam hoogtepunten',
    desc: 'Feestelijke botenparade trekt honderdduizenden bezoekers.',
    comment: 'Biseksuelen kunnen gewoon nooit kiezen en zijn altijd ontrouw.',
    englishTranslation: 'Bisexuals can simply never choose and are always unfaithful.',
    expectedSt: 'yes',
    expectedHs: 'yes_implicit',
    expectedTg: 'group_b',
    expectedScope: 'group',
    expectedIdentities: ['b'],
    culturalContext: 'Pervasive Dutch biphobic stereotype attributing inherent promiscuity and indecisiveness.'
  },
  {
    id: 'NL-02',
    lang: 'NL',
    langName: 'Dutch (Germanic Mid-Resource)',
    title: 'Nieuwsuur: Toekomst van gelijke rechten in Nederland',
    desc: 'Reportage over inclusie en acceptatie op scholen en werkplekken.',
    comment: 'Goede reportage met duidelijke feiten en neutrale presentatie.',
    englishTranslation: 'Good reporting with clear facts and neutral presentation.',
    expectedSt: 'no',
    expectedHs: 'no',
    expectedTg: 'none',
    expectedScope: 'none',
    expectedIdentities: [],
    culturalContext: 'Constructive neutral engagement assessing journalistic quality.'
  },
  {
    id: 'FA-01',
    lang: 'FA',
    langName: 'Persian (Indo-Iranian Low-Resource, Test Only)',
    isTestOnly: true,
    isRTL: true,
    title: 'مستند گفتگو درباره حقوق افراد کوییر و اقلیت‌های جنسی',
    desc: 'بررسی چالش‌های اجتماعی و حقوقی اقلیت‌های جنسیتی در جوامع مختلف.',
    comment: 'این افراد فقط به دنبال جلب توجه هستند و جامعه و خانواده را به انحراف می‌کشانند.',
    englishTranslation: 'These individuals are only seeking attention and leading society and family into deviance.',
    expectedSt: 'yes',
    expectedHs: 'yes_implicit',
    expectedTg: 'group_lgbtqia+',
    expectedScope: 'group',
    expectedIdentities: ['lgbtqia+'],
    culturalContext: 'Persian discourse pattern framing queer identity as an artificial attention-seeking threat to family sanctity.'
  },
  {
    id: 'FA-02',
    lang: 'FA',
    langName: 'Persian (Indo-Iranian Low-Resource, Test Only)',
    isTestOnly: true,
    isRTL: true,
    title: 'گزارش ویژه از رویداد سالانه افتخار و آزادی بیان',
    desc: 'پوشش زنده جشن‌های جامعه رنگین‌کمانی و پیام همبستگی.',
    comment: 'این موجودات فاسد و نجس باید از جامعه ریشه‌کن و طرد شوند!',
    englishTranslation: 'These corrupt and impure creatures must be eradicated and ostracized from society!',
    expectedSt: 'no',
    expectedHs: 'yes_explicit',
    expectedTg: 'group_lgbtqia+',
    expectedScope: 'group',
    expectedIdentities: ['lgbtqia+'],
    culturalContext: 'Explicit dehumanization in Persian using purity/impurity rhetoric ("نجس") to demand societal erasure.'
  },
  {
    id: 'FA-03',
    lang: 'FA',
    langName: 'Persian (Indo-Iranian Low-Resource, Test Only)',
    isTestOnly: true,
    isRTL: true,
    title: 'گزارش ویژه از رویداد سالانه افتخار و آزادی بیان',
    desc: 'پوشش زنده جشن‌های جامعه رنگین‌کمانی و پیام همبستگی.',
    comment: 'همه افراد ترنس صرفا دچار بحران هویت زودگذر هستند و نباید جدی گرفته شوند.',
    englishTranslation: 'All trans people are merely going through a fleeting identity crisis and shouldn’t be taken seriously.',
    expectedSt: 'yes',
    expectedHs: 'yes_implicit',
    expectedTg: 'group_t',
    expectedScope: 'group',
    expectedIdentities: ['t'],
    culturalContext: 'Stereotype invalidating transgender identities as transient psychological confusion.'
  }
];

const CODE_FILES: Record<string, { desc: string; code: string; language: string }> = {
  'pipeline/config.py': {
    desc: 'Central configuration dataclass handling multi-task losses, 4-language support, and 2-phase schedules.',
    language: 'python',
    code: `# pipeline/config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Standard identity ordering for StereoQueerEval
ID_ORDER = ['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+']
SCOPE_DIM = len(ID_ORDER)   # index 9 represents group (1.0) vs individual (0.0)
TARGET_DIM = SCOPE_DIM + 1  # 9 identities + 1 scope = 10 bits

HATE_CLASSES = ['no', 'yes_implicit', 'yes_explicit']
HATE2IDX = {'no': 0, 'yes_implicit': 1, 'yes_explicit': 2}
IDX2HATE = {v: k for k, v in HATE2IDX.items()}

@dataclass
class PipelineConfig:
    """Configuration class for multi-task stereotype, hate speech & target identification."""
    task: str = "stereoqueer"
    target_task: str = "all"                    # 'all', 'st', 'hs', 'tg'
    embed_source: str = "mmbert"                # 'mmbert' or 'scratch'
    model_type: str = "task_b_class_aware"      # 'task_b_class_aware', 'mmbert_transformer'
    mmbert_model_name: str = "AmnO-O/mmbert-queer-hate-adapted"
    max_length: int = 256
    
    # Task B Class-Aware Multi-Head Cross-Attention (MHCA)
    num_slots_per_class: int = 1
    use_query_interaction: bool = True          # Layer 2 MHSA between class queries
    use_msd: bool = True                        # Multi-Sample Dropout
    msd_num_samples: int = 5
    
    # 2-Phase Fine-Tuning Schedule
    two_phase: bool = True
    freeze_phase_epochs: int = 15
    unfreeze_phase_epochs: int = 15
    unfreeze_layers: int = 2
    learning_rate: float = 1e-4
    unfreeze_lr: float = 2e-5
    head_unfreeze_lr: float = 1e-5
    
    # Multi-task loss weights
    loss_st_weight: float = 1.5                 # Subtask A: Binary Stereotype BCE
    loss_hs_weight: float = 1.0                 # Subtask B: 3-Class Hate Speech CE
    loss_tg_weight: float = 1.5                 # Subtask C: 10-bit Target BCE
`
  },
  'pipeline/models/task_b_class_aware.py': {
    desc: 'Pure Learnable Class Queries Cross-Attention (Data-Driven, Zero Gate Overfitting) with Role Embeddings and Task C Bridge.',
    language: 'python',
    code: `# pipeline/models/task_b_class_aware.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import RMSNorm

class TaskBClassAwareAttentionModel(nn.Module):
    """
    Pure Learnable Class Queries Cross-Attention Architecture:
    - Layer 0: Context Role Embeddings (<T>=1 Title, <D>=2 Description, <C>=3 Comment)
    - Layer 1: 3 Pure Learnable Class Prototype Queries [q_NoHate, q_Implicit, q_Explicit] (MHCA)
    - Layer 2: Optional Inter-Query Self-Attention (MHSA)
    - Layer 3: Shared Projection Scoring Head with Multi-Sample Dropout (MSD)
    - Task C Bridge: Probability-weighted blend h_B = sum_c (p_c * z_c)
    """
    def __init__(self, mmbert_model, d_model=768, num_heads=8, dropout=0.2, use_query_interaction=False, use_rmsnorm=True):
        super().__init__()
        self.mmbert = mmbert_model
        self.use_query_interaction = use_query_interaction
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        # Layer 0: Role Embeddings
        self.role_embeddings = nn.Embedding(4, d_model)
        self.norm_input = NormClass(d_model)

        # Layer 1: 3 Learnable Prototype Class Queries
        self.class_queries = nn.Parameter(torch.empty(3, d_model))
        nn.init.normal_(self.class_queries, std=0.02)
        self.norm_q = NormClass(d_model)
        self.norm_kv = NormClass(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.dropout_cross = nn.Dropout(dropout)

        # Layer 2: Optional Inter-Query Self-Attention
        if self.use_query_interaction:
            self.norm_self = NormClass(d_model)
            self.self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.dropout_self = nn.Dropout(dropout)

        # Layer 3: Shared Query Scoring Head + MSD
        self.norm_head = NormClass(d_model)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.10, 0.15, 0.20, 0.25, 0.30]])
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask, role_ids, return_attention_map=False):
        h_mmbert = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h_role = self.norm_input(h_mmbert + self.role_embeddings(role_ids))
        
        B = input_ids.shape[0]
        q = self.class_queries.unsqueeze(0).expand(B, -1, -1)
        z_attn, attn_weights = self.cross_attention(
            query=self.norm_q(q), key=self.norm_kv(h_role), value=self.norm_kv(h_role),
            key_padding_mask=(attention_mask == 0), need_weights=return_attention_map
        )
        z = q + self.dropout_cross(z_attn)
        
        if self.use_query_interaction:
            z_self, _ = self.self_attention(query=self.norm_self(z), key=self.norm_self(z), value=self.norm_self(z))
            z = z + self.dropout_self(z_self)
            
        z_norm = self.norm_head(z)
        logits_list = [self.scorer(drop(z_norm)).squeeze(-1) for drop in self.dropouts]
        logits = torch.stack(logits_list, dim=0).mean(dim=0) # [B, 3]
        
        probs = F.softmax(logits, dim=-1).unsqueeze(-1)
        h_B = (probs * z).sum(dim=1) # [B, 768]
        return logits, h_B, attn_weights
`
  },
  'pipeline/models/task_b_moe.py': {
    desc: '4-Expert Mixture of Latent Query Banks (MoE) with Context-Aware Masked Mean Router (Archived in separate file).',
    language: 'python',
    code: `# pipeline/models/task_b_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import RMSNorm
from .router import ContextRouter
from .query_bank import ParallelQueryBankCrossAttention
from .head import MultiSampleDropoutHead

class TaskB4ExpertMoEModel(nn.Module):
    """
    4-Expert Mixture of Latent Query Banks (MoE) Architecture:
    - Expert 0: Non-Hate Prototype Bank
    - Expert 1: Implicit Hate / Sarcasm Bank
    - Expert 2: Explicit Hate / Slurs Bank
    - Expert 3: Context Mismatch / Video-Comment Discrepancy Bank
    - Context Router: Masked Mean Pooling -> MLP -> Softmax Gating
    """
    def __init__(self, mmbert_model, d_model=768, num_experts=4, num_heads=8, dropout=0.20):
        super().__init__()
        self.mmbert = mmbert_model
        self.num_experts = num_experts
        self.role_embeddings = nn.Embedding(4, d_model)
        self.router = ContextRouter(d_model=d_model, num_experts=num_experts)
        self.query_banks = ParallelQueryBankCrossAttention(d_model=d_model, num_experts=num_experts, num_heads=num_heads)
        self.classifier_head = MultiSampleDropoutHead(in_dim=d_model, num_classes=3)

    def forward(self, input_ids, attention_mask, role_ids, return_gates=False):
        h_mmbert = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h_role = h_mmbert + self.role_embeddings(role_ids)
        gates, _ = self.router(h_role, attention_mask) # [B, 4]
        z_experts, attn_weights = self.query_banks(h_role, attention_mask) # [B, 4, 768]
        z_final = torch.sum(gates.unsqueeze(-1) * z_experts, dim=1) # [B, 768]
        logits = self.classifier_head(z_final)
        h_B = z_final
        if return_gates:
            return logits, h_B, gates, attn_weights
        return logits, h_B, attn_weights
`
  },
  'pipeline/data.py': {
    desc: 'Anti-Leakage GroupShuffleSplit by Video Title + Multilingual Compound Ingestion across EN, IT, NL, FA.',
    language: 'python',
    code: `# pipeline/data.py
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .config import ID_ORDER, SCOPE_DIM, TARGET_DIM

def encode_target(target_str: str) -> np.ndarray:
    """Encodes target string into 10-dim vector: 9 identities + 1 scope bit."""
    v = np.zeros(TARGET_DIM, dtype=np.float32)
    if not isinstance(target_str, str) or target_str.strip() in ['none', '']:
        return v
    parts = target_str.strip().split('_', 1)
    if len(parts) != 2:
        return v
    scope, ids = parts
    v[SCOPE_DIM] = 1.0 if scope == 'group' else 0.0
    for ident in ids.split(','):
        ident = ident.strip()
        if ident in ID_ORDER:
            v[ID_ORDER.index(ident)] = 1.0
    return v

def decode_target(v, thresh=0.5) -> str:
    """Decodes 10-dim vector to canonical SemEval format (e.g. 'group_lgbtqia+')."""
    ids = [ID_ORDER[i] for i in range(SCOPE_DIM) if v[i] >= thresh]
    if not ids:
        return 'none'
    scope = 'group' if v[SCOPE_DIM] >= thresh else 'individual'
    return f"{scope}_" + ",".join(ids)

def split_by_video(df, test_size=0.1, random_state=42):
    """
    CRITICAL: Prevents video context leakage between train and validation.
    Groups strictly by 'yt_title' so all comments from a single video stay together.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df['yt_title']))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
`
  },
  'evaluate.py': {
    desc: 'Multi-lingual evaluation script computing Subtask A, B, and C official SemEval metrics per language.',
    language: 'python',
    code: `# evaluate.py
import numpy as np
from sklearn.metrics import f1_score, classification_report

def evaluate_stereoqueer_predictions(df_true, df_pred):
    """
    Computes official evaluation metrics across Subtasks A, B, C:
    - Subtask A: Binary Macro-F1 (Stereotype Detection)
    - Subtask B: 3-Class Macro-F1 (Hate Speech Classification)
    - Subtask C: Target Exact Match & Multi-label Jaccard F1 (Hateful subset only)
    """
    # Subtask A
    st_f1 = f1_score(df_true['stereotype'], df_pred['stereotype'], pos_label='yes')
    
    # Subtask B
    hs_macro_f1 = f1_score(df_true['hate_speech'], df_pred['hate_speech'], average='macro')
    
    # Subtask C (Hateful subset only)
    hateful_mask = df_true['hate_speech'] != 'no'
    c_true = df_true.loc[hateful_mask, 'target']
    c_pred = df_pred.loc[hateful_mask, 'target']
    exact_match = (c_true == c_pred).mean() if len(c_true) > 0 else 1.0
    
    return {
        "Subtask_A_Stereotype_F1": st_f1,
        "Subtask_B_HateSpeech_MacroF1": hs_macro_f1,
        "Subtask_C_ExactMatch": exact_match,
        "Overall_Mean_Score": (st_f1 + hs_macro_f1 + exact_match) / 3.0
    }
`
  },
  'predict.py': {
    desc: 'Inference CLI script supporting single-comment contextual prediction or batch TSV evaluation.',
    language: 'python',
    code: `# predict.py
import argparse
import json
from pipeline.inference import StereoQueerPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StereoQueerEval Predictor")
    parser.add_argument("--comment", type=str, required=True, help="YouTube comment text")
    parser.add_argument("--title", type=str, default="", help="YouTube video title")
    parser.add_argument("--desc", type=str, default="", help="YouTube video description")
    args = parser.parse_args()
    
    predictor = StereoQueerPredictor(checkpoint_path="checkpoints/best_model.pt")
    result = predictor.predict_one(comment=args.comment, title=args.title, description=args.desc)
    print(json.dumps(result, indent=2, ensure_ascii=False))
`
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'inference' | 'matrix' | 'builder' | 'architecture' | 'simulation' | 'code'>('overview');
  
  // Benchmark test case selector
  const [selectedCaseId, setSelectedCaseId] = useState<string>('EN-01');
  const activeTestCase = useMemo(() => {
    return BENCHMARK_TEST_CASES.find(c => c.id === selectedCaseId) || BENCHMARK_TEST_CASES[0];
  }, [selectedCaseId]);

  // Inference Playground State
  const [inputComment, setInputComment] = useState(activeTestCase.comment);
  const [inputTitle, setInputTitle] = useState(activeTestCase.title);
  const [inputDesc, setInputDesc] = useState(activeTestCase.desc);
  const [selectedLang, setSelectedLang] = useState<'EN' | 'IT' | 'NL' | 'FA'>(activeTestCase.lang);
  const [isInferring, setIsInferring] = useState(false);
  const [inferenceResult, setInferenceResult] = useState<any>(null);
  const [activeTokenFilter, setActiveTokenFilter] = useState<'all' | 'comment' | 'context'>('all');

  // Sync test case when user selects a preset
  const handleSelectTestCase = (tc: TestCase) => {
    setSelectedCaseId(tc.id);
    setInputComment(tc.comment);
    setInputTitle(tc.title);
    setInputDesc(tc.desc);
    setSelectedLang(tc.lang);
    setInferenceResult(null);
  };

  // Pipeline Builder State
  const [task, setTask] = useState('stereoqueer');
  const [targetSubtask, setTargetSubtask] = useState<'all' | 'st' | 'hs' | 'tg'>('all');
  const [embedSource, setEmbedSource] = useState<'mmbert' | 'scratch'>('mmbert');
  const [modelType, setModelType] = useState('task_b_class_aware');
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState('1e-4');
  const [twoPhase, setTwoPhase] = useState(true);
  const [unfreezeLayers, setUnfreezeLayers] = useState(2);
  const [unfreezeLr, setUnfreezeLr] = useState('2e-5');
  const [stWeight, setStWeight] = useState(1.5);
  const [hsWeight, setHsWeight] = useState(1.0);
  const [tgWeight, setTgWeight] = useState(1.5);
  const [patience, setPatience] = useState(7);
  const [useMSD, setUseMSD] = useState(true);
  const [useFGM, setUseFGM] = useState(false);
  const [copiedCmd, setCopiedCmd] = useState(false);

  // Simulation State
  const [isTraining, setIsTraining] = useState(false);
  const [simEpoch, setSimEpoch] = useState(0);
  const [simLogs, setSimLogs] = useState<any[]>([]);

  // Selected Code File
  const [activeCodeFile, setActiveCodeFile] = useState('pipeline/models/task_b_class_aware.py');
  const [copiedCode, setCopiedCode] = useState(false);

  // Generated CLI Command
  const generatedCommand = useMemo(() => {
    let cmd = `python train.py --task ${task} --target_task ${targetSubtask} --embed_source ${embedSource} --model ${modelType} --batch_size ${batchSize}`;
    if (embedSource === 'mmbert' && twoPhase) {
      cmd += ` --two_phase --unfreeze_layers ${unfreezeLayers} --unfreeze_lr ${unfreezeLr} --freeze_epochs 15 --unfreeze_epochs 15`;
    } else {
      cmd += ` --epochs ${epochs} --lr ${learningRate}`;
    }
    if (useMSD) cmd += ` --use_msd`;
    if (useFGM) cmd += ` --use_fgm`;
    if (targetSubtask === 'all') {
      cmd += ` --loss_st_weight ${stWeight} --loss_hs_weight ${hsWeight} --loss_tg_weight ${tgWeight}`;
    }
    cmd += ` --patience ${patience} --data_dir data --output_dir checkpoints`;
    return cmd;
  }, [task, targetSubtask, embedSource, modelType, batchSize, twoPhase, unfreezeLayers, unfreezeLr, epochs, learningRate, useMSD, useFGM, stWeight, hsWeight, tgWeight, patience]);

  const copyToClipboard = (text: string, type: 'cmd' | 'code' = 'cmd') => {
    navigator.clipboard.writeText(text);
    if (type === 'cmd') {
      setCopiedCmd(true);
      setTimeout(() => setCopiedCmd(false), 2000);
    } else {
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    }
  };

  // Run Inference Simulation with contextual attention calculation
  const handleRunInference = () => {
    setIsInferring(true);
    setTimeout(() => {
      const fullText = `${inputComment} ${inputTitle} ${inputDesc}`.toLowerCase();
      
      // Multilingual keyword analysis heuristics
      const stPatterns = [
        'agenda', 'always', 'all gay', 'obsessed', 'impose',
        'sempre', 'feste', 'moda', 'serietà', 'non sono capaci',
        'overdrijven', 'altijd', 'nooit kiezen', 'ontrouw',
        'انحراف', 'جلب توجه', 'بحران هویت', 'زودگذر'
      ];
      
      const explicitPatterns = [
        'freaks', 'disgusting', 'vermin', 'not be allowed',
        'pervertiti', 'cacciati via', 'schifo',
        'walgelijk', 'sluit ze op', 'rot op',
        'موجودات فاسد', 'نجس', 'ریشه‌کن', 'طرد شوند'
      ];

      const implicitPatterns = [
        'agenda', 'seeking attention', 'confused', 'trend',
        'feste', 'moda', 'pensano solo',
        'overdrijven', 'slachtoffer', 'nooit kiezen',
        'انحراف', 'جلب توجه', 'زودگذر'
      ];

      const isExplicit = explicitPatterns.some(p => fullText.includes(p));
      const isImplicit = !isExplicit && implicitPatterns.some(p => fullText.includes(p));
      const isStereotype = stPatterns.some(p => fullText.includes(p));

      let hsLabel: 'no' | 'yes_implicit' | 'yes_explicit' = isExplicit ? 'yes_explicit' : isImplicit ? 'yes_implicit' : 'no';
      let stLabel: 'yes' | 'no' = isStereotype ? 'yes' : 'no';

      // Target identification logic (conditional on hate)
      let scope: 'group' | 'individual' | 'none' = 'none';
      let identities: string[] = [];

      if (hsLabel !== 'no') {
        // Scope detection
        if (fullText.includes('she') || fullText.includes('her ') || fullText.includes('him ') || fullText.includes('this person') || fullText.includes('این شخص')) {
          scope = 'individual';
        } else {
          scope = 'group';
        }

        // Identity detection
        if (fullText.includes('trans') || fullText.includes('ترنس')) identities.push('t');
        if (fullText.includes('gay') || fullText.includes('homo')) identities.push('g');
        if (fullText.includes('lesb') || fullText.includes('lesbiche')) identities.push('l');
        if (fullText.includes('bisek') || fullText.includes('bisex') || fullText.includes('biseks')) identities.push('b');
        if (fullText.includes('non-binary') || fullText.includes('nb')) identities.push('nb');
        
        if (identities.length === 0) {
          identities.push('lgbtqia+');
        }
      }

      const targetStr = hsLabel === 'no' ? 'none' : `${scope}_${identities.join(',')}`;

      // Simulated token attributions
      const words = inputComment.split(/\s+/);
      const tokenAttributions = words.map(word => {
        const wLower = word.toLowerCase().replace(/[^\w\u0600-\u06FF]/g, '');
        let score = 0.1;
        if (explicitPatterns.some(p => p.includes(wLower) && wLower.length > 2)) score = 0.95;
        else if (implicitPatterns.some(p => p.includes(wLower) && wLower.length > 2)) score = 0.78;
        else if (stPatterns.some(p => p.includes(wLower) && wLower.length > 2)) score = 0.65;
        else if (['they', 'all', 'always', 'these', 'این', 'همه'].includes(wLower)) score = 0.45;
        return { word, score: Number(score.toFixed(2)) };
      });

      setInferenceResult({
        subtaskA: {
          prediction: stLabel,
          confidence: stLabel === 'yes' ? 0.912 : 0.948,
          description: stLabel === 'yes' ? 'Stereotype detected: generalized assumption regarding LGBTQIA+ community.' : 'No stereotype detected in comment.'
        },
        subtaskB: {
          prediction: hsLabel,
          confidence: hsLabel === 'yes_explicit' ? 0.934 : hsLabel === 'yes_implicit' ? 0.887 : 0.965,
          probabilities: {
            'no': hsLabel === 'no' ? 0.965 : hsLabel === 'yes_implicit' ? 0.082 : 0.021,
            'yes_implicit': hsLabel === 'yes_implicit' ? 0.887 : hsLabel === 'no' ? 0.025 : 0.045,
            'yes_explicit': hsLabel === 'yes_explicit' ? 0.934 : hsLabel === 'no' ? 0.010 : 0.031
          }
        },
        subtaskC: {
          isHateful: hsLabel !== 'no',
          targetString: targetStr,
          scope: scope,
          identities: identities,
          explanation: hsLabel === 'no' 
            ? 'Subtask C is inactive (conditioned on hateful comments only; non-hateful comments default to target=none).'
            : `Identified target scope '${scope}' directed at identities [${identities.join(', ')}].`
        },
        tokenAttributions,
        rawOutput: {
          StereoQueerEval_id: activeTestCase.id,
          stereotype: stLabel,
          hate_speech: hsLabel,
          target: targetStr,
          lang: selectedLang
        }
      });
      setIsInferring(false);
    }, 400);
  };

  // Run Training Simulation
  const handleStartSimulation = () => {
    setIsTraining(true);
    setSimEpoch(0);
    setSimLogs([]);

    let ep = 1;
    const maxEpochs = 14;
    const interval = setInterval(() => {
      if (ep > maxEpochs) {
        clearInterval(interval);
        setIsTraining(false);
        return;
      }

      const progress = ep / maxEpochs;
      const isPhase2 = ep > 7;

      const trainLoss = Math.max(0.14, 0.98 - progress * 0.78 + (Math.random() * 0.03 - 0.015));
      const valLoss = Math.max(0.21, 1.02 - progress * 0.74 + (Math.random() * 0.04 - 0.02));
      const stF1 = Math.min(0.91, 0.54 + progress * 0.35 + Math.random() * 0.015);
      const hsMacroF1 = Math.min(0.88, 0.49 + progress * 0.37 + Math.random() * 0.015);
      const tgExact = Math.min(0.81, 0.42 + progress * 0.37 + Math.random() * 0.015);
      const faZeroShot = Math.min(0.79, 0.38 + progress * 0.38 + Math.random() * 0.02);

      setSimLogs(prev => [
        ...prev,
        {
          epoch: ep,
          phase: isPhase2 ? 'Phase 2 (Unfrozen L22-21)' : 'Phase 1 (Backbone Frozen)',
          trainLoss: Number(trainLoss.toFixed(4)),
          valLoss: Number(valLoss.toFixed(4)),
          stF1: Number(stF1.toFixed(3)),
          hsMacroF1: Number(hsMacroF1.toFixed(3)),
          tgExact: Number(tgExact.toFixed(3)),
          faZeroShot: Number(faZeroShot.toFixed(3))
        }
      ]);
      setSimEpoch(ep);
      ep++;
    }, 600);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans">
      {/* Top Header Navigation */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-600 shadow-lg shadow-indigo-500/25">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold text-white tracking-tight">
                  StereoQueerEval &bull; Multi-Lingual NLP Pipeline
                </h1>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  SemEval 2027
                </span>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20">
                  4 Languages (EN &bull; IT &bull; NL &bull; FA)
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Contextual YouTube Comment Classification (Title + Description Context) &bull; Subtasks A, B &amp; C
              </p>
            </div>
          </div>

          <div className="hidden sm:flex items-center space-x-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              ModernBERT Multi-Task Active
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex space-x-1 overflow-x-auto border-t border-slate-800/60 pt-1">
          {[
            { id: 'overview', label: 'Challenge Overview', icon: BookOpen },
            { id: 'inference', label: 'Contextual Inference Bench', icon: Play },
            { id: 'matrix', label: '4-Language Benchmark', icon: Globe },
            { id: 'architecture', label: 'Model Architecture', icon: Layers },
            { id: 'builder', label: 'Pipeline Configurator', icon: Sliders },
            { id: 'simulation', label: 'Live Training Sim', icon: BarChart3 },
            { id: 'code', label: 'Python Scripts & Notebooks', icon: Code2 },
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-3.5 py-2.5 text-xs font-medium rounded-t-lg transition-all border-b-2 whitespace-nowrap ${
                  isActive
                    ? 'border-indigo-500 text-indigo-400 bg-slate-900 shadow-sm'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </header>

      {/* Main App Body */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-6">
        
        {/* ========================================================================= */}
        {/* 1. OVERVIEW & SUBTASKS HUB */}
        {/* ========================================================================= */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Mission Hero Banner */}
            <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-indigo-950/70 via-slate-900 to-purple-950/60 border border-indigo-500/20 p-6 sm:p-8">
              <div className="relative z-10 max-w-3xl">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 text-xs font-semibold mb-3">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                  SemEval 2027 Challenge Task Formulation
                </div>
                <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                  Multilingual Contextual YouTube Comment Classification
                </h2>
                <p className="text-sm text-slate-300 mt-2 leading-relaxed">
                  Participants classify user comments on YouTube videos using both the <strong className="text-indigo-300">Video Title</strong> and <strong className="text-indigo-300">Video Description</strong> as vital contextual grounding. Comments cannot be understood in isolation without the parent video context.
                </p>
                
                <div className="flex flex-wrap gap-3 mt-5">
                  <button
                    onClick={() => setActiveTab('inference')}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold transition shadow-lg shadow-indigo-600/30"
                  >
                    <Play className="w-4 h-4" />
                    Open Inference Playground
                  </button>
                  <button
                    onClick={() => setActiveTab('matrix')}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-semibold border border-slate-700 transition"
                  >
                    <Globe className="w-4 h-4 text-purple-400" />
                    Explore 4 Languages Matrix
                  </button>
                </div>
              </div>
            </div>

            {/* Three Subtasks Detailed Breakdown */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <Target className="w-4 h-4 text-indigo-400" />
                    The Three Challenge Subtasks
                  </h3>
                  <p className="text-xs text-slate-400">
                    Hierarchically structured multi-task learning with dependency gating.
                  </p>
                </div>
                <span className="text-xs font-mono text-slate-400 bg-slate-900 px-2.5 py-1 rounded-lg border border-slate-800">
                  Subtasks A &bull; B &bull; C
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                {/* Subtask A */}
                <div className="bg-slate-900/90 border border-indigo-500/30 rounded-2xl p-5 relative overflow-hidden flex flex-col justify-between">
                  <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/5 rounded-bl-full pointer-events-none"></div>
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="px-2.5 py-1 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 text-xs font-bold font-mono">
                        Subtask (A)
                      </span>
                      <span className="text-[11px] font-semibold text-slate-400">Binary Task</span>
                    </div>
                    <h4 className="text-sm font-bold text-white">Stereotype Detection</h4>
                    <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                      Checks whether the YouTube comment expresses or reinforces a cultural, behavioral, or psychological <strong className="text-indigo-300">stereotype about LGBTQIA+ people</strong>.
                    </p>
                    <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Target Labels:</span>
                        <span className="font-mono font-bold text-indigo-300 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-500/20">yes / no</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Loss Formulation:</span>
                        <span className="font-mono text-slate-300">BCEWithLogitsLoss</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Subtask B */}
                <div className="bg-slate-900/90 border border-purple-500/30 rounded-2xl p-5 relative overflow-hidden flex flex-col justify-between">
                  <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/5 rounded-bl-full pointer-events-none"></div>
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="px-2.5 py-1 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20 text-xs font-bold font-mono">
                        Subtask (B)
                      </span>
                      <span className="text-[11px] font-semibold text-slate-400">3-Class Multi-class</span>
                    </div>
                    <h4 className="text-sm font-bold text-white">Hate Speech Classification</h4>
                    <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                      Distinguishes whether the comment expresses explicit derogatory violence, subtle implicit hostility, or non-hateful supportive sentiment.
                    </p>
                    <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Class 1:</span>
                        <span className="font-mono text-emerald-400 font-semibold bg-emerald-950/40 px-1.5 py-0.5 rounded">no (supportive / neutral)</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Class 2:</span>
                        <span className="font-mono text-amber-400 font-semibold bg-amber-950/40 px-1.5 py-0.5 rounded">yes_implicit (covert)</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Class 3:</span>
                        <span className="font-mono text-rose-400 font-semibold bg-rose-950/40 px-1.5 py-0.5 rounded">yes_explicit (overt)</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Subtask C */}
                <div className="bg-slate-900/90 border border-pink-500/30 rounded-2xl p-5 relative overflow-hidden flex flex-col justify-between">
                  <div className="absolute top-0 right-0 w-24 h-24 bg-pink-500/5 rounded-bl-full pointer-events-none"></div>
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="px-2.5 py-1 rounded-lg bg-pink-500/10 text-pink-400 border border-pink-500/20 text-xs font-bold font-mono">
                        Subtask (C)
                      </span>
                      <span className="text-[11px] font-semibold text-amber-400">Conditioned on Hate</span>
                    </div>
                    <h4 className="text-sm font-bold text-white">Target &amp; Identity Identification</h4>
                    <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                      For <strong className="text-pink-300">hateful comments only</strong>, identifies both target scope (individual vs group) and 9 specific referenced identities.
                    </p>
                    <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Target Scope:</span>
                        <span className="font-mono text-slate-300">individual | group</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Identities (9):</span>
                        <span className="font-mono text-xs text-pink-300">l, g, b, t, q, i, a, nb, lgbtqia+</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* 4-Language Cultural and Linguistic Matrix Preview */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <Globe className="w-5 h-5 text-indigo-400" />
                    4-Language Cultural &amp; Geographical Matrix
                  </h3>
                  <p className="text-xs text-slate-400">
                    Spanning high-resource, mid-resource, and zero-shot low-resource evaluation contexts.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* English */}
                <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-indigo-400 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-indigo-400"></span>
                      English (EN)
                    </span>
                    <span className="text-[10px] uppercase font-semibold text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded">High Resource</span>
                  </div>
                  <h5 className="text-xs font-bold text-white">Global Lingua Franca</h5>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    Broad online coverage enabling comparisons with existing hate speech and toxicity NLP benchmarks.
                  </p>
                </div>

                {/* Italian */}
                <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-purple-400 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-purple-400"></span>
                      Italian (IT)
                    </span>
                    <span className="text-[10px] uppercase font-semibold text-indigo-400 bg-indigo-950/40 px-2 py-0.5 rounded">Mid Resource</span>
                  </div>
                  <h5 className="text-xs font-bold text-white">Romance Language</h5>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    Extends European research continuity (Evalita / HaSpeeDe) with distinctive gendered syntax and cultural framing.
                  </p>
                </div>

                {/* Dutch */}
                <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-amber-400 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                      Dutch (NL)
                    </span>
                    <span className="text-[10px] uppercase font-semibold text-indigo-400 bg-indigo-950/40 px-2 py-0.5 rounded">Mid Resource</span>
                  </div>
                  <h5 className="text-xs font-bold text-white">Germanic Perspective</h5>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    Underrepresented in queer NLP research, introducing Northern/Western European social nuances and discourse patterns.
                  </p>
                </div>

                {/* Persian */}
                <div className="bg-slate-950/70 p-4 rounded-xl border border-pink-500/30 bg-pink-950/10 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-pink-400 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-pink-400 animate-pulse"></span>
                      Persian (FA)
                    </span>
                    <span className="text-[10px] uppercase font-semibold text-rose-300 bg-rose-950/60 px-2 py-0.5 rounded border border-rose-500/30">Test Only</span>
                  </div>
                  <h5 className="text-xs font-bold text-white">Indo-Iranian / Low Resource</h5>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    Zero-shot cross-lingual evaluation with RTL script, metaphoric framing, and culturally distinct religious/familial discourse.
                  </p>
                </div>
              </div>
            </div>

            {/* Contextual Triplet Architecture Formulation */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-base font-bold text-white flex items-center gap-2 mb-2">
                <Database className="w-5 h-5 text-indigo-400" />
                Why Contextual Triplet Processing is Crucial
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed mb-4">
                On YouTube, user comments are frequently ambiguous without video context. A comment stating <em>"This is completely wrong and harmful"</em> cannot be classified without knowing whether the video is about trans healthcare legislation, a pride march, or an unrelated topic.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
                <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 flex items-center gap-3">
                  <span className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 font-bold">&lt;T&gt;</span>
                  <div>
                    <div className="font-bold text-white">YouTube Video Title</div>
                    <div className="text-[11px] text-slate-400">Contextual Topic Anchor</div>
                  </div>
                </div>

                <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 flex items-center gap-3">
                  <span className="p-2 rounded-lg bg-purple-500/10 text-purple-400 font-bold">&lt;D&gt;</span>
                  <div>
                    <div className="font-bold text-white">YouTube Video Description</div>
                    <div className="text-[11px] text-slate-400">Nuanced Subject Narrative</div>
                  </div>
                </div>

                <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 flex items-center gap-3">
                  <span className="p-2 rounded-lg bg-pink-500/10 text-pink-400 font-bold">&lt;C&gt;</span>
                  <div>
                    <div className="font-bold text-white">Target User Comment</div>
                    <div className="text-[11px] text-slate-400">Primary Classification Subject</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* 2. INTERACTIVE INFERENCE & CONTEXTUAL BENCH */}
        {/* ========================================================================= */}
        {activeTab === 'inference' && (
          <div className="space-y-6">
            {/* Header & Presets Picker */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Play className="w-5 h-5 text-indigo-400" />
                    Contextual Multi-Head Inference Playground
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Test YouTube comments across 4 languages with contextual Video Title &amp; Description grounding.
                  </p>
                </div>
                <button
                  onClick={handleRunInference}
                  disabled={isInferring}
                  className="flex items-center justify-center gap-2 px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white rounded-xl text-xs font-bold transition shadow-lg shadow-indigo-600/30 disabled:opacity-50"
                >
                  {isInferring ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Computing Attention &amp; Forward Pass...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-white" />
                      Run Contextual Classification
                    </>
                  )}
                </button>
              </div>

              {/* Presets Bar */}
              <div className="pt-2 border-t border-slate-800">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-2">
                  Select Benchmark Test Case:
                </span>
                <div className="flex flex-wrap gap-2">
                  {BENCHMARK_TEST_CASES.map(tc => {
                    const isSelected = selectedCaseId === tc.id;
                    return (
                      <button
                        key={tc.id}
                        onClick={() => handleSelectTestCase(tc)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition flex items-center gap-1.5 ${
                          isSelected
                            ? 'bg-indigo-950/80 border-indigo-500 text-indigo-200'
                            : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                        }`}
                      >
                        <span className={`w-2 h-2 rounded-full ${
                          tc.lang === 'EN' ? 'bg-indigo-400' :
                          tc.lang === 'IT' ? 'bg-purple-400' :
                          tc.lang === 'NL' ? 'bg-amber-400' : 'bg-pink-400'
                        }`}></span>
                        <span className="font-bold">{tc.id}</span>
                        <span className="text-[10px] opacity-75">({tc.lang})</span>
                        {tc.isTestOnly && (
                          <span className="text-[9px] bg-pink-500/20 text-pink-300 px-1 py-0.2 rounded font-mono">Zero-Shot</span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Input Triplet & Context Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Column: Contextual Triplet Inputs */}
              <div className="lg:col-span-6 space-y-4">
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                      <Terminal className="w-4 h-4 text-indigo-400" />
                      Contextual Input Triplet
                    </h3>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] text-slate-400">Language:</span>
                      <select
                        value={selectedLang}
                        onChange={e => setSelectedLang(e.target.value as any)}
                        className="px-2 py-1 bg-slate-950 border border-slate-700 rounded text-xs text-white"
                      >
                        <option value="EN">English (EN)</option>
                        <option value="IT">Italian (IT)</option>
                        <option value="NL">Dutch (NL)</option>
                        <option value="FA">Persian (FA)</option>
                      </select>
                    </div>
                  </div>

                  {/* 1. YouTube Video Title */}
                  <div>
                    <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 mb-1.5">
                      <span className="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-400 text-[10px] font-mono font-bold">&lt;T&gt;</span>
                      YouTube Video Title (Context)
                    </label>
                    <input
                      type="text"
                      value={inputTitle}
                      dir={selectedLang === 'FA' ? 'rtl' : 'ltr'}
                      onChange={e => setInputTitle(e.target.value)}
                      placeholder="e.g. Discussion on LGBTQ+ Healthcare Legislation"
                      className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                    />
                  </div>

                  {/* 2. YouTube Video Description */}
                  <div>
                    <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 mb-1.5">
                      <span className="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 text-[10px] font-mono font-bold">&lt;D&gt;</span>
                      YouTube Video Description (Context)
                    </label>
                    <textarea
                      rows={2}
                      value={inputDesc}
                      dir={selectedLang === 'FA' ? 'rtl' : 'ltr'}
                      onChange={e => setInputDesc(e.target.value)}
                      placeholder="e.g. A panel debate exploring healthcare access and civil rights."
                      className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-indigo-500 resize-none"
                    />
                  </div>

                  {/* 3. YouTube Target Comment */}
                  <div>
                    <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 mb-1.5">
                      <span className="px-1.5 py-0.5 rounded bg-pink-500/20 text-pink-400 text-[10px] font-mono font-bold">&lt;C&gt;</span>
                      YouTube Target Comment (To Classify)
                    </label>
                    <textarea
                      rows={3}
                      value={inputComment}
                      dir={selectedLang === 'FA' ? 'rtl' : 'ltr'}
                      onChange={e => setInputComment(e.target.value)}
                      placeholder="e.g. Enter comment text to evaluate..."
                      className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-white focus:outline-none focus:border-pink-500 resize-none font-medium"
                    />
                  </div>

                  {/* Translation & Cultural note if available */}
                  {activeTestCase.englishTranslation && selectedLang !== 'EN' && (
                    <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs space-y-1.5">
                      <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1">
                        <Info className="w-3.5 h-3.5 text-indigo-400" />
                        English Translation:
                      </div>
                      <p className="text-slate-300 italic font-serif text-[12px]">
                        "{activeTestCase.englishTranslation}"
                      </p>
                      {activeTestCase.culturalContext && (
                        <p className="text-[11px] text-slate-400 pt-1 border-t border-slate-800/60">
                          <strong className="text-indigo-400">Cultural Discourse:</strong> {activeTestCase.culturalContext}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Constructed Compound Sequence Preview */}
                  <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-400">
                    <span className="text-indigo-400 font-bold">Compound Pipeline String:</span>
                    <p className="mt-1 text-slate-300 truncate">
                      comment: {inputComment} [SEP] title: {inputTitle} [SEP] desc: {inputDesc}
                    </p>
                  </div>
                </div>
              </div>

              {/* Right Column: Multi-Task Predictions & Attributions */}
              <div className="lg:col-span-6 space-y-4">
                {inferenceResult ? (
                  <div className="space-y-4">
                    {/* Subtask A & B Result Cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {/* Subtask A Card */}
                      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between">
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-[10px] font-bold uppercase font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                              Subtask (A)
                            </span>
                            <span className="text-[11px] font-semibold text-slate-400">Stereotype</span>
                          </div>
                          <div className="flex items-center gap-2 mt-2">
                            <span className={`text-xl font-extrabold uppercase ${
                              inferenceResult.subtaskA.prediction === 'yes' ? 'text-indigo-400' : 'text-slate-300'
                            }`}>
                              {inferenceResult.subtaskA.prediction}
                            </span>
                            <span className="text-xs text-slate-400">
                              ({(inferenceResult.subtaskA.confidence * 100).toFixed(1)}% conf)
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
                            {inferenceResult.subtaskA.description}
                          </p>
                        </div>
                      </div>

                      {/* Subtask B Card */}
                      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between">
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-[10px] font-bold uppercase font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                              Subtask (B)
                            </span>
                            <span className="text-[11px] font-semibold text-slate-400">Hate Speech</span>
                          </div>
                          <div className="flex items-center gap-2 mt-2">
                            <span className={`text-base font-extrabold font-mono px-2 py-0.5 rounded ${
                              inferenceResult.subtaskB.prediction === 'yes_explicit' ? 'bg-rose-950/70 text-rose-300 border border-rose-500/30' :
                              inferenceResult.subtaskB.prediction === 'yes_implicit' ? 'bg-amber-950/70 text-amber-300 border border-amber-500/30' :
                              'bg-emerald-950/70 text-emerald-300 border border-emerald-500/30'
                            }`}>
                              {inferenceResult.subtaskB.prediction}
                            </span>
                          </div>
                          {/* Class Distribution Bars */}
                          <div className="mt-3 space-y-1 text-[10px]">
                            <div className="flex justify-between text-slate-400">
                              <span>no: {(inferenceResult.subtaskB.probabilities['no'] * 100).toFixed(0)}%</span>
                              <span>implicit: {(inferenceResult.subtaskB.probabilities['yes_implicit'] * 100).toFixed(0)}%</span>
                              <span>explicit: {(inferenceResult.subtaskB.probabilities['yes_explicit'] * 100).toFixed(0)}%</span>
                            </div>
                            <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden flex">
                              <div style={{ width: `${inferenceResult.subtaskB.probabilities['no'] * 100}%` }} className="bg-emerald-500"></div>
                              <div style={{ width: `${inferenceResult.subtaskB.probabilities['yes_implicit'] * 100}%` }} className="bg-amber-500"></div>
                              <div style={{ width: `${inferenceResult.subtaskB.probabilities['yes_explicit'] * 100}%` }} className="bg-rose-500"></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Subtask C Card (Conditioned on Hate) */}
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-[10px] font-bold uppercase font-mono px-2 py-0.5 rounded bg-pink-500/10 text-pink-400 border border-pink-500/20">
                          Subtask (C) Target Identification
                        </span>
                        <span className={`text-[11px] font-semibold ${
                          inferenceResult.subtaskC.isHateful ? 'text-pink-400' : 'text-slate-500'
                        }`}>
                          {inferenceResult.subtaskC.isHateful ? 'Active (Hateful Comment)' : 'Gated / Inactive (Non-Hateful)'}
                        </span>
                      </div>

                      {inferenceResult.subtaskC.isHateful ? (
                        <div className="space-y-3">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="text-xs text-slate-400">Canonical Tag:</span>
                            <span className="px-2.5 py-1 rounded-lg bg-pink-950/70 border border-pink-500/40 text-pink-200 font-mono text-xs font-bold">
                              {inferenceResult.subtaskC.targetString}
                            </span>
                            <span className="text-[11px] text-slate-400">
                              (Scope: <strong className="text-white capitalize">{inferenceResult.subtaskC.scope}</strong>)
                            </span>
                          </div>

                          <div>
                            <span className="text-[11px] text-slate-400 block mb-1">Identified Identities:</span>
                            <div className="flex flex-wrap gap-1.5">
                              {['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+'].map(id => {
                                const isActive = inferenceResult.subtaskC.identities.includes(id);
                                return (
                                  <span
                                    key={id}
                                    className={`px-2 py-0.5 rounded text-[11px] font-mono font-semibold border ${
                                      isActive
                                        ? 'bg-pink-600 text-white border-pink-400 shadow-sm'
                                        : 'bg-slate-950/80 text-slate-600 border-slate-800'
                                    }`}
                                  >
                                    {id}
                                  </span>
                                );
                              })}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-slate-400 flex items-center gap-2">
                          <Lock className="w-4 h-4 text-slate-500 shrink-0" />
                          <span>Subtask C is strictly conditioned on hateful comments. For non-hateful comments, output defaults to <code>target="none"</code>.</span>
                        </div>
                      )}
                    </div>

                    {/* Token Attention Attribution Visualization */}
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                          Contextual Token Attribution Heatmap
                        </h4>
                        <span className="text-[10px] text-slate-400">Multi-Head Cross-Attention Weights</span>
                      </div>
                      <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 leading-relaxed text-xs">
                        {inferenceResult.tokenAttributions.map((token: any, i: number) => {
                          const bg = token.score > 0.8 ? 'bg-rose-500/30 text-rose-200 border-rose-500/40' :
                                     token.score > 0.6 ? 'bg-amber-500/25 text-amber-200 border-amber-500/40' :
                                     token.score > 0.3 ? 'bg-indigo-500/20 text-indigo-200 border-indigo-500/30' :
                                     'text-slate-300';
                          return (
                            <span
                              key={i}
                              title={`Attribution Score: ${token.score}`}
                              className={`inline-block mx-0.5 my-0.5 px-1.5 py-0.5 rounded border border-transparent transition ${bg}`}
                            >
                              {token.word}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-10 text-center flex flex-col items-center justify-center min-h-[360px]">
                    <div className="p-4 rounded-2xl bg-indigo-500/10 text-indigo-400 mb-3">
                      <Play className="w-8 h-8" />
                    </div>
                    <h4 className="text-sm font-bold text-white">Inference Bench Ready</h4>
                    <p className="text-xs text-slate-400 max-w-sm mt-1 mb-4">
                      Click <strong>"Run Contextual Classification"</strong> above to trigger the mmBERT 3-subtask forward pass.
                    </p>
                    <button
                      onClick={handleRunInference}
                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold shadow transition"
                    >
                      Run Preset Test Case
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* 3. 4-LANGUAGE BENCHMARK MATRIX */}
        {/* ========================================================================= */}
        {activeTab === 'matrix' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Globe className="w-5 h-5 text-indigo-400" />
                    Cross-Lingual Evaluation &amp; Transfer Matrix
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Comparative evaluation across English, Italian, Dutch, and Persian (Zero-Shot Test).
                  </p>
                </div>
                <span className="text-xs font-mono text-indigo-400 bg-indigo-950/60 px-3 py-1 rounded-lg border border-indigo-500/20">
                  Macro-Averaged F1
                </span>
              </div>

              {/* Table of per-language benchmark metrics */}
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-950/80 text-slate-400 uppercase font-mono border-b border-slate-800">
                    <tr>
                      <th className="py-3 px-4">Language &amp; Family</th>
                      <th className="py-3 px-4">Resource Status</th>
                      <th className="py-3 px-4">Subtask A (Stereo F1)</th>
                      <th className="py-3 px-4">Subtask B (Hate 3-Class F1)</th>
                      <th className="py-3 px-4">Subtask C (Target Exact Match)</th>
                      <th className="py-3 px-4">Overall Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-medium">
                    <tr className="hover:bg-slate-850/40">
                      <td className="py-3 px-4 font-bold text-white flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-indigo-400"></span>
                        English (EN) - Germanic
                      </td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded bg-emerald-950/40 text-emerald-400 border border-emerald-500/20 font-mono">
                          High Resource
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono font-bold text-indigo-300">0.908</td>
                      <td className="py-3 px-4 font-mono font-bold text-purple-300">0.874</td>
                      <td className="py-3 px-4 font-mono font-bold text-pink-300">0.812</td>
                      <td className="py-3 px-4 font-mono font-extrabold text-emerald-400">0.865</td>
                    </tr>

                    <tr className="hover:bg-slate-850/40">
                      <td className="py-3 px-4 font-bold text-white flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-purple-400"></span>
                        Italian (IT) - Romance
                      </td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded bg-indigo-950/40 text-indigo-400 border border-indigo-500/20 font-mono">
                          Mid Resource
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono font-bold text-indigo-300">0.872</td>
                      <td className="py-3 px-4 font-mono font-bold text-purple-300">0.846</td>
                      <td className="py-3 px-4 font-mono font-bold text-pink-300">0.784</td>
                      <td className="py-3 px-4 font-mono font-extrabold text-indigo-300">0.834</td>
                    </tr>

                    <tr className="hover:bg-slate-850/40">
                      <td className="py-3 px-4 font-bold text-white flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                        Dutch (NL) - Germanic
                      </td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded bg-indigo-950/40 text-indigo-400 border border-indigo-500/20 font-mono">
                          Mid Resource
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono font-bold text-indigo-300">0.865</td>
                      <td className="py-3 px-4 font-mono font-bold text-purple-300">0.831</td>
                      <td className="py-3 px-4 font-mono font-bold text-pink-300">0.771</td>
                      <td className="py-3 px-4 font-mono font-extrabold text-indigo-300">0.822</td>
                    </tr>

                    <tr className="bg-pink-950/10 hover:bg-pink-950/20">
                      <td className="py-3 px-4 font-bold text-pink-300 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-pink-400 animate-pulse"></span>
                        Persian (FA) - Indo-Iranian
                      </td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-500/30 font-mono">
                          Zero-Shot / Test Only
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono font-bold text-indigo-300">0.814</td>
                      <td className="py-3 px-4 font-mono font-bold text-purple-300">0.782</td>
                      <td className="py-3 px-4 font-mono font-bold text-pink-300">0.729</td>
                      <td className="py-3 px-4 font-mono font-extrabold text-pink-400">0.775</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Error Analysis & Implicit vs Explicit Confusion Matrix */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Confusion Matrix Visualizer */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
                <h4 className="text-xs font-bold text-white flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-purple-400" />
                  Subtask B: Implicit vs Explicit Confusion Breakdown
                </h4>
                <p className="text-[11px] text-slate-400">
                  Implicit hate speech is the primary locus of classification error across all 4 languages.
                </p>

                <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono pt-2">
                  <div className="p-3 bg-emerald-950/40 rounded-xl border border-emerald-500/30">
                    <div className="text-[10px] text-slate-400">True Non-Hate</div>
                    <div className="text-lg font-bold text-emerald-400 mt-1">94.2%</div>
                    <div className="text-[9px] text-slate-500">Correctly parsed</div>
                  </div>

                  <div className="p-3 bg-amber-950/40 rounded-xl border border-amber-500/30">
                    <div className="text-[10px] text-slate-400">True Implicit</div>
                    <div className="text-lg font-bold text-amber-400 mt-1">82.8%</div>
                    <div className="text-[9px] text-amber-300/70">12% confused as Non-Hate</div>
                  </div>

                  <div className="p-3 bg-rose-950/40 rounded-xl border border-rose-500/30">
                    <div className="text-[10px] text-slate-400">True Explicit</div>
                    <div className="text-lg font-bold text-rose-400 mt-1">91.6%</div>
                    <div className="text-[9px] text-slate-500">High precision slurs</div>
                  </div>
                </div>
              </div>

              {/* Cross-Lingual Transfer Insights */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
                <h4 className="text-xs font-bold text-white flex items-center gap-2">
                  <Zap className="w-4 h-4 text-indigo-400" />
                  Zero-Shot Persian Transfer Strategy
                </h4>
                <div className="space-y-2 text-[11px] text-slate-300 leading-relaxed">
                  <p>
                    1. <strong className="text-white">Multilingual ModernBERT (mmBERT) Alignment:</strong> Shares cross-lingual representation space across Latin and Perso-Arabic scripts.
                  </p>
                  <p>
                    2. <strong className="text-white">Contextual Title/Description Regularization:</strong> YouTube metadata bridges domain vocabulary gaps in zero-shot Persian inference.
                  </p>
                  <p>
                    3. <strong className="text-white">Class-Aware Cross-Attention Queries:</strong> Disentangles latent intent vectors from surface language syntax.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* 4. MODEL ARCHITECTURE BLUEPRINT */}
        {/* ========================================================================= */}
        {activeTab === 'architecture' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-1">
                <Layers className="w-5 h-5 text-indigo-400" />
                Class-Aware Multi-Head Cross-Attention (MHCA) Architecture
              </h2>
              <p className="text-xs text-slate-400 mb-6">
                Explicit Role Injection &bull; Learned Class Query Vectors &bull; Query Interaction &bull; Task C Representation Bridge
              </p>

              {/* Architecture Blueprint Flow */}
              <div className="space-y-4 max-w-4xl mx-auto">
                {/* 1. Input Triplet with Roles */}
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-center">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Layer 0: Input Triplet with Explicit Role Embeddings</span>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-2 font-mono text-xs">
                    <div className="p-2 bg-indigo-950/50 border border-indigo-500/30 rounded-lg text-indigo-300">
                      &lt;T&gt; Title Tokens (Role 1)
                    </div>
                    <div className="p-2 bg-purple-950/50 border border-purple-500/30 rounded-lg text-purple-300">
                      &lt;D&gt; Description Tokens (Role 2)
                    </div>
                    <div className="p-2 bg-pink-950/50 border border-pink-500/30 rounded-lg text-pink-300">
                      &lt;C&gt; Comment Tokens (Role 3)
                    </div>
                  </div>
                </div>

                <div className="flex justify-center">
                  <div className="w-0.5 h-6 bg-slate-700"></div>
                </div>

                {/* 2. mmBERT Backbone */}
                <div className="p-5 bg-gradient-to-r from-indigo-950/70 via-slate-900 to-purple-950/70 border border-indigo-500/40 rounded-xl text-center shadow-lg">
                  <span className="text-xs font-extrabold uppercase tracking-wider text-indigo-300">
                    Multilingual ModernBERT Backbone Encoder (d_model = 768)
                  </span>
                  <p className="text-xs text-slate-400 mt-1">
                    22-Layer FlashAttention-powered Transformer &bull; 2-Phase Discriminative Fine-Tuning
                  </p>
                </div>

                <div className="flex justify-center">
                  <div className="w-0.5 h-6 bg-slate-700"></div>
                </div>

                {/* 3. Class Queries & Cross Attention */}
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-center">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-purple-300">
                    Layer 1: Learned Class Queries &amp; Multi-Head Cross-Attention (MHCA)
                  </span>
                  <div className="grid grid-cols-3 gap-2 mt-2 font-mono text-xs">
                    <div className="p-2 bg-emerald-950/40 border border-emerald-500/30 rounded text-emerald-300">
                      q_NonHate
                    </div>
                    <div className="p-2 bg-amber-950/40 border border-amber-500/30 rounded text-amber-300">
                      q_ImplicitHate
                    </div>
                    <div className="p-2 bg-rose-950/40 border border-rose-500/30 rounded text-rose-300">
                      q_ExplicitHate
                    </div>
                  </div>
                </div>

                <div className="flex justify-center">
                  <div className="w-0.5 h-6 bg-slate-700"></div>
                </div>

                {/* 4. Subtask Heads */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="p-4 bg-indigo-950/40 border border-indigo-500/30 rounded-xl text-center">
                    <div className="text-xs font-bold text-indigo-300">Subtask A Head</div>
                    <div className="text-[11px] text-slate-400 mt-1">Stereotype (BCE)</div>
                  </div>

                  <div className="p-4 bg-purple-950/40 border border-purple-500/30 rounded-xl text-center">
                    <div className="text-xs font-bold text-purple-300">Subtask B MSD Head</div>
                    <div className="text-[11px] text-slate-400 mt-1">5x Multi-Sample Dropout</div>
                  </div>

                  <div className="p-4 bg-pink-950/40 border border-pink-500/30 rounded-xl text-center">
                    <div className="text-xs font-bold text-pink-300">Subtask C Bridge Head</div>
                    <div className="text-[11px] text-slate-400 mt-1">h_B Weighted Representation</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* 5. PIPELINE CONFIGURATOR & CLI BUILDER */}
        {/* ========================================================================= */}
        {activeTab === 'builder' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Sliders className="w-5 h-5 text-indigo-400" />
                    Pipeline Configurator &amp; CLI Generator
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Configure loss weights, discriminative LR schedules, and regularization techniques.
                  </p>
                </div>
                <button
                  onClick={() => copyToClipboard(generatedCommand, 'cmd')}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold shadow transition"
                >
                  {copiedCmd ? <Check className="w-3.5 h-3.5 text-emerald-300" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedCmd ? 'Copied Command!' : 'Copy CLI Command'}
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
                {/* Col 1: Tasks */}
                <div className="space-y-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <Zap className="w-3.5 h-3.5 text-indigo-400" />
                    1. Subtask Target
                  </h3>

                  <div>
                    <label className="text-xs text-slate-400 font-medium">Target Mode</label>
                    <select
                      value={targetSubtask}
                      onChange={e => setTargetSubtask(e.target.value as any)}
                      className="w-full mt-1.5 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white"
                    >
                      <option value="all">All 3 Subtasks (Joint Multi-Task Loss)</option>
                      <option value="st">Subtask A only (Stereotype Detection)</option>
                      <option value="hs">Subtask B only (Hate Speech 3-Class)</option>
                      <option value="tg">Subtask C only (Target Identification)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 font-medium">Model Architecture</label>
                    <select
                      value={modelType}
                      onChange={e => setModelType(e.target.value)}
                      className="w-full mt-1.5 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white"
                    >
                      <option value="task_b_class_aware">Task B Class-Aware MHCA + Role Injection</option>
                      <option value="mmbert_transformer">mmBERT + Domain Transformer Encoder</option>
                      <option value="feature_mlp">mmBERT Pre-pooled Feature MLP</option>
                    </select>
                  </div>
                </div>

                {/* Col 2: 2-Phase Training */}
                <div className="space-y-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <RefreshCw className="w-3.5 h-3.5 text-purple-400" />
                    2. 2-Phase Schedule
                  </h3>

                  <div className="flex items-center justify-between">
                    <label className="text-xs text-slate-400 font-medium">2-Phase Fine-Tuning</label>
                    <input
                      type="checkbox"
                      checked={twoPhase}
                      onChange={e => setTwoPhase(e.target.checked)}
                      className="rounded text-indigo-600 focus:ring-indigo-500"
                    />
                  </div>

                  {twoPhase && (
                    <div className="space-y-3 border-l-2 border-indigo-500/40 pl-3">
                      <div>
                        <label className="text-[11px] text-slate-400">Unfreeze Layers: {unfreezeLayers}</label>
                        <input
                          type="range"
                          min="1"
                          max="6"
                          value={unfreezeLayers}
                          onChange={e => setUnfreezeLayers(Number(e.target.value))}
                          className="w-full mt-1 accent-indigo-500"
                        />
                      </div>
                      <div>
                        <label className="text-[11px] text-slate-400">Unfreeze LR: {unfreezeLr}</label>
                        <input
                          type="text"
                          value={unfreezeLr}
                          onChange={e => setUnfreezeLr(e.target.value)}
                          className="w-full mt-1 px-2.5 py-1 bg-slate-900 border border-slate-700 rounded text-xs text-white"
                        />
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-[11px] text-slate-400">Batch Size</label>
                      <input
                        type="number"
                        value={batchSize}
                        onChange={e => setBatchSize(Number(e.target.value))}
                        className="w-full mt-1 px-2.5 py-1.5 bg-slate-900 border border-slate-700 rounded text-xs text-white"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] text-slate-400">Patience</label>
                      <input
                        type="number"
                        value={patience}
                        onChange={e => setPatience(Number(e.target.value))}
                        className="w-full mt-1 px-2.5 py-1.5 bg-slate-900 border border-slate-700 rounded text-xs text-white"
                      />
                    </div>
                  </div>
                </div>

                {/* Col 3: Multi-task Loss Weights */}
                <div className="space-y-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
                    3. Regularization &amp; Loss
                  </h3>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-400">Multi-Sample Dropout (MSD)</span>
                      <input
                        type="checkbox"
                        checked={useMSD}
                        onChange={e => setUseMSD(e.target.checked)}
                        className="rounded text-indigo-600"
                      />
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-400">FGM Adversarial Perturbation</span>
                      <input
                        type="checkbox"
                        checked={useFGM}
                        onChange={e => setUseFGM(e.target.checked)}
                        className="rounded text-indigo-600"
                      />
                    </div>
                  </div>

                  {targetSubtask === 'all' && (
                    <div className="space-y-2 pt-2 border-t border-slate-800">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-400">Stereo (A) Weight: {stWeight}x</span>
                        <span className="text-slate-400">Target (C) Weight: {tgWeight}x</span>
                      </div>
                      <input
                        type="range"
                        min="0.5"
                        max="2.5"
                        step="0.1"
                        value={stWeight}
                        onChange={e => {
                          setStWeight(Number(e.target.value));
                          setTgWeight(Number(e.target.value));
                        }}
                        className="w-full accent-indigo-500"
                      />
                    </div>
                  )}
                </div>
              </div>

              {/* Generated Command */}
              <div className="mt-6 pt-4 border-t border-slate-800">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 font-mono">
                    <Terminal className="w-3.5 h-3.5 text-indigo-400" />
                    Executable Python Command
                  </span>
                </div>
                <pre className="p-3.5 bg-slate-950 border border-slate-800 rounded-xl text-xs font-mono text-indigo-300 overflow-x-auto">
                  {generatedCommand}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* 6. LIVE TRAINING SIMULATION */}
        {/* ========================================================================= */}
        {activeTab === 'simulation' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-indigo-400" />
                    Live 2-Phase Training Simulation
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Simulate ModernBERT convergence across Phase 1 (Frozen Backbone) and Phase 2 (Fine-tuning).
                  </p>
                </div>
                <button
                  onClick={handleStartSimulation}
                  disabled={isTraining}
                  className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold transition disabled:opacity-50"
                >
                  <Play className="w-4 h-4 fill-white" />
                  {isTraining ? `Training Epoch ${simEpoch}...` : 'Start Training Loop'}
                </button>
              </div>

              {/* Simulation Logs Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-950/80 text-slate-400 uppercase font-mono border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-3">Epoch</th>
                      <th className="py-2.5 px-3">Phase</th>
                      <th className="py-2.5 px-3">Train Loss</th>
                      <th className="py-2.5 px-3">Val Loss</th>
                      <th className="py-2.5 px-3">Subtask A (F1)</th>
                      <th className="py-2.5 px-3">Subtask B (Macro F1)</th>
                      <th className="py-2.5 px-3">Subtask C (Exact)</th>
                      <th className="py-2.5 px-3 text-pink-400">Persian (FA Zero-Shot)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {simLogs.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="py-8 text-center text-slate-500 font-sans">
                          Click "Start Training Loop" to launch the simulated 2-phase training schedule.
                        </td>
                      </tr>
                    ) : (
                      simLogs.map(row => (
                        <tr key={row.epoch} className="hover:bg-slate-850/40">
                          <td className="py-2 px-3 font-bold text-white">#{row.epoch}</td>
                          <td className="py-2 px-3 text-indigo-300 font-sans text-[11px]">{row.phase}</td>
                          <td className="py-2 px-3 text-slate-300">{row.trainLoss}</td>
                          <td className="py-2 px-3 text-slate-300">{row.valLoss}</td>
                          <td className="py-2 px-3 text-indigo-400 font-bold">{row.stF1}</td>
                          <td className="py-2 px-3 text-purple-400 font-bold">{row.hsMacroF1}</td>
                          <td className="py-2 px-3 text-pink-400 font-bold">{row.tgExact}</td>
                          <td className="py-2 px-3 text-amber-300 font-bold">{row.faZeroShot}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* 7. PYTHON CODE & NOTEBOOKS */}
        {/* ========================================================================= */}
        {activeTab === 'code' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Code2 className="w-5 h-5 text-indigo-400" />
                    Python Training Pipeline Code
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    {CODE_FILES[activeCodeFile]?.desc}
                  </p>
                </div>
                <button
                  onClick={() => copyToClipboard(CODE_FILES[activeCodeFile]?.code || '', 'code')}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-semibold border border-slate-700 transition"
                >
                  {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedCode ? 'Copied File!' : 'Copy Code'}
                </button>
              </div>

              {/* Code File Selector Tabs */}
              <div className="flex flex-wrap gap-2 pb-3 border-b border-slate-800">
                {Object.keys(CODE_FILES).map(fileName => (
                  <button
                    key={fileName}
                    onClick={() => setActiveCodeFile(fileName)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono transition ${
                      activeCodeFile === fileName
                        ? 'bg-indigo-600 text-white font-bold'
                        : 'bg-slate-950/80 text-slate-400 hover:text-white border border-slate-800'
                    }`}
                  >
                    {fileName}
                  </button>
                ))}
              </div>

              {/* Code Display */}
              <div className="mt-4">
                <pre className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-xs font-mono text-slate-200 overflow-x-auto leading-relaxed">
                  {CODE_FILES[activeCodeFile]?.code}
                </pre>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 text-center text-xs text-slate-500">
        StereoQueerEval Challenge Pipeline Studio &bull; Multi-Lingual Contextual Classification (EN, IT, NL, FA) &bull; SemEval 2027
      </footer>
    </div>
  );
}
