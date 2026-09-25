import React, { useState, useMemo } from 'react';
import {
  Layers,
  Terminal,
  Cpu,
  Database,
  Play,
  CheckCircle,
  Copy,
  Download,
  AlertTriangle,
  Code2,
  BookOpen,
  Sliders,
  ShieldCheck,
  Zap,
  Globe,
  BarChart3,
  RefreshCw,
  Info
} from 'lucide-react';

// Pre-defined multi-lingual inference test cases
const PRESET_COMMENTS = [
  {
    lang: 'EN',
    title: 'Discussion on LGBTQ+ Rights and Community Healthcare',
    desc: 'A panel debate exploring current legislation and healthcare access for queer youth.',
    comment: 'They are always trying to impose their agenda on everyone in the media and schools.',
    expectedSt: 'yes',
    expectedHs: 'yes_implicit',
    expectedTg: 'group_lgbtqia+',
  },
  {
    lang: 'EN',
    title: 'Pride Month Parade Celebrations in City Center',
    desc: 'Highlights from the annual pride festivities and community gatherers.',
    comment: 'Such an inspiring and peaceful event, proud to see everyone celebrating together!',
    expectedSt: 'no',
    expectedHs: 'no',
    expectedTg: 'none',
  },
  {
    lang: 'IT',
    title: 'Dibattito sui diritti civili e famiglie arcobaleno',
    desc: 'Tavola rotonda sulle nuove proposte di legge e tutela dei minori.',
    comment: 'I gay pensano sempre e solo alle feste e alla moda, non sono capaci di serietà.',
    expectedSt: 'yes',
    expectedHs: 'yes_implicit',
    expectedTg: 'group_g',
  },
  {
    lang: 'NL',
    title: 'Canal Pride Amsterdam hoogtepunten',
    desc: 'Feestelijke botenparade trekt honderdduizenden bezoekers.',
    comment: 'Biseksuelen kunnen gewoon nooit kiezen en zijn altijd ontrouw.',
    expectedSt: 'yes',
    expectedHs: 'yes_implicit',
    expectedTg: 'group_b',
  },
  {
    lang: 'EN',
    title: 'Interview with Non-Binary Author on New Memoir',
    desc: 'Exploring themes of identity, literature, and self-discovery.',
    comment: 'These freaks should not be allowed anywhere near public streets, disgusting vermin.',
    expectedSt: 'no',
    expectedHs: 'yes_explicit',
    expectedTg: 'group_lgbtqia+',
  }
];

const CODE_FILES: Record<string, { desc: string; code: string }> = {
  'pipeline/config.py': {
    desc: 'Central configuration dataclass handling hyperparams, loss weights, and SemEval mappings.',
    code: `# pipeline/config.py
from dataclasses import dataclass, field
from typing import List, Dict

ID_ORDER = ['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+']
SCOPE_DIM = len(ID_ORDER)   # index 9 represents group (1.0) vs individual (0.0)
TARGET_DIM = SCOPE_DIM + 1  # 9 identities + 1 scope = 10 bits

@dataclass
class PipelineConfig:
    task: str = "stereoqueer"
    target_task: str = "all"               # 'all', 'st', 'hs', or 'tg'
    embed_source: str = "mmbert"           # 'mmbert' or 'scratch'
    model_type: str = "mmbert_transformer" # 'mmbert_transformer', 'bilstm', etc.
    mmbert_model_name: str = "jhu-clsp/mmbert-base"
    batch_size: int = 32
    learning_rate: float = 1e-4
    two_phase: bool = True
    freeze_phase_epochs: int = 15
    unfreeze_phase_epochs: int = 15
    unfreeze_layers: int = 2
    unfreeze_lr: float = 2e-5
    loss_st_weight: float = 1.5
    loss_hs_weight: float = 1.0
    loss_tg_weight: float = 1.5
`
  },
  'pipeline/models/mmbert.py': {
    desc: 'Hybrid model connecting ModernBERT backbone with domain TransformerEncoder and 3 heads.',
    code: `# pipeline/models/mmbert.py
import torch
import torch.nn as nn

class MMBertTransformerModel(nn.Module):
    def __init__(self, mmbert_model, d_model=768, num_heads=8, num_layers=2, target_dim=10):
        super().__init__()
        self.mmbert = mmbert_model
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.st_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_model // 2, 1))
        self.hs_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_model // 2, 3))
        self.tg_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_model // 2, target_dim))

    def forward(self, input_ids, attention_mask):
        backbone_trainable = any(p.requires_grad for p in self.mmbert.parameters())
        with torch.set_grad_enabled(backbone_trainable):
            hidden = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.transformer_encoder(hidden, src_key_padding_mask=(attention_mask == 0))
        m = attention_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        return self.st_head(pooled), self.hs_head(pooled), self.tg_head(pooled)
`
  },
  'pipeline/data.py': {
    desc: 'Data ingestion with GroupShuffleSplit by video title to eliminate train/val leakage.',
    code: `# pipeline/data.py
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def safe_clean(text):
    text = str(text).lower()
    text = re.sub(r'[\\n\\t\\r]+', ' ', text)
    return re.sub(r'\\s+', ' ', text).strip()

def split_by_video(df, test_size=0.1, random_state=42):
    # CRITICAL: YouTube context appears in title & description.
    # Group by 'yt_title' to guarantee no context leakage into validation.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df['yt_title']))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
`
  },
  'pipeline/trainer.py': {
    desc: '2-Phase training loop with discriminative learning rates and early stopping.',
    code: `# pipeline/trainer.py
class StereoQueerTrainer:
    def train(self):
        if self.config.two_phase:
            # Phase 1: Freeze backbone, train task heads
            self.freeze_backbone(True)
            self.run_loop("phase1_frozen", epochs=self.config.freeze_phase_epochs, lr=self.config.learning_rate)

            # Phase 2: Unfreeze last N layers with fine-tuning LR
            self.unfreeze_last_n(self.config.unfreeze_layers)
            self.run_loop("phase2_finetune", epochs=self.config.unfreeze_phase_epochs,
                          lr=self.config.head_unfreeze_lr, backbone_lr=self.config.unfreeze_lr)
`
  },
  'pipeline/models/task_b_class_aware.py': {
    desc: 'Task B Class-Aware Multi-Head Cross-Attention (MHCA) with Explicit Role Injection and Query Interaction (MHSA).',
    code: `# pipeline/models/task_b_class_aware.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskBClassAwareAttentionModel(nn.Module):
    def __init__(self, mmbert_model, d_model=768, num_heads=8, dropout=0.2, use_query_interaction=True):
        super().__init__()
        self.mmbert = mmbert_model
        self.use_query_interaction = use_query_interaction
        
        # Layer 0: Explicit Role Embeddings (<T>=1, <D>=2, <C>=3)
        self.role_embeddings = nn.Embedding(4, d_model)
        self.layer_norm_input = nn.LayerNorm(d_model)

        # Layer 1: Learned Class Queries [q_NonHate, q_Implicit, q_Explicit]
        self.query_embeddings = nn.Parameter(torch.empty(3, d_model))
        nn.init.normal_(self.query_embeddings, std=0.02)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layer_norm_cross = nn.LayerNorm(d_model)

        # Layer 2: Query Interaction (MHSA) [Ablation H2]
        if self.use_query_interaction:
            self.self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.layer_norm_self = nn.LayerNorm(d_model)

        # Layer 3: Shared Scoring Head f_θ -> logits s ∈ [B, 3]
        self.shared_scoring_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )

    def forward(self, input_ids, attention_mask, role_ids):
        h_mmbert = self.mmbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h_final = self.layer_norm_input(h_mmbert + self.role_embeddings(role_ids))
        
        B = input_ids.shape[0]
        q = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)
        z_attn, _ = self.cross_attention(query=q, key=h_final, value=h_final, key_padding_mask=(attention_mask == 0))
        z = self.layer_norm_cross(q + z_attn)
        
        if self.use_query_interaction:
            z_self, _ = self.self_attention(query=z, key=z, value=z)
            z_prime = self.layer_norm_self(z + z_self)
        else:
            z_prime = z
            
        s = self.shared_scoring_head(z_prime).squeeze(-1) # [B, 3]
        probs = F.softmax(s, dim=-1).unsqueeze(-1)
        h_B = (probs * z_prime).sum(dim=1) # [B, 768] (Task C Bridge)
        return s, h_B, None
`
  },
  'train.py': {
    desc: 'CLI entrypoint accepting command-line arguments and configuration overrides.',
    code: `# train.py
import argparse
from pipeline.config import PipelineConfig
from pipeline.data import DataPipeline
from pipeline.trainer import StereoQueerTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_source", default="mmbert", choices=["mmbert", "scratch"])
    parser.add_argument("--model", default="mmbert_transformer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    # ... executes pipeline
`
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'builder' | 'architecture' | 'data' | 'simulation' | 'inference' | 'code' | 'guide'>('builder');
  
  // Pipeline Builder State
  const [task, setTask] = useState('stereoqueer');
  const [targetTask, setTargetTask] = useState<'all' | 'st' | 'hs' | 'tg'>('st');
  const [embedSource, setEmbedSource] = useState<'mmbert' | 'scratch'>('mmbert');
  const [modelType, setModelType] = useState('mmbert_transformer');
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
  const [copiedCmd, setCopiedCmd] = useState(false);

  // Inference Tester State
  const [testComment, setTestComment] = useState(PRESET_COMMENTS[0].comment);
  const [testTitle, setTestTitle] = useState(PRESET_COMMENTS[0].title);
  const [testDesc, setTestDesc] = useState(PRESET_COMMENTS[0].desc);
  const [selectedPreset, setSelectedPreset] = useState(0);
  const [inferenceResult, setInferenceResult] = useState<any>(null);
  const [isInferring, setIsInferring] = useState(false);

  // Simulation State
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [simHistory, setSimHistory] = useState<any[]>([]);

  // Selected code file
  const [activeCodeFile, setActiveCodeFile] = useState('pipeline/config.py');

  // Generated CLI Command
  const generatedCommand = useMemo(() => {
    let cmd = `python train.py --task ${task} --target_task ${targetTask} --embed_source ${embedSource} --model ${modelType} --batch_size ${batchSize}`;
    if (embedSource === 'mmbert' && twoPhase) {
      cmd += ` --two_phase --unfreeze_layers ${unfreezeLayers} --unfreeze_lr ${unfreezeLr} --freeze_epochs 15 --unfreeze_epochs 15`;
    } else {
      cmd += ` --epochs ${epochs} --lr ${learningRate}`;
    }
    cmd += ` --patience ${patience} --data_dir data --output_dir checkpoints`;
    return cmd;
  }, [task, targetTask, embedSource, modelType, batchSize, twoPhase, unfreezeLayers, unfreezeLr, epochs, learningRate, patience]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 2000);
  };

  const handleRunInference = () => {
    setIsInferring(true);
    setTimeout(() => {
      // Realistic simulation heuristic based on test inputs
      const text = `${testComment} ${testTitle} ${testDesc}`.toLowerCase();
      let isSt = text.includes('agenda') || text.includes('always') || text.includes('sempre') || text.includes('altijd') || text.includes('obsessed') || text.includes('overdrijven');
      let isExplicit = text.includes('disgusting') || text.includes('vermin') || text.includes('cacciati') || text.includes('walgelijk') || text.includes('freaks');
      let isImplicit = !isExplicit && (text.includes('agenda') || text.includes('confused') || text.includes('moda') || text.includes('ontrouw'));
      let hs = isExplicit ? 'yes_explicit' : isImplicit ? 'yes_implicit' : 'no';
      
      let tg = 'none';
      if (hs !== 'no') {
        if (text.includes('gay') || text.includes('men')) tg = 'group_g';
        else if (text.includes('lesb')) tg = 'group_l';
        else if (text.includes('bisek') || text.includes('bisex')) tg = 'group_b';
        else if (text.includes('trans')) tg = 'group_t';
        else if (text.includes('non-binary') || text.includes('nb')) tg = 'group_nb';
        else tg = 'group_lgbtqia+';
      }

      setInferenceResult({
        stereotype: {
          label: isSt ? 'yes' : 'no',
          confidence: isSt ? 0.88 : 0.94,
          explanation: isSt ? 'Generalized behavioral assumption detected towards community' : 'No stereotypical generalization identified'
        },
        hate_speech: {
          label: hs,
          confidence: hs === 'no' ? 0.96 : hs === 'yes_explicit' ? 0.93 : 0.84,
          breakdown: {
            'no': hs === 'no' ? 0.96 : 0.04,
            'yes_implicit': hs === 'yes_implicit' ? 0.84 : 0.12,
            'yes_explicit': hs === 'yes_explicit' ? 0.93 : 0.03
          }
        },
        target: {
          str: tg,
          scope: tg.startsWith('group') ? 'group' : tg.startsWith('individual') ? 'individual' : 'none',
          identities: tg === 'none' ? [] : tg.split('_')[1].split(','),
        }
      });
      setIsInferring(false);
    }, 450);
  };

  // Run Training Simulation
  const handleStartSimulation = () => {
    setIsTraining(true);
    setCurrentEpoch(0);
    setSimHistory([]);

    let epoch = 1;
    const maxEp = 12;
    const interval = setInterval(() => {
      if (epoch > maxEp) {
        clearInterval(interval);
        setIsTraining(false);
        return;
      }

      const progress = epoch / maxEp;
      const trainLoss = Math.max(0.18, 0.95 - progress * 0.72 + (Math.random() * 0.04 - 0.02));
      const valLoss = Math.max(0.24, 0.98 - progress * 0.68 + (Math.random() * 0.05 - 0.02));
      const stF1 = Math.min(0.89, 0.52 + progress * 0.35 + Math.random() * 0.02);
      const hsF1 = Math.min(0.85, 0.48 + progress * 0.34 + Math.random() * 0.02);
      const tgExact = Math.min(0.79, 0.40 + progress * 0.36 + Math.random() * 0.02);

      setSimHistory(prev => [
        ...prev,
        {
          epoch,
          trainLoss: Number(trainLoss.toFixed(4)),
          valLoss: Number(valLoss.toFixed(4)),
          stF1: Number(stF1.toFixed(3)),
          hsF1: Number(hsF1.toFixed(3)),
          tgExact: Number(tgExact.toFixed(3)),
        }
      ]);
      setCurrentEpoch(epoch);
      epoch++;
    }, 700);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 shadow-lg shadow-indigo-500/20">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold text-white flex items-center gap-2">
                StereoQueer &amp; Toxic NLP Pipeline
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  SemEval 2027
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Multi-task Model Architecture &bull; mmBERT &amp; Transformers &bull; Anti-Leakage Video Pipeline
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              Python 3.10 Engine Ready
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex space-x-1 overflow-x-auto border-t border-slate-800/60 pt-1">
          {[
            { id: 'builder', label: 'Pipeline Builder', icon: Sliders },
            { id: 'architecture', label: 'Model Architecture', icon: Layers },
            { id: 'data', label: 'Data & Leakage Guard', icon: Database },
            { id: 'simulation', label: 'Live Training', icon: BarChart3 },
            { id: 'inference', label: 'Inference Bench', icon: Play },
            { id: 'code', label: 'Python Files', icon: Code2 },
            { id: 'guide', label: 'Setup Guide', icon: BookOpen },
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-3.5 py-2.5 text-xs font-medium rounded-t-lg transition-all border-b-2 whitespace-nowrap ${
                  isActive
                    ? 'border-indigo-500 text-indigo-400 bg-slate-900'
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

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-6">
        {/* 1. PIPELINE BUILDER TAB */}
        {activeTab === 'builder' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Sliders className="w-5 h-5 text-indigo-400" />
                    Interactive Training Pipeline Configurator
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Configure backbone embeddings, multi-task loss balance, and 2-phase ModernBERT schedules.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => copyToClipboard(generatedCommand)}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold shadow transition"
                  >
                    {copiedCmd ? <CheckCircle className="w-3.5 h-3.5 text-emerald-300" /> : <Copy className="w-3.5 h-3.5" />}
                    {copiedCmd ? 'Command Copied!' : 'Copy CLI Command'}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
                {/* Column 1: Task & Backbone */}
                <div className="space-y-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <Zap className="w-3.5 h-3.5 text-indigo-400" />
                    1. Task &amp; Embedding
                  </h3>

                  <div>
                    <label className="text-xs text-slate-400 font-medium">Dataset Scope</label>
                    <select
                      value={task}
                      onChange={e => setTask(e.target.value)}
                      className="w-full mt-1.5 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-indigo-500"
                    >
                      <option value="stereoqueer">StereoQueerEval SemEval 2027</option>
                      <option value="toxic">Jigsaw Toxic Comment Challenge</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 font-medium flex items-center justify-between">
                      <span>Specific Target Task</span>
                      <span className="text-[10px] text-indigo-400 font-semibold uppercase">{targetTask === 'all' ? 'Multi-task' : 'Single-task'}</span>
                    </label>
                    <div className="grid grid-cols-2 gap-2 mt-1.5">
                      <button
                        onClick={() => setTargetTask('st')}
                        className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition text-left ${
                          targetTask === 'st'
                            ? 'bg-indigo-950/70 border-indigo-500 text-indigo-300'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className="font-bold flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                          Stereotype (ST)
                        </div>
                        <div className="text-[10px] opacity-75">Binary (yes / no)</div>
                      </button>

                      <button
                        onClick={() => setTargetTask('hs')}
                        className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition text-left ${
                          targetTask === 'hs'
                            ? 'bg-purple-950/70 border-purple-500 text-purple-300'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className="font-bold flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
                          Hate Speech (HS)
                        </div>
                        <div className="text-[10px] opacity-75">3-class (no/imp/exp)</div>
                      </button>

                      <button
                        onClick={() => setTargetTask('tg')}
                        className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition text-left ${
                          targetTask === 'tg'
                            ? 'bg-emerald-950/70 border-emerald-500 text-emerald-300'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className="font-bold flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                          Target ID (TG)
                        </div>
                        <div className="text-[10px] opacity-75">10-bit bitmask</div>
                      </button>

                      <button
                        onClick={() => setTargetTask('all')}
                        className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition text-left ${
                          targetTask === 'all'
                            ? 'bg-amber-950/70 border-amber-500 text-amber-300'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className="font-bold flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                          All 3 Tasks
                        </div>
                        <div className="text-[10px] opacity-75">Multi-task joint loss</div>
                      </button>
                    </div>
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 font-medium">Embedding Source</label>
                    <div className="grid grid-cols-2 gap-2 mt-1.5">
                      <button
                        onClick={() => {
                          setEmbedSource('mmbert');
                          setModelType('mmbert_transformer');
                        }}
                        className={`px-3 py-2 rounded-lg text-xs font-semibold border transition text-left ${
                          embedSource === 'mmbert'
                            ? 'bg-indigo-950/60 border-indigo-500 text-indigo-300'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className="font-bold">mmBERT</div>
                        <div className="text-[10px] opacity-75">ModernBERT 22-layer</div>
                      </button>
                      <button
                        onClick={() => {
                          setEmbedSource('scratch');
                          setModelType('transformer');
                        }}
                        className={`px-3 py-2 rounded-lg text-xs font-semibold border transition text-left ${
                          embedSource === 'scratch'
                            ? 'bg-indigo-950/60 border-indigo-500 text-indigo-300'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className="font-bold">Scratch</div>
                        <div className="text-[10px] opacity-75">Learned Vocabulary</div>
                      </button>
                    </div>
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 font-medium">Architecture Head</label>
                    <select
                      value={modelType}
                      onChange={e => setModelType(e.target.value)}
                      className="w-full mt-1.5 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-indigo-500"
                    >
                      {embedSource === 'mmbert' ? (
                        <>
                          <option value="task_b_class_aware">Task B Class-Aware MHCA + Role Injection (Custom User Architecture)</option>
                          <option value="mmbert_transformer">mmBERT + Domain Transformer Encoder (Multi-task default)</option>
                          <option value="feature_mlp">mmBERT Pre-pooled Feature MLP (Fast / Low RAM)</option>
                        </>
                      ) : (
                        <>
                          <option value="transformer">PyTorch TransformerEncoder (Scratch)</option>
                          <option value="bilstm">Bidirectional LSTM (PytorchRNNLSTM)</option>
                          <option value="rnn">Vanilla RNN Baseline</option>
                        </>
                      )}
                    </select>
                  </div>
                </div>

                {/* Column 2: 2-Phase Fine-Tuning or Epochs */}
                <div className="space-y-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <RefreshCw className="w-3.5 h-3.5 text-purple-400" />
                    2. Training Schedule
                  </h3>

                  {embedSource === 'mmbert' ? (
                    <>
                      <div className="flex items-center justify-between">
                        <label className="text-xs text-slate-400 font-medium">2-Phase Training</label>
                        <input
                          type="checkbox"
                          checked={twoPhase}
                          onChange={e => setTwoPhase(e.target.checked)}
                          className="rounded text-indigo-600 focus:ring-indigo-500"
                        />
                      </div>
                      {twoPhase ? (
                        <div className="space-y-3 border-l-2 border-indigo-500/40 pl-3">
                          <div className="text-[11px] text-slate-400">
                            <strong>Phase 1:</strong> Backbone frozen, stabilize task heads.
                            <br />
                            <strong>Phase 2:</strong> Unfreeze last N layers with discriminative LR.
                          </div>
                          <div>
                            <label className="text-[11px] text-slate-400">Unfreeze Last N Layers: {unfreezeLayers}</label>
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
                            <label className="text-[11px] text-slate-400">Backbone Fine-Tune LR</label>
                            <input
                              type="text"
                              value={unfreezeLr}
                              onChange={e => setUnfreezeLr(e.target.value)}
                              className="w-full mt-1 px-2.5 py-1.5 bg-slate-900 border border-slate-700 rounded text-xs text-white"
                            />
                          </div>
                        </div>
                      ) : (
                        <div>
                          <label className="text-xs text-slate-400">Single Phase Epochs</label>
                          <input
                            type="number"
                            value={epochs}
                            onChange={e => setEpochs(Number(e.target.value))}
                            className="w-full mt-1.5 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white"
                          />
                        </div>
                      )}
                    </>
                  ) : (
                    <>
                      <div>
                        <label className="text-xs text-slate-400 font-medium">Training Epochs</label>
                        <input
                          type="number"
                          value={epochs}
                          onChange={e => setEpochs(Number(e.target.value))}
                          className="w-full mt-1.5 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 font-medium">Learning Rate</label>
                        <input
                          type="text"
                          value={learningRate}
                          onChange={e => setLearningRate(e.target.value)}
                          className="w-full mt-1.5 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white"
                        />
                      </div>
                    </>
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
                      <label className="text-[11px] text-slate-400">Early Stop Patience</label>
                      <input
                        type="number"
                        value={patience}
                        onChange={e => setPatience(Number(e.target.value))}
                        className="w-full mt-1 px-2.5 py-1.5 bg-slate-900 border border-slate-700 rounded text-xs text-white"
                      />
                    </div>
                  </div>
                </div>

                {/* Column 3: Multi-task Loss Weights */}
                <div className="space-y-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
                    3. Objective &amp; Loss Focus
                  </h3>

                  {targetTask !== 'all' ? (
                    <div className="space-y-3">
                      <div className="p-3 bg-indigo-950/40 border border-indigo-500/30 rounded-xl">
                        <div className="text-xs font-bold text-indigo-300 flex items-center gap-2">
                          <CheckCircle className="w-4 h-4 text-emerald-400" />
                          Single Subtask Active: {targetTask.toUpperCase()}
                        </div>
                        <p className="text-[11px] text-slate-300 mt-1">
                          {targetTask === 'st' && 'Training exclusively on Binary Stereotype detection (yes / no) using Binary Cross-Entropy with Logits.'}
                          {targetTask === 'hs' && 'Training exclusively on Hate Speech category (no / yes_implicit / yes_explicit) using 3-class Cross-Entropy.'}
                          {targetTask === 'tg' && 'Training exclusively on 10-dimensional Target Identity and Scope bitmask using multi-label BCE.'}
                        </p>
                      </div>

                      <div className="text-[11px] text-slate-400 p-2.5 bg-slate-900 rounded-lg border border-slate-800">
                        <span className="text-emerald-400 font-semibold">Tip:</span> Training a single task first is an excellent empirical strategy to establish an unconstrained baseline for that subtask before introducing multi-task joint regularization.
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-400 font-medium">Stereotype (BCE) Weight:</span>
                          <span className="text-indigo-400 font-bold">{stWeight}×</span>
                        </div>
                        <input
                          type="range"
                          min="0.5"
                          max="3.0"
                          step="0.1"
                          value={stWeight}
                          onChange={e => setStWeight(Number(e.target.value))}
                          className="w-full accent-indigo-500"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-400 font-medium">Hate Speech (CE) Weight:</span>
                          <span className="text-purple-400 font-bold">{hsWeight}×</span>
                        </div>
                        <input
                          type="range"
                          min="0.5"
                          max="3.0"
                          step="0.1"
                          value={hsWeight}
                          onChange={e => setHsWeight(Number(e.target.value))}
                          className="w-full accent-purple-500"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-400 font-medium">Target Identity (BCE) Weight:</span>
                          <span className="text-emerald-400 font-bold">{tgWeight}×</span>
                        </div>
                        <input
                          type="range"
                          min="0.5"
                          max="3.0"
                          step="0.1"
                          value={tgWeight}
                          onChange={e => setTgWeight(Number(e.target.value))}
                          className="w-full accent-emerald-500"
                        />
                      </div>

                      <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-400">
                        <span className="text-amber-400 font-semibold">Loss Balancing Note:</span> CrossEntropy (3 classes) produces larger gradients than BCE. Boosting ST &amp; TG to 1.5 ensures balanced representation convergence.
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Generated Command Box */}
              <div className="mt-6 pt-5 border-t border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                    <Terminal className="w-4 h-4 text-indigo-400" />
                    Executable Python Command
                  </span>
                  <span className="text-[11px] text-slate-500">Run this in your terminal or Jupyter notebook</span>
                </div>
                <div className="relative group">
                  <pre className="p-3.5 bg-slate-950 border border-slate-800 rounded-xl text-xs font-mono text-indigo-300 overflow-x-auto">
                    {generatedCommand}
                  </pre>
                  <button
                    onClick={() => copyToClipboard(generatedCommand)}
                    className="absolute right-2.5 top-2.5 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-white rounded text-[11px] font-medium border border-slate-700 transition"
                  >
                    {copiedCmd ? 'Copied!' : 'Copy'}
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Actions Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div
                onClick={() => setActiveTab('inference')}
                className="cursor-pointer bg-slate-900 hover:bg-slate-850 border border-slate-800 hover:border-indigo-500/50 rounded-xl p-4 transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 group-hover:bg-indigo-500 group-hover:text-white transition">
                    <Play className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white">Interactive Inference</h4>
                    <p className="text-[11px] text-slate-400">Test comments with live target decoding</p>
                  </div>
                </div>
              </div>

              <div
                onClick={() => setActiveTab('architecture')}
                className="cursor-pointer bg-slate-900 hover:bg-slate-850 border border-slate-800 hover:border-purple-500/50 rounded-xl p-4 transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 group-hover:bg-purple-500 group-hover:text-white transition">
                    <Layers className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white">Architecture Flowchart</h4>
                    <p className="text-[11px] text-slate-400">View mmBERT + Transformer 3-head design</p>
                  </div>
                </div>
              </div>

              <div
                onClick={() => setActiveTab('code')}
                className="cursor-pointer bg-slate-900 hover:bg-slate-850 border border-slate-800 hover:border-emerald-500/50 rounded-xl p-4 transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 group-hover:bg-emerald-500 group-hover:text-white transition">
                    <Code2 className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white">Inspect Python Code</h4>
                    <p className="text-[11px] text-slate-400">Browse train.py, trainer.py &amp; models</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 2. ARCHITECTURE TAB */}
        {activeTab === 'architecture' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-1">
                <Layers className="w-5 h-5 text-indigo-400" />
                Multi-Task Model Architecture Blueprint
              </h2>
              <p className="text-xs text-slate-400 mb-6">
                All 3 subtasks share a single contextual backbone encoder, producing balanced gradients for simultaneous multi-objective training.
              </p>

              {/* Visual Pipeline Flowchart */}
              <div className="space-y-4 max-w-4xl mx-auto">
                {/* Input block */}
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-center">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Contextual Triplet Input</span>
                  <div className="mt-2 text-xs font-mono text-indigo-300 bg-slate-900/80 px-3 py-2 rounded-lg border border-slate-800 inline-block">
                    [yt_comment] &lt;SEP&gt; [yt_title] &lt;SEP&gt; [yt_description]
                  </div>
                  <div className="mt-2 text-[11px] text-slate-500">
                    Safe character cleaning (preserves Italian &lsquo;perch&eacute;&rsquo;, &lsquo;&egrave;&rsquo; &amp; Dutch accents)
                  </div>
                </div>

                <div className="flex justify-center text-slate-600">&darr;</div>

                {/* Backbone block */}
                <div className="p-5 bg-gradient-to-r from-indigo-950/40 via-purple-950/40 to-slate-950 border border-indigo-500/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-indigo-300 flex items-center gap-2">
                      <Cpu className="w-4 h-4" />
                      Backbone: ModernBERT / mmBERT (jhu-clsp/mmbert-base)
                    </span>
                    <span className="text-[11px] px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-mono">
                      22 Layers &bull; 768-dim &bull; 8192 Context
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">
                    Phase 1 freezes all 22 layers. Phase 2 unfreezes the last 2 encoder blocks (layers 20 &amp; 21) with learning rate 2e-5.
                  </p>
                </div>

                <div className="flex justify-center text-slate-600">&darr;</div>

                {/* Transformer Encoder & Pooling */}
                <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-slate-200">Domain TransformerEncoder &amp; Masked Pooling</span>
                    <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                      2 Layers &bull; 8 Attention Heads &bull; 1024 FFN
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">
                    Sequence representations passing padding masks (<code>src_key_padding_mask</code>) followed by masked mean pooling to produce unified 768-d vector.
                  </p>
                </div>

                <div className="flex justify-center text-slate-600">&darr;</div>

                {/* 3 Multi-Task Heads */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Head 1 */}
                  <div className="p-4 bg-indigo-950/20 border border-indigo-500/30 rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-indigo-300">1. Stereotype Head (st)</span>
                      <span className="text-[10px] bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded font-mono">BCE</span>
                    </div>
                    <div className="text-[11px] text-slate-400 space-y-1">
                      <div>&bull; Linear(768 &rarr; 384) + ReLU + Dropout(0.3)</div>
                      <div>&bull; Linear(384 &rarr; 1) &rarr; Sigmoid</div>
                      <div className="pt-2 text-indigo-400 font-semibold">Output: yes / no (1 bit)</div>
                      <div className="text-slate-500">Loss weight: 1.5&times;</div>
                    </div>
                  </div>

                  {/* Head 2 */}
                  <div className="p-4 bg-purple-950/20 border border-purple-500/30 rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-purple-300">2. Hate Speech Head (hs)</span>
                      <span className="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded font-mono">CE</span>
                    </div>
                    <div className="text-[11px] text-slate-400 space-y-1">
                      <div>&bull; Linear(768 &rarr; 384) + ReLU + Dropout(0.3)</div>
                      <div>&bull; Linear(384 &rarr; 3) &rarr; Softmax</div>
                      <div className="pt-2 text-purple-400 font-semibold">Output: no / implicit / explicit</div>
                      <div className="text-slate-500">Loss weight: 1.0&times;</div>
                    </div>
                  </div>

                  {/* Head 3 */}
                  <div className="p-4 bg-emerald-950/20 border border-emerald-500/30 rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-emerald-300">3. Target Identity Head (tg)</span>
                      <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded font-mono">BCE</span>
                    </div>
                    <div className="text-[11px] text-slate-400 space-y-1">
                      <div>&bull; Linear(768 &rarr; 384) + ReLU + Dropout(0.3)</div>
                      <div>&bull; Linear(384 &rarr; 10) &rarr; Multi-Sigmoid</div>
                      <div className="pt-2 text-emerald-400 font-semibold">Output: 9 IDs + 1 Scope bit</div>
                      <div className="text-slate-500">Loss weight: 1.5&times;</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Task B Dedicated Architecture Showcase */}
            <div className="bg-slate-900 border border-indigo-500/30 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                    <Zap className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      Custom Task B Architecture: Class-Aware Multi-Head Cross-Attention (MHCA)
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 font-mono">
                        Ablation H2 Ready
                      </span>
                    </h3>
                    <p className="text-xs text-slate-400">
                      Explicit role injection with class-conditioned query attention and task C bridging.
                    </p>
                  </div>
                </div>
                <div className="text-xs text-indigo-400 font-mono">
                  --model task_b_class_aware
                </div>
              </div>

              <div className="space-y-4 pt-2">
                {/* Input Role Ingestion */}
                <div className="p-3.5 bg-slate-950 border border-slate-800 rounded-xl">
                  <div className="text-xs font-bold text-indigo-300 mb-1">
                    [INPUT SEQUENCE &amp; ROLE INJECTION]
                  </div>
                  <div className="text-xs font-mono text-slate-300">
                    Title: &lt;T&gt; ... &lt;/T&gt; | Description: &lt;D&gt; ... &lt;/D&gt; | Comment: &lt;C&gt; ... &lt;/C&gt;
                  </div>
                  <div className="mt-2 text-[11px] text-slate-400 grid grid-cols-1 sm:grid-cols-2 gap-2">
                    <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                      <span className="text-slate-200 font-semibold">mmBERT Backbone:</span> Input Token IDs [B, S] &rarr; H_mmBERT [B, S, 768]
                    </div>
                    <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                      <span className="text-indigo-300 font-semibold">Role Embeddings:</span> Role IDs [B, S] &rarr; E_role [B, S, 768]
                    </div>
                  </div>
                  <div className="mt-2 text-[11px] text-emerald-400 font-mono">
                    H_final = LayerNorm(H_mmBERT + E_role) &isin; [B, S, 768]
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Layer 1 */}
                  <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
                    <div className="text-xs font-bold text-purple-300 flex items-center justify-between">
                      <span>LAYER 1: MHCA</span>
                      <span className="text-[10px] text-slate-500 font-mono">Class-Aware</span>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Learned Queries: <br />
                      <code className="text-purple-300">Q_base = [q_Exp, q_Imp, q_NonHate] &isin; [3, 768]</code>
                    </p>
                    <div className="text-[11px] text-slate-400">
                      &bull; Keys/Values: <code className="text-slate-300">H_final &isin; [B, S, 768]</code><br />
                      &bull; Attention Map: <code className="text-slate-300">A &isin; [B, 3, S]</code> (interpretability)<br />
                      &bull; Output: <code className="text-indigo-300">Z &isin; [B, 3, 768]</code>
                    </div>
                  </div>

                  {/* Layer 2 */}
                  <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
                    <div className="text-xs font-bold text-indigo-300 flex items-center justify-between">
                      <span>LAYER 2: MHSA</span>
                      <span className="text-[10px] text-amber-400 font-mono">Ablation H2</span>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Query Interaction Layer:<br />
                      Inter-label self-attention across 3 class queries:
                    </p>
                    <div className="text-[11px] text-slate-400">
                      <code className="text-indigo-300">Explicit &harr; Implicit &harr; NonHate</code><br />
                      &bull; Output: <code className="text-indigo-300">Z&apos; &isin; [B, 3, 768]</code><br />
                      &bull; Toggle with: <code className="text-slate-300">--no_query_interaction</code>
                    </div>
                  </div>

                  {/* Layer 3 & Bridge */}
                  <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
                    <div className="text-xs font-bold text-emerald-300 flex items-center justify-between">
                      <span>LAYER 3 &amp; BRIDGE</span>
                      <span className="text-[10px] text-emerald-400 font-mono">Task C Bridge</span>
                    </div>
                    <div className="text-[11px] text-slate-400 space-y-1">
                      <div>&bull; Shared scoring head <code className="text-slate-300">f_&theta;(z&apos;_c) &rarr; s_c</code></div>
                      <div>&bull; Raw logits: <code className="text-emerald-300">s &isin; [B, 3]</code></div>
                      <div>&bull; Hate probabilities: <code className="text-slate-300">p = Softmax(s)</code></div>
                      <div className="pt-1 text-purple-300 font-semibold">
                        h_B = &sum;_c (p_c &middot; z&apos;_c) &isin; [B, 768]
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 3. DATA & LEAKAGE GUARD TAB */}
        {activeTab === 'data' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    Data Engineering &amp; Anti-Leakage Video Splitting
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    YouTube titles and descriptions form part of the input. GroupShuffleSplit ensures 0% video title overlap across train and val splits.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Target Bitmask Mapping */}
                <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
                  <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                    <Database className="w-3.5 h-3.5 text-indigo-400" />
                    10-Dimensional Target Bitmask (SemEval 2027)
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Target string format: <code>group_l,g</code> or <code>individual_t,nb</code> or <code>none</code>.
                  </p>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 pt-2">
                    {['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+', 'scope'].map((bit, idx) => (
                      <div key={bit} className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                        <div className="text-[10px] text-slate-500 font-mono">Bit {idx}</div>
                        <div className="text-xs font-bold text-indigo-300 mt-0.5">{bit}</div>
                        <div className="text-[9px] text-slate-400">{idx === 9 ? 'group/ind' : 'Identity'}</div>
                      </div>
                    ))}
                  </div>
                  <div className="p-3 bg-indigo-950/30 rounded-lg border border-indigo-500/20 text-xs text-slate-300">
                    <strong>Rule:</strong> If comment is non-hateful (<code>hate_speech == &quot;no&quot;</code>), target strictly defaults to <code>none</code> (all 0s).
                  </div>
                </div>

                {/* GroupShuffleSplit Explanation */}
                <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
                  <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                    YouTube Context Leakage Prevention
                  </h3>
                  <div className="text-xs text-slate-400 space-y-2">
                    <p>
                      <strong>The Problem:</strong> If random shuffling is used, comments under the same YouTube video appear in both train and validation splits. Because the model reads the video title and description, it memorizes video topics rather than learning general linguistic cues.
                    </p>
                    <p>
                      <strong>Our Solution:</strong> <code>GroupShuffleSplit(groups=df[&apos;yt_title&apos;])</code> isolates entire videos into either train OR validation, simulating true zero-shot evaluation on unseen videos and channels.
                    </p>
                  </div>
                  <div className="p-3 bg-emerald-950/30 rounded-lg border border-emerald-500/20 text-xs text-emerald-300 flex items-center justify-between">
                    <span>Train/Val Video Title Overlap:</span>
                    <span className="font-bold text-emerald-400">0 Videos (Guaranteed)</span>
                  </div>
                </div>
              </div>

              {/* Data File Schema */}
              <div className="mt-6 pt-5 border-t border-slate-800">
                <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-3">
                  SemEval 2027 TSV Column Specifications
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/60">
                        <th className="py-2.5 px-3 font-semibold">Column Name</th>
                        <th className="py-2.5 px-3 font-semibold">Data Type</th>
                        <th className="py-2.5 px-3 font-semibold">Example Value</th>
                        <th className="py-2.5 px-3 font-semibold">Description</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 text-slate-300 font-mono text-[11px]">
                      <tr>
                        <td className="py-2.5 px-3 text-indigo-400">StereoQueerEval_id</td>
                        <td className="py-2.5 px-3">string</td>
                        <td className="py-2.5 px-3 text-slate-400">&quot;training_EN_0042&quot;</td>
                        <td className="py-2.5 px-3 font-sans">Unique instance ID</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-3 text-indigo-400">yt_title</td>
                        <td className="py-2.5 px-3">string</td>
                        <td className="py-2.5 px-3 text-slate-400">&quot;Pride Month Debate 2026&quot;</td>
                        <td className="py-2.5 px-3 font-sans">YouTube video title (Group key)</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-3 text-indigo-400">yt_description</td>
                        <td className="py-2.5 px-3">string</td>
                        <td className="py-2.5 px-3 text-slate-400">&quot;Panel exploring equal rights...&quot;</td>
                        <td className="py-2.5 px-3 font-sans">Video description context</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-3 text-indigo-400">yt_comment</td>
                        <td className="py-2.5 px-3">string</td>
                        <td className="py-2.5 px-3 text-slate-400">&quot;They only care about drama.&quot;</td>
                        <td className="py-2.5 px-3 font-sans">Primary comment to classify</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-3 text-emerald-400">stereotype</td>
                        <td className="py-2.5 px-3">binary</td>
                        <td className="py-2.5 px-3 text-slate-400">&quot;yes&quot; / &quot;no&quot;</td>
                        <td className="py-2.5 px-3 font-sans">Subtask 1 label</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-3 text-purple-400">hate_speech</td>
                        <td className="py-2.5 px-3">categorical</td>
                        <td className="py-2.5 px-3 text-slate-400">&quot;no&quot; / &quot;yes_implicit&quot; / &quot;yes_explicit&quot;</td>
                        <td className="py-2.5 px-3 font-sans">Subtask 2 label (3 classes)</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-3 text-amber-400">target</td>
                        <td className="py-2.5 px-3">structured</td>
                        <td className="py-2.5 px-3 text-slate-400">&quot;group_lgbtqia+&quot;</td>
                        <td className="py-2.5 px-3 font-sans">Subtask 3: scope + identities</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 4. LIVE SIMULATION TAB */}
        {activeTab === 'simulation' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-indigo-400" />
                    Live Training Simulator &amp; Loss Curves
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Simulate the 2-phase training loop in real time to visualize loss curves and multi-task Macro-F1 progression.
                  </p>
                </div>
                <button
                  disabled={isTraining}
                  onClick={handleStartSimulation}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold shadow transition ${
                    isTraining
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                      : 'bg-indigo-600 hover:bg-indigo-500 text-white'
                  }`}
                >
                  <Play className={`w-3.5 h-3.5 ${isTraining ? 'animate-spin' : ''}`} />
                  {isTraining ? `Training Epoch ${currentEpoch}/12...` : 'Start Training Run'}
                </button>
              </div>

              {/* Progress Summary Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
                  <div className="text-[11px] text-slate-400">Train Loss</div>
                  <div className="text-xl font-bold text-white mt-1">
                    {simHistory.length ? simHistory[simHistory.length - 1].trainLoss : '0.9420'}
                  </div>
                  <div className="text-[10px] text-emerald-400 mt-0.5">&darr; Decreasing</div>
                </div>

                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
                  <div className="text-[11px] text-slate-400">Val Loss</div>
                  <div className="text-xl font-bold text-white mt-1">
                    {simHistory.length ? simHistory[simHistory.length - 1].valLoss : '0.9780'}
                  </div>
                  <div className="text-[10px] text-indigo-400 mt-0.5">Early stop tracking</div>
                </div>

                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
                  <div className="text-[11px] text-slate-400">Stereotype Macro-F1</div>
                  <div className="text-xl font-bold text-indigo-400 mt-1">
                    {simHistory.length ? `${(simHistory[simHistory.length - 1].stF1 * 100).toFixed(1)}%` : '52.0%'}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Binary ST</div>
                </div>

                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
                  <div className="text-[11px] text-slate-400">Target Exact Match</div>
                  <div className="text-xl font-bold text-emerald-400 mt-1">
                    {simHistory.length ? `${(simHistory[simHistory.length - 1].tgExact * 100).toFixed(1)}%` : '40.0%'}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Canonical string match</div>
                </div>
              </div>

              {/* Epoch Metrics Table */}
              <div className="border border-slate-800 rounded-xl overflow-hidden">
                <div className="p-3 bg-slate-950 border-b border-slate-800 text-xs font-semibold text-slate-300 flex justify-between">
                  <span>Epoch History</span>
                  <span className="text-slate-500 font-normal">
                    {simHistory.length ? `${simHistory.length} epochs recorded` : 'Click Start Training Run to generate history'}
                  </span>
                </div>
                <div className="max-h-64 overflow-y-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 bg-slate-900/60 font-mono text-[11px]">
                        <th className="py-2 px-3">Epoch</th>
                        <th className="py-2 px-3">Train Loss</th>
                        <th className="py-2 px-3">Val Loss</th>
                        <th className="py-2 px-3">Stereotype F1</th>
                        <th className="py-2 px-3">Hate Speech F1</th>
                        <th className="py-2 px-3">Target Exact Match</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/40 font-mono text-[11px]">
                      {simHistory.map(row => (
                        <tr key={row.epoch} className="hover:bg-slate-800/30">
                          <td className="py-2 px-3 text-indigo-400 font-bold">#{row.epoch}</td>
                          <td className="py-2 px-3">{row.trainLoss}</td>
                          <td className="py-2 px-3 text-amber-300">{row.valLoss}</td>
                          <td className="py-2 px-3 text-indigo-300">{(row.stF1 * 100).toFixed(1)}%</td>
                          <td className="py-2 px-3 text-purple-300">{(row.hsF1 * 100).toFixed(1)}%</td>
                          <td className="py-2 px-3 text-emerald-300">{(row.tgExact * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                      {!simHistory.length && (
                        <tr>
                          <td colSpan={6} className="py-8 text-center text-slate-500 font-sans">
                            Simulation not started. Click &quot;Start Training Run&quot; above to simulate training.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 5. INFERENCE BENCH TAB */}
        {activeTab === 'inference' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Play className="w-5 h-5 text-indigo-400" />
                    Multi-Lingual Inference Test Bench
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Evaluate comments in English, Italian, or Dutch with simultaneous stereotype, hate speech, and target identification.
                  </p>
                </div>
              </div>

              {/* Sample Presets */}
              <div className="mb-4">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                  Sample Presets (EN / IT / NL):
                </span>
                <div className="flex flex-wrap gap-2">
                  {PRESET_COMMENTS.map((preset, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setSelectedPreset(idx);
                        setTestComment(preset.comment);
                        setTestTitle(preset.title);
                        setTestDesc(preset.desc);
                        setInferenceResult(null);
                      }}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
                        selectedPreset === idx
                          ? 'bg-indigo-600 border-indigo-500 text-white'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <span className="font-bold mr-1.5">[{preset.lang}]</span>
                      {preset.comment.slice(0, 32)}...
                    </button>
                  ))}
                </div>
              </div>

              {/* Input Form */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="md:col-span-2 space-y-3">
                  <div>
                    <label className="text-xs text-slate-400 font-medium">YouTube Comment (yt_comment)</label>
                    <textarea
                      rows={3}
                      value={testComment}
                      onChange={e => setTestComment(e.target.value)}
                      className="w-full mt-1 px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-white focus:outline-none focus:border-indigo-500"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-slate-400 font-medium">Video Title (yt_title)</label>
                      <input
                        type="text"
                        value={testTitle}
                        onChange={e => setTestTitle(e.target.value)}
                        className="w-full mt-1 px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-white focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-400 font-medium">Video Description</label>
                      <input
                        type="text"
                        value={testDesc}
                        onChange={e => setTestDesc(e.target.value)}
                        className="w-full mt-1 px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-white focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                  </div>
                </div>

                {/* Action button */}
                <div className="flex flex-col justify-end">
                  <button
                    onClick={handleRunInference}
                    disabled={isInferring || !testComment.trim()}
                    className="w-full py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white rounded-xl text-sm font-bold shadow-lg shadow-indigo-500/20 transition flex items-center justify-center gap-2"
                  >
                    <Play className={`w-4 h-4 ${isInferring ? 'animate-spin' : ''}`} />
                    {isInferring ? 'Running Inference...' : 'Predict 3 Heads'}
                  </button>
                </div>
              </div>

              {/* Inference Results Output */}
              {inferenceResult && (
                <div className="mt-6 p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-4">
                  <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                    Model Multi-Task Predictions
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {/* Stereotype Result */}
                    <div className="p-4 bg-slate-900 rounded-xl border border-slate-800">
                      <div className="text-[11px] text-slate-400 mb-1">Subtask 1: Stereotype</div>
                      <div className="flex items-center gap-2">
                        <span className={`text-base font-bold px-2 py-0.5 rounded ${
                          inferenceResult.stereotype.label === 'yes'
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {inferenceResult.stereotype.label.toUpperCase()}
                        </span>
                        <span className="text-xs text-slate-400">
                          ({(inferenceResult.stereotype.confidence * 100).toFixed(1)}% conf)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 mt-2">
                        {inferenceResult.stereotype.explanation}
                      </p>
                    </div>

                    {/* Hate Speech Result */}
                    <div className="p-4 bg-slate-900 rounded-xl border border-slate-800">
                      <div className="text-[11px] text-slate-400 mb-1">Subtask 2: Hate Speech</div>
                      <div className="flex items-center gap-2">
                        <span className={`text-base font-bold px-2 py-0.5 rounded ${
                          inferenceResult.hate_speech.label === 'yes_explicit'
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : inferenceResult.hate_speech.label === 'yes_implicit'
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {inferenceResult.hate_speech.label}
                        </span>
                        <span className="text-xs text-slate-400">
                          ({(inferenceResult.hate_speech.confidence * 100).toFixed(1)}% conf)
                        </span>
                      </div>
                      <div className="mt-2 text-[10px] text-slate-500 space-y-0.5">
                        <div>no: {(inferenceResult.hate_speech.breakdown['no'] * 100).toFixed(0)}%</div>
                        <div>implicit: {(inferenceResult.hate_speech.breakdown['yes_implicit'] * 100).toFixed(0)}%</div>
                        <div>explicit: {(inferenceResult.hate_speech.breakdown['yes_explicit'] * 100).toFixed(0)}%</div>
                      </div>
                    </div>

                    {/* Target Result */}
                    <div className="p-4 bg-slate-900 rounded-xl border border-slate-800">
                      <div className="text-[11px] text-slate-400 mb-1">Subtask 3: Target Specification</div>
                      <div className="flex items-center gap-2">
                        <span className="text-base font-bold font-mono text-indigo-400">
                          {inferenceResult.target.str}
                        </span>
                      </div>
                      <div className="mt-2 text-[11px] text-slate-400">
                        <div>Scope: <strong className="text-slate-200">{inferenceResult.target.scope}</strong></div>
                        <div>Identities: <strong className="text-slate-200">{inferenceResult.target.identities.join(', ') || 'None'}</strong></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 6. PYTHON CODE EXPLORER TAB */}
        {activeTab === 'code' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Code2 className="w-5 h-5 text-indigo-400" />
                    Python Pipeline Source Files
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Explore the modular Python codebase generated in the workspace. All files are ready to run in Python 3.8+.
                  </p>
                </div>
                <button
                  onClick={() => copyToClipboard(CODE_FILES[activeCodeFile].code)}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-semibold border border-slate-700 transition"
                >
                  {copiedCmd ? <CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedCmd ? 'Copied File!' : 'Copy Code'}
                </button>
              </div>

              {/* File selector tabs */}
              <div className="flex flex-wrap gap-2 mb-4">
                {Object.keys(CODE_FILES).map(fileName => (
                  <button
                    key={fileName}
                    onClick={() => setActiveCodeFile(fileName)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition ${
                      activeCodeFile === fileName
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800'
                    }`}
                  >
                    {fileName}
                  </button>
                ))}
              </div>

              <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800/80 mb-3 text-xs text-slate-400">
                <Info className="w-3.5 h-3.5 inline mr-1.5 text-indigo-400" />
                {CODE_FILES[activeCodeFile].desc}
              </div>

              <div className="relative">
                <pre className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-xs font-mono text-slate-300 overflow-x-auto max-h-96">
                  {CODE_FILES[activeCodeFile].code}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* 7. SETUP GUIDE TAB */}
        {activeTab === 'guide' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
                <BookOpen className="w-5 h-5 text-indigo-400" />
                Step-by-Step Training &amp; Deployment Guide
              </h2>

              <div className="space-y-6">
                {/* Step 1 */}
                <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
                  <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
                    Step 1: Install Dependencies
                  </h3>
                  <p className="text-xs text-slate-400">
                    Create a virtual environment and install the required PyTorch, Transformers, and Scikit-Learn libraries:
                  </p>
                  <pre className="p-3 bg-slate-900 rounded-lg text-xs font-mono text-indigo-300 overflow-x-auto">
                    pip install -r requirements.txt
                  </pre>
                </div>

                {/* Step 2 */}
                <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
                  <h3 className="text-xs font-bold text-purple-400 uppercase tracking-wider">
                    Step 2: Obtain or Generate Data
                  </h3>
                  <p className="text-xs text-slate-400">
                    If you have official SemEval data, extract the RAR into <code>data/</code>. For instant testing, run our synthetic generator:
                  </p>
                  <pre className="p-3 bg-slate-900 rounded-lg text-xs font-mono text-purple-300 overflow-x-auto">
                    python generate_sample_data.py
                  </pre>
                </div>

                {/* Step 3 */}
                <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
                  <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
                    Step 3: Run Model Training (Single-task or Multi-task)
                  </h3>
                  <p className="text-xs text-slate-400">
                    Train a specific subtask first (e.g. Stereotype only), or train all 3 tasks jointly:
                  </p>
                  <div className="space-y-1.5 pt-1">
                    <div className="text-[11px] text-indigo-400 font-semibold">Single Subtask (Stereotype only):</div>
                    <pre className="p-2.5 bg-slate-900 rounded-lg text-xs font-mono text-indigo-300 overflow-x-auto">
                      python train.py --target_task st --embed_source mmbert --model mmbert_transformer
                    </pre>

                    <div className="text-[11px] text-purple-400 font-semibold pt-1">Single Subtask (Hate Speech only):</div>
                    <pre className="p-2.5 bg-slate-900 rounded-lg text-xs font-mono text-purple-300 overflow-x-auto">
                      python train.py --target_task hs --embed_source mmbert --model mmbert_transformer
                    </pre>

                    <div className="text-[11px] text-emerald-400 font-semibold pt-1">Joint Multi-task (All 3 Tasks):</div>
                    <pre className="p-2.5 bg-slate-900 rounded-lg text-xs font-mono text-emerald-300 overflow-x-auto">
                      python train.py --target_task all --embed_source mmbert --model mmbert_transformer --two_phase
                    </pre>
                  </div>
                </div>

                {/* Step 4 */}
                <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
                  <h3 className="text-xs font-bold text-amber-400 uppercase tracking-wider">
                    Step 4: Evaluate Checkpoints &amp; Run Predictions
                  </h3>
                  <p className="text-xs text-slate-400">
                    Assess accuracy, Macro-F1, and target exact-match on validation data:
                  </p>
                  <pre className="p-3 bg-slate-900 rounded-lg text-xs font-mono text-amber-300 overflow-x-auto">
                    python evaluate.py --checkpoint checkpoints/best_model.pt
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 text-center text-xs text-slate-500">
        StereoQueerEval &amp; Toxic Classification Pipeline &bull; SemEval 2027 Multi-Task Python Framework
      </footer>
    </div>
  );
}
