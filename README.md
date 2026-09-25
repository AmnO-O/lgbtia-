# StereoQueerEval & Toxic NLP Model Training Pipeline

A modular, production-ready Python training and evaluation pipeline for **StereoQueerEval (SemEval 2027)** and **Toxic Comment Classification**.

The pipeline detects queer stereotypes, hate speech, and target victim identities/scopes in YouTube comments across **English (EN)**, **Italian (IT)**, and **Dutch (NL)** using multi-task learning.

---

## 🏗️ Architecture & Multi-Task Design

The model processes a contextual triplet:
`comment [SEP] title [SEP] description`

Three subtask heads branch from a shared encoder:
| Subtask | Target | Type | Loss Function | Weight |
|---|---|---|---|---|
| **Stereotype (`st`)** | `yes` / `no` | Binary classification | `BCEWithLogitsLoss` | `1.5` |
| **Hate Speech (`hs`)** | `no` / `yes_implicit` / `yes_explicit` | 3-class classification | `CrossEntropyLoss` | `1.0` |
| **Target (`tg`)** | `none` or `scope + identities` (`l,g,b,t,q,i,a,nb,lgbtqia+`) | 10-dim bitmask (9 ID + 1 scope) | `BCEWithLogitsLoss` | `1.5` |

> **Note on Loss Weights**: Cross-entropy produces higher numerical magnitude than binary cross-entropy. Weights of `1.5 * st + 1.0 * hs + 1.5 * tg` ensure balanced multi-task gradient propagation.

### Supported Backbones & Models
1. **mmBERT / ModernBERT (`jhu-clsp/mmbert-base`)**:
   - ModernBERT 22-layer transformer encoder (768-d).
   - **2-Phase Fine-Tuning**: Phase 1 freezes the backbone to calibrate the heads and custom transformer layers. Phase 2 unfreezes the last $N$ layers (default: 2) with a discriminative learning rate (`2e-5`).
2. **PyTorch Transformer (Scratch)**:
   - Positional encoding + multi-layer `TransformerEncoder` + global average pooling.
3. **Bi-LSTM (`PytorchRNNLSTM`)**:
   - Bidirectional Long Short-Term Memory network capturing context forwards & backwards.
4. **Vanilla RNN (`VanillaRNNModel`)**:
   - Baseline recurrent network.
5. **Feature MLP (`FeatureClassifier`)**:
   - Fast multi-layer perceptron on pre-pooled embeddings.

---

## 📂 Project Structure

```
├── pipeline/                   # Core Python package
│   ├── __init__.py
│   ├── config.py               # PipelineConfig dataclass & hyperparams
│   ├── data.py                 # Multi-task dataset, GroupShuffleSplit, TSV parsing
│   ├── losses.py               # MultiTaskLoss with customizable weighting
│   ├── metrics.py              # Macro-F1, accuracy, exact-match target scoring
│   ├── trainer.py              # 2-phase trainer, early stopping, checkpointing
│   ├── inference.py            # StereoQueerPredictor engine
│   └── models/
│       ├── __init__.py
│       ├── rnn.py              # Vanilla RNN baseline
│       ├── lstm.py             # Bidirectional LSTM model
│       ├── transformer.py      # Custom & PyTorch Transformer models
│       ├── mmbert.py           # ModernBERT / mmBERT backbone + heads
│       └── classifier.py       # Classification heads & MLP
├── train.py                    # Training CLI entrypoint
├── evaluate.py                 # Model evaluation & per-language breakdown
├── predict.py                  # Single-sample inference CLI
├── generate_sample_data.py     # Synthetic sample TSV generator for testing
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installer
└── README.md                   # Documentation
```

---

## ⚡ Quick Start

### 1. Installation
```bash
# In your virtual environment or Conda environment:
pip install -r requirements.txt
pip install -e .
```

### 2. Generate Sample Data (or Place Official TSVs in `data/`)
```bash
python generate_sample_data.py
```
This generates `StereoQueerEval_{EN,IT,NL}_training.tsv` in the `data/` folder.

If you have the official SemEval dataset, extract the password-protected RAR into the `data/` directory:
- `data/StereoQueerEval_EN_training.tsv`
- `data/StereoQueerEval_IT_training.tsv`
- `data/StereoQueerEval_NL_training.tsv`

### 3. Train Models

#### A. Train a Specific Single Task First (Recommended for Step-by-Step Tuning)
You can isolate any of the subtasks by passing `--target_task`:
```bash
# 1. Train Stereotype detection only (BCE with Logits, binary classification)
python train.py --target_task st --embed_source mmbert --model mmbert_transformer

# 2. Train Hate Speech classification only (Cross-Entropy, 3 classes: no / implicit / explicit)
python train.py --target_task hs --embed_source mmbert --model mmbert_transformer

# 3. Train Target Identity / Scope only (Multi-label BCE, 10-bit bitmask)
python train.py --target_task tg --embed_source mmbert --model mmbert_transformer
```

#### B. Custom Task B: Class-Aware Multi-Head Cross-Attention (MHCA) with Role Injection
Your custom architecture with explicit `<T>`, `<D>`, `<C>` role embeddings, class-conditioned queries `[q_NonHate, q_Implicit, q_Explicit]`, Layer 2 Query Interaction (MHSA), shared scoring head $f_\theta$, and Task C representation bridge $h_B$:
```bash
# Full model with Layer 2 Query Interaction
python train.py --target_task hs \
                --embed_source mmbert \
                --model task_b_class_aware \
                --use_query_interaction \
                --freeze_epochs 15 \
                --unfreeze_epochs 15

# Ablation Hypothesis H2: Disable Layer 2 Query Interaction (Self-Attention between class queries)
python train.py --target_task hs \
                --embed_source mmbert \
                --model task_b_class_aware \
                --no_query_interaction
```

#### B. Joint Multi-Task Training (All 3 Tasks Jointly)
```bash
# mmBERT / ModernBERT (2-Phase Fine-Tuning)
python train.py --target_task all \
                --embed_source mmbert \
                --model mmbert_transformer \
                --batch_size 32 \
                --freeze_epochs 15 \
                --unfreeze_epochs 15 \
                --unfreeze_layers 2 \
                --output_dir checkpoints
```

#### Transformer from Scratch
```bash
python train.py --embed_source scratch \
                --model transformer \
                --epochs 30 \
                --lr 0.0001 \
                --output_dir checkpoints
```

#### Bi-LSTM Baseline
```bash
python train.py --embed_source scratch \
                --model bilstm \
                --epochs 25 \
                --lr 0.001 \
                --output_dir checkpoints
```

### 4. Evaluate Checkpoints
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt \
                   --config checkpoints/pipeline_config.json
```
Output includes:
- Stereotype Accuracy & Macro-F1
- Hate Speech Accuracy & Macro-F1 (3 classes)
- Target Exact Match String Accuracy
- Per-Language breakdown for English, Italian, and Dutch

### 5. Interactive Inference
```bash
python predict.py --checkpoint checkpoints/best_model.pt \
                  --comment "They are trying to push their agenda everywhere." \
                  --title "Debate on LGBTQ+ Curriculum" \
                  --desc "Public discussion regarding education policies."
```

---

## 🛡️ Anti-Leakage Video Splitting
Because YouTube titles and descriptions serve as textual context for comments, comments from the same video **must never** appear in both train and validation splits.
The `DataPipeline` employs `GroupShuffleSplit(groups=df['yt_title'])`, guaranteeing 0% title and description leakage between splits.

---

## 🎯 Target Representation
Targets are encoded into a 10-dimensional binary representation:
- Dimensions `0..8`: Identity tags (`l`, `g`, `b`, `t`, `q`, `i`, `a`, `nb`, `lgbtqia+`)
- Dimension `9`: Scope flag (`1.0` = `group`, `0.0` = `individual`)
Decoded output matches the canonical SemEval target specification (e.g. `group_lgbtqia+`, `individual_t,nb`, or `none`).
