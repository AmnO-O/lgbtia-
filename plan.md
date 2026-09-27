# Domain-Adaptive Warmup Plan: mmBERT for LGBTQIA+ & Hate Speech Detection

## Executive Summary
This document outlines the end-to-end strategy for Domain-Adaptive Pretraining (DAPT/TAPT) and Intermediate Supervised Warmup for `jhu-clsp/mmbert-base`. 
By adapting mmBERT to social media hate speech, dogwhistles, LGBTQIA+ terminology, and implicit toxicity before downstream fine-tuning, the model builds strong contextual representations that prevent the "Implicit Hate squeeze" observed in Task B.

---

## 1. Target Datasets & Filtering Strategy

| Dataset | HF Hub ID | Size | Target Subset to Filter |
|---|---|---|---|
| **Measuring Hate Speech** | `ucberkeley-dlab/measuring-hate-speech` | ~39.5k rows | Filter rows where `target_sexuality == True` or `target_gender == True` or `hate_speech_score > 0.0` (~15k high-signal rows). |
| **ToxiGen** | `toxigen/toxigen-data` | ~274k rows | Filter `target_group in ['lgbtq', 'gay', 'trans']` and take both implicit toxic and neutral/affirmative pairs (~20k rows). |
| **TIDES** | `falseiftrue/tides` | ~10k rows | Extract LGBTQ+ disparagement, exclusion, and stereotyping text. |
| **In-Domain YouTube Corpus** | `StereoQueer (Train + Val + Unlabeled)` | ~7k rows | Raw comments, video titles, and descriptions. |

**Total Curated Corpus**: **~40,000–50,000 highly focused, domain-aligned sentences**.

---

## 2. Two-Stage Warmup Architecture

```
                                  [jhu-clsp/mmbert-base]
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ STAGE 1: Masked Language Modeling (MLM / DAPT)   │
                  │ Corpus: In-domain YouTube + Filtered LGBTQ data   │
                  │ Epochs: 3-4 | LR: 3e-5 | Batch: 32               │
                  │ Goal: Absorb social slang, sarcasm, dogwhistles  │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ STAGE 2: Intermediate Implicit/Hate Contrastive  │
                  │ Multi-task Binary Warmup (Hate vs Not-Hate)      │
                  │ Epochs: 2 | LR: 2e-5                             │
                  │ Goal: Align latent space for subtle bigotry      │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                            [Push to HuggingFace Hub / Kaggle]
                               e.g. `your-username/mmbert-queer-hate`
                                            │
                                            ▼
                           [Main StereoQueer Task B Training]
```

---

## 3. Step-by-Step Execution Plan

### Step 1: Data Preparation in Kaggle Notebook
1. Load datasets directly from HuggingFace via `datasets` library.
2. Filter for LGBTQIA+ target groups and subtle/implicit hate.
3. Clean and deduplicate text (remove URLs, strip extra spaces, keep emoji & casing).
4. Combine with the in-domain YouTube comments, titles, and descriptions.

### Step 2: Run Masked Language Modeling (DAPT)
1. Tokenize with `AutoTokenizer.from_pretrained("jhu-clsp/mmbert-base")`.
2. Use `DataCollatorForLanguageModeling(mlm_probability=0.15)`.
3. Train `AutoModelForMaskedLM` with AdamW, Cosine schedule, warmup ratio 0.1.
4. **Key Safeguard**: Use low LR (`3e-5` to `5e-5`) so mmBERT does NOT forget Italian & Dutch multilingual grammar.

### Step 3: Push Checkpoint to Hugging Face Model Hub
```python
from huggingface_hub import login
login(token="YOUR_HF_WRITE_TOKEN")

model.save_pretrained("./mmbert-queer-hate-adapted")
tokenizer.save_pretrained("./mmbert-queer-hate-adapted")

model.push_to_hub("your-username/mmbert-queer-hate-adapted")
tokenizer.push_to_hub("your-username/mmbert-queer-hate-adapted")
```

### Step 4: Plug into Main StereoQueer Pipeline
In your Task B training notebook / `train.py`:
```python
# Simply swap the backbone source:
config.mmbert_model_name = "your-username/mmbert-queer-hate-adapted"
```

---

## 4. Expected Impact on Task B Metrics

| Metric | Base mmBERT (Current) | Domain-Adapted mmBERT (Expected) | Why It Improves |
|---|---|---|---|
| **No-Hate F1** | ~0.66 | **~0.70–0.72** | Better comprehension of neutral YouTube discussions |
| **Implicit Hate F1** | **~0.35** ⚠️ | **~0.48–0.55+** 🚀 | Understands dogwhistles & sarcasm without losing to Argmax |
| **Explicit Hate F1** | ~0.62 | **~0.65–0.68** | Sharp boundary on slurs |
| **Macro-F1** | **0.5445** | **0.6100–0.6500+** | Closes the single bottleneck holding the model back |
