"""
=============================================================================
Kaggle Notebook Script: Domain-Adaptive Warmup Pretraining (TAPT / MLM)
Backbone: jhu-clsp/mmbert-base
Datasets: ucberkeley-dlab/measuring-hate-speech, toxigen-data, falseiftrue/tides + In-domain
Output: Hugging Face Model Hub
=============================================================================
"""

import os
# Force single GPU to prevent ModernBERT DataParallel StopIteration bug
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# ---------------------------------------------------------------------------
# 0. CONFIGURATION & HUGGINGFACE AUTH
# ---------------------------------------------------------------------------
MODEL_NAME = "jhu-clsp/mmbert-base"
OUTPUT_REPO_NAME = "your-username/mmbert-queer-hate-adapted"  # Change to your HF repo
HF_WRITE_TOKEN = "hf_..."  # Set your Hugging Face write token

EPOCHS = 4
BATCH_SIZE = 32
MAX_LENGTH = 256
LEARNING_RATE = 4e-5
MLM_PROBABILITY = 0.15

# Authenticate if token provided
if HF_WRITE_TOKEN.startswith("hf_"):
    from huggingface_hub import login
    login(token=HF_WRITE_TOKEN)
    print("✅ Successfully authenticated with Hugging Face Hub.")

# ---------------------------------------------------------------------------
# 1. LOAD AND FILTER DOMAIN DATASETS
# ---------------------------------------------------------------------------
print("\n>>> [1/4] Loading & Filtering Datasets from Hugging Face...")
collected_texts = []

# --- A. Measuring Hate Speech (UC Berkeley D-Lab) ---
try:
    print("  Loading ucberkeley-dlab/measuring-hate-speech...")
    mhs_ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    mhs_df = mhs_ds.to_pandas()
    # Filter for LGBTQ+ related or hateful comments
    cond = (
        (mhs_df.get('target_sexuality', False) == True) |
        (mhs_df.get('target_gender', False) == True) |
        (mhs_df.get('hate_speech_score', 0) > 0.0)
    )
    mhs_filtered = mhs_df[cond]['text'].dropna().unique().tolist()
    collected_texts.extend(mhs_filtered)
    print(f"    --> Retained {len(mhs_filtered)} LGBTQ+ & hate speech examples from MHS.")
except Exception as e:
    print(f"    ⚠️ Warning loading MHS: {e}")

# --- B. ToxiGen (Implicit Toxicity Dataset) ---
try:
    print("  Loading toxigen/toxigen-data (annotated)...")
    tox_ds = load_dataset("toxigen/toxigen-data", name="annotated", split="train")
    tox_df = tox_ds.to_pandas()
    # Filter for LGBTQ+ target groups
    if 'target_group' in tox_df.columns:
        tox_filtered = tox_df[tox_df['target_group'].astype(str).str.contains('lgbtq|gay|trans|lesbian', case=False, na=False)]['text'].dropna().unique().tolist()
    else:
        tox_filtered = tox_df['text'].dropna().unique().tolist()[:15000]
    collected_texts.extend(tox_filtered)
    print(f"    --> Retained {len(tox_filtered)} examples from ToxiGen.")
except Exception as e:
    print(f"    ⚠️ Warning loading ToxiGen: {e}")

# --- C. TIDES Dataset ---
try:
    print("  Loading falseiftrue/tides...")
    tides_ds = load_dataset("falseiftrue/tides", split="train")
    tides_df = tides_ds.to_pandas()
    text_col = 'text' if 'text' in tides_df.columns else tides_df.columns[0]
    tides_texts = tides_df[text_col].dropna().unique().tolist()
    collected_texts.extend(tides_texts)
    print(f"    --> Retained {len(tides_texts)} examples from TIDES.")
except Exception as e:
    print(f"    ⚠️ Warning loading TIDES: {e}")

# --- D. In-Domain YouTube Dataset (StereoQueer) ---
in_domain_path = "data/train.csv" # or upload to Kaggle input
if os.path.exists(in_domain_path):
    print(f"  Loading in-domain StereoQueer data from {in_domain_path}...")
    df_in = pd.read_csv(in_domain_path)
    for col in ['yt_comment', 'yt_title', 'yt_description']:
        if col in df_in.columns:
            collected_texts.extend(df_in[col].dropna().unique().tolist())
    print(f"    --> Added in-domain YouTube context texts.")

# Clean & Deduplicate
collected_texts = [str(t).strip() for t in collected_texts if len(str(t).strip()) > 10]
collected_texts = list(set(collected_texts))
print(f"\n✨ Total Unique Cleaned Domain Sentences: {len(collected_texts)}")

# ---------------------------------------------------------------------------
# 2. TOKENIZATION & DATA COLLATOR
# ---------------------------------------------------------------------------
print(f"\n>>> [2/4] Tokenizing with {MODEL_NAME} tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

raw_ds = Dataset.from_dict({"text": collected_texts})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True
    )

tokenized_ds = raw_ds.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"]
)

# 15% Dynamic MLM Masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROBABILITY
)

# ---------------------------------------------------------------------------
# 3. TRAINING ARGUMENTS & MLM LOOP
# ---------------------------------------------------------------------------
print("\n>>> [3/4] Configuring Trainer & Starting MLM Pretraining...")
output_dir = "./mmbert-queer-hate-adapted"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    dataloader_num_workers=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_ds,
)

trainer.train()
print("\n✅ Domain-Adaptive MLM Pretraining Complete!")

# ---------------------------------------------------------------------------
# 4. SAVE & PUSH TO HUGGING FACE
# ---------------------------------------------------------------------------
print(f"\n>>> [4/4] Saving and Exporting Checkpoint...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Saved local checkpoint to: {output_dir}")

if OUTPUT_REPO_NAME != "your-username/mmbert-queer-hate-adapted":
    print(f"Pushing model and tokenizer to Hugging Face: {OUTPUT_REPO_NAME}...")
    model.push_to_hub(OUTPUT_REPO_NAME)
    tokenizer.push_to_hub(OUTPUT_REPO_NAME)
    print(f"\n🎉 Successfully published to https://huggingface.co/{OUTPUT_REPO_NAME}!")
    print(f"You can now load it in your main pipeline with:")
    print(f"  config.mmbert_model_name = '{OUTPUT_REPO_NAME}'")
