# Comprehensive Codebase Bug Audit Report (bugs.md)

This document declares, classifies, and details all bugs, logic flaws, shape mismatches, runtime exceptions, and subtle defects discovered throughout the codebase across the Python pipeline modules, scripts, and Jupyter notebooks.

---

## Summary Matrix

| ID | Severity | File | Affected Function / Component | Bug Summary |
|---|---|---|---|---|
| **BUG-01** | **CRITICAL** | `pipeline/trainer.py` & `pipeline/losses.py` | `StereoQueerTrainer.train_epoch` & `eval_epoch` | `MultiTaskLoss` argument signature and return tuple/dict unpacking mismatch |
| **BUG-02** | **CRITICAL** | `pipeline/losses.py` | `MultiTaskLoss.forward` | Dimension mismatch between `st` logits `[B, 1]` and binary target `[B]` |
| **BUG-03** | **CRITICAL** | `train.py` & `pipeline/trainer.py` | `FeatureClassifier` (`--model feature_mlp`) | Batch tuple length mismatch & Linear layer receiving raw token IDs |
| **BUG-04** | **HIGH** | `pipeline/trainer.py` & `pipeline/losses.py` | `FocalLoss` / `StereoQueerTrainer` | Device mismatch on loss buffer `self.alpha` when training on CUDA GPU |
| **BUG-05** | **HIGH** | `pipeline/models/lstm.py` | `PytorchRNNLSTM.forward` | Bidirectional LSTM pooling on padded sequences causes catastrophic context loss |
| **BUG-06** | **HIGH** | `smoke_test.py` | `TestPipelineSmoke.test_04_model_forward_backward` | Attention map shape assertion failure `(B, 3, S)` vs actual `(B, 8, S)` |
| **BUG-07** | **HIGH** | `evaluate.py` | `main()` | Model architecture hardcoded to `MMBertTransformerModel` / `PytorchTransformerModel` |
| **BUG-08** | **HIGH** | `pipeline/inference.py` | `StereoQueerPredictor._load_model` & `predict_one` | Missing architecture support for `task_b_class_aware` and signature mismatch |
| **BUG-09** | **MEDIUM** | `pipeline/config.py` & `pipeline/task_b_trainer.py` | `FGM` Adversarial Regularization | `fgm_emb_name = "word_embeddings"` does not match ModernBERT's `tok_embeddings` |
| **BUG-10** | **HIGH** | `notebook/task_b_class_aware_train.ipynb` | Cell 19 (`cell_19_cm_code`) | `KeyError: 'hate_speech'` & `KeyError: 'pred_hate_speech'` on prediction CSV |
| **BUG-11** | **MEDIUM** | `notebook/task_b_class_aware_train.ipynb` | Cell 23 (`cell_23_attn_vis`) | Query attention heatmap labels count mismatch (3 labels for 8 aspect queries) |
| **BUG-12** | **MEDIUM** | `pipeline/augmentation.py` | `MULTILINGUAL_SLANG_MAP` | Persian slang typo `"نمی‌دانm"` with Latin character `'m'` |
| **BUG-13** | **MEDIUM** | `pipeline/augmentation.py` | `_augment_single_lang_df` | Compound `'text'` column not updated during augmentation |
| **BUG-14** | **LOW** | `train.py` | `parse_args()` | CLI argument `--two_phase` cannot be disabled via flag |
| **BUG-15** | **MEDIUM** | `pipeline/losses.py` | `FocalLoss.forward` | Numerical instability (`0.0 * inf = NaN`) under AMP / float16 |
| **BUG-16** | **MEDIUM** | `pipeline/models/task_b_class_aware.py` | `init_queries_from_text` | `token_type_ids` kwargs rejected by ModernBERT |
| **BUG-17** | **LOW** | `pipeline/data.py` | `StereoQueerDataset.__init__` | Object array casting on target Series in NumPy >= 1.20 |

---

## Detailed Bug Reports

### 1. BUG-01: MultiTaskLoss Invocation & Return Unpacking Signature Mismatch
- **Severity**: **CRITICAL** (Blocks training and validation)
- **Location**:
  - `pipeline/trainer.py`: Line 88, Line 121
  - `pipeline/losses.py`: Lines 151–166
- **Description**:
  In `pipeline/trainer.py`, `StereoQueerTrainer.train_epoch` and `eval_epoch` call the loss function as:
  ```python
  # trainer.py line 88 & 121
  loss, _ = self.loss_fn(st_logits, hs_logits, tg_logits, st, hs, tg)
  ```
  However, `MultiTaskLoss.forward` in `pipeline/losses.py` is defined as:
  ```python
  # losses.py line 151
  def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      l_st = self.loss_st(preds['st'], targets['st'])
      l_hs = self.loss_hs(preds['hs'], targets['hs'])
      l_tg = self.loss_tg(preds['tg'], targets['tg'])
      total = self.w_st * l_st + self.w_hs * l_hs + self.w_tg * l_tg
      return {'total': total, 'st': l_st, 'hs': l_hs, 'tg': l_tg}
  ```
- **Consequence**:
  1. Passing 6 positional tensors raises `TypeError: MultiTaskLoss.forward() takes 3 positional arguments but 7 were given`.
  2. Unpacking a 4-key dict into `loss, _` raises `ValueError: too many values to unpack (expected 2, got 4)`.
- **Fix**:
  Support either dictionary inputs or 6 positional arguments in `MultiTaskLoss.forward`, and return a tuple `(total_loss, loss_dict)` or access `loss['total']`.

---

### 2. BUG-02: Shape / Broadcast Mismatch in `MultiTaskLoss.loss_st`
- **Severity**: **CRITICAL** (Loss dimension corruption or PyTorch broadcasting warning)
- **Location**:
  - `pipeline/losses.py`: Line 156
- **Description**:
  `st_head` in all models (`MMBertTransformerModel`, `FeatureClassifier`, `PytorchTransformerModel`, `PytorchRNNLSTM`, `VanillaRNNModel`) produces logits of shape `[BatchSize, 1]`.
  Meanwhile, `st` labels from `StereoQueerDataset` or `MMBertSeqDataset` have shape `[BatchSize]`.
  `nn.BCEWithLogitsLoss()` requires matching shapes; when given `[B, 1]` and `[B]`, it broadcasts to `[B, B]` loss matrix before reduction.
- **Fix**:
  Ensure `preds['st'].squeeze(-1)` or `targets['st'].view_as(preds['st'])` before passing into `self.loss_st`.

---

### 3. BUG-03: `FeatureClassifier` (`--model feature_mlp`) Crash & Dataset Unpack Failure
- **Severity**: **CRITICAL** (Blocks feature MLP model execution)
- **Location**:
  - `train.py`: Lines 160–163
  - `pipeline/trainer.py`: Lines 78–86, Lines 111–119
- **Description**:
  When `embed_source="mmbert"` and `model_type="feature_mlp"`, `train.py` sets `is_mmbert_tf = False`.
  `data_pipeline.create_dataloaders(tokenizer)` returns `MMBertSeqDataset` which outputs 5 tensors per batch: `(ids, mask, st, hs, tg)`.
  In `pipeline/trainer.py`:
  ```python
  if self.is_mmbert_tf:
      ids, mask, st, hs, tg = batch
  else:
      inputs, st, hs, tg = batch # Fails with ValueError: too many values to unpack (expected 4, got 5)
  ```
  Additionally, `FeatureClassifier` contains `Linear(768, 384)` with no embedding lookup layer, yet `inputs` contains integer token IDs `[B, 256]` rather than 768-d pooled vectors.
- **Fix**:
  If using `feature_mlp`, either pre-compute pooled embeddings with mmBERT or embed token IDs before the linear classifier.

---

### 4. BUG-04: Device Mismatch on Loss Function Buffer `self.alpha` in `StereoQueerTrainer`
- **Severity**: **HIGH** (Crashes GPU training when `class_weights` is set)
- **Location**:
  - `pipeline/trainer.py`: Line 44–46
  - `pipeline/losses.py`: Lines 36–40, 74–76
- **Description**:
  In `StereoQueerTrainer.__init__`, `self.model.to(self.device)` is executed, but `self.loss_fn.to(self.device)` is never called.
  When `class_weights` is provided, `FocalLoss` registers `alpha` via `self.register_buffer("alpha", alpha)`. Because `self.loss_fn` remains on CPU, `alpha_t = self.alpha[targets]` in `FocalLoss.forward` attempts to index a CPU tensor with CUDA `targets`, triggering:
  `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor`.
- **Fix**:
  Call `self.loss_fn.to(self.device)` in `StereoQueerTrainer.__init__`.

---

### 5. BUG-05: Bidirectional LSTM Context Truncation on Padded Sequences
- **Severity**: **HIGH** (Model degradation and catastrophic forgetting)
- **Location**:
  - `pipeline/models/lstm.py`: Line 37
- **Description**:
  In `PytorchRNNLSTM.forward`:
  ```python
  lstm_out, _ = self.lstm(embedded)
  final_hidden_state = lstm_out[:, -1, :]
  ```
  Since inputs are padded with `<PAD>=0` up to `max_len=256`:
  1. The forward LSTM at timestep `-1` has processed dozens/hundreds of trailing `<PAD>` tokens, forgetting real sequence content.
  2. The backward LSTM at timestep `-1` is at its very first processing step.
- **Fix**:
  Apply masked mean pooling, max pooling across valid sequence tokens, or use `torch.nn.utils.rnn.pack_padded_sequence`.

---

### 6. BUG-06: Smoke Test Assertion Failure on Multi-Query Attention Map Shape
- **Severity**: **HIGH** (Automated CI/CD test failure)
- **Location**:
  - `smoke_test.py`: Line 142
  - `pipeline/models/task_b_questions.py`: Lines 48–140
- **Description**:
  `smoke_test.py` line 142 asserts:
  ```python
  self.assertEqual(attn.shape, (B, 3, S))
  ```
  However, `TaskBClassAwareAttentionModel` initializes with `DEFAULT_TASK_B_PROBES`, which contains 8 fine-grained aspect probes (3 for Non-Hate, 3 for Implicit, 2 for Explicit). The returned cross-attention map `attn_weights` has shape `(B, 8, S)`, causing the assertion to fail (`torch.Size([2, 8, 16]) != (2, 3, 16)`).
- **Fix**:
  Update `smoke_test.py` to assert `(B, model.num_queries, S)` or pass 3 probes in test initialization.

---

### 7. BUG-07: Hardcoded Model Classes in `evaluate.py`
- **Severity**: **HIGH** (Evaluation script fails on non-default architectures)
- **Location**:
  - `evaluate.py`: Lines 46–59
- **Description**:
  `evaluate.py` does not inspect `config.model_type`. If `embed_source == "mmbert"`, it strictly instantiates `MMBertTransformerModel`. If `embed_source == "scratch"`, it strictly instantiates `PytorchTransformerModel`.
  If a user trains `--model bilstm`, `--model rnn`, or `--model task_b_class_aware`, running `evaluate.py` crashes with `RuntimeError: Error(s) in loading state_dict for PytorchTransformerModel: Missing key(s)... Unexpected key(s)...`.
- **Fix**:
  Dispatch model instantiation based on `config.model_type` (`bilstm`, `rnn`, `transformer`, `task_b_class_aware`, `mmbert_transformer`).

---

### 8. BUG-08: Predictor Model Instantiation and Output Mismatch
- **Severity**: **HIGH** (Inference CLI fails for Task B Class-Aware model)
- **Location**:
  - `pipeline/inference.py`: Lines 41–80, 108–117
- **Description**:
  1. `StereoQueerPredictor._load_model` does not instantiate `TaskBClassAwareAttentionModel` when `config.model_type == 'task_b_class_aware'`.
  2. `predict_one` assumes `model(ids, mask)` returns a 3-tuple `(st_logits, hs_logits, tg_logits)`. For `TaskBClassAwareAttentionModel`, forward requires `role_ids` and returns `(logits, h_B, attn)`.
- **Fix**:
  Add `task_b_class_aware` branch in `_load_model` and handle role encoding in `predict_one`.

---

### 9. BUG-09: Fast Gradient Method (FGM) Parameter Name Mismatch for ModernBERT
- **Severity**: **MEDIUM** (Adversarial regularization is silently skipped)
- **Location**:
  - `pipeline/config.py`: Line 43 (`fgm_emb_name = "word_embeddings"`)
  - `pipeline/task_b_trainer.py`: Lines 31, 38–40
- **Description**:
  In `FGM.attack()`:
  ```python
  if param.requires_grad and self.emb_name in name and param.grad is not None:
  ```
  In ModernBERT / mmBERT (`jhu-clsp/mmbert-base`), the embedding parameter is named `embeddings.tok_embeddings.weight`. Because `"word_embeddings"` is not in `name`, `FGM` never finds any embedding weights, silently skipping adversarial perturbation during training.
- **Fix**:
  Check for `"tok_embeddings"` or `"word_embeddings"` in parameter name matching.

---

### 10. BUG-10: Task B Jupyter Notebook Column Name `KeyError`
- **Severity**: **HIGH** (Notebook cell crash during final evaluation)
- **Location**:
  - `notebook/task_b_class_aware_train.ipynb`: Cell 19 (lines 491–494)
  - `pipeline/task_b_trainer.py`: Lines 420–435
- **Description**:
  Cell 19 in `task_b_class_aware_train.ipynb` reads `task_b_val_predictions.csv` with:
  ```python
  y_true = df_preds['hate_speech'].tolist()
  y_pred = df_preds['pred_hate_speech'].tolist()
  ```
  However, `TaskBTrainer` saves predictions with column names `pred_class`, `prob_no`, `prob_implicit`, `prob_explicit`. The columns `'hate_speech'` and `'pred_hate_speech'` are absent from the CSV, resulting in an unhandled `KeyError`.
- **Fix**:
  Map `pred_class` to string names (`IDX2HATE`) and include `df_val['hate_speech']` in the exported DataFrame.

---

### 11. BUG-11: Attention Heatmap Y-Tick Labels Count Mismatch
- **Severity**: **MEDIUM** (Visual distortion / warning in interpretability heatmap)
- **Location**:
  - `notebook/task_b_class_aware_train.ipynb`: Cell 23 (lines 574–582)
- **Description**:
  Cell 23 plots `attn_map_np` of shape `[8, valid_len]` (8 aspect probes), but provides only 3 labels in `yticklabels=['NonHate Query', 'Implicit Query', 'Explicit Query']`.
- **Fix**:
  Extract query aspect descriptions dynamically from `model.probes` (e.g. `[p.aspect for p in model.probes]`).

---

### 12. BUG-12: Multilingual Slang Typo in Persian Dictionary
- **Severity**: **MEDIUM** (Data augmentation defect)
- **Location**:
  - `pipeline/augmentation.py`: Line 68
- **Description**:
  In `MULTILINGUAL_SLANG_MAP["fa"]`:
  ```python
  "نمی‌دانm": ["نمیدونم", "نمیدanم"],
  ```
  The Persian verb "نمی‌دانم" contains an accidental Latin character `'m'` at the end instead of Persian `'م'`. Exact string lookups in `augment_slang_noise` will never match the correct Persian word.
- **Fix**:
  Replace `"نمی‌دانm"` with `"نمی‌دانم"`.

---

### 13. BUG-13: Incomplete Text Field Augmentation in `_augment_single_lang_df`
- **Severity**: **MEDIUM** (Augmented comments not reflected in pre-constructed compound text)
- **Location**:
  - `pipeline/augmentation.py`: Lines 450–461
- **Description**:
  `_augment_single_lang_df` updates `new_row['yt_comment']`, `new_row['yt_title']`, and `new_row['yt_description']`. However, if the DataFrame already has a `'text'` column (built earlier via `DataPipeline.load_data`), `new_row['text']` is not reconstructed. Downstream models consuming `row['text']` will process the unaugmented comment.
- **Fix**:
  Re-synthesize `new_row['text'] = safe_clean(f"{new_row['yt_comment']} [SEP] {new_row['yt_title']} [SEP] {new_row['yt_description']}")` when `'text'` is in `new_row`.

---

### 14. BUG-14: Inability to Disable `--two_phase` Training via CLI
- **Severity**: **LOW** (CLI usability defect)
- **Location**:
  - `train.py`: Lines 62–63
- **Description**:
  `parser.add_argument("--two_phase", action="store_true", default=True)` is configured with `default=True` and `action="store_true"`. There is no `--no_two_phase` flag, meaning CLI users cannot disable two-phase training without writing a JSON configuration file.
- **Fix**:
  Add `--no_two_phase` with `dest="two_phase", action="store_false"`.

---

### 15. BUG-15: Potential NaN in `FocalLoss` under Automatic Mixed Precision (AMP)
- **Severity**: **MEDIUM** (Training instability)
- **Location**:
  - `pipeline/losses.py`: Lines 52–63
- **Description**:
  In `FocalLoss.forward`:
  ```python
  log_p = F.log_softmax(inputs, dim=-1)
  p = torch.exp(log_p)
  ```
  Under float16 / AMP, `p` can underflow to `0.0` or reach `1.0`. When `target_p == 0.0`, `target_log_p` becomes `-inf`. When label smoothing is applied, multiplying `(1 - target_p)^gamma` with `smooth_loss` can lead to `0.0 * inf = NaN`.
- **Fix**:
  Clamp `p` and `target_p` with `torch.clamp(..., min=1e-7, max=1.0 - 1e-7)`.

---

### 16. BUG-16: Tokenizer Kwargs Collision in `init_queries_from_text`
- **Severity**: **MEDIUM** (Runtime error when initialized with standard BERT tokenizers)
- **Location**:
  - `pipeline/models/task_b_class_aware.py`: Lines 228–231
- **Description**:
  When `tokens = tokenizer(...)` is passed to ModernBERT via `self.mmbert(**tokens)`, standard BERT tokenizers include `'token_type_ids'`. ModernBERT does not accept `token_type_ids`, raising `TypeError: ModernBertModel.forward() got an unexpected keyword argument 'token_type_ids'`.
- **Fix**:
  Filter tokens to only `input_ids` and `attention_mask`:
  `tokens = {k: v.to(dev) for k, v in tokens.items() if k in ('input_ids', 'attention_mask')}`.

---

### 17. BUG-17: Target Series Array Object Casting in `StereoQueerDataset`
- **Severity**: **LOW** (NumPy deprecation / ValueError on newer NumPy versions)
- **Location**:
  - `pipeline/data.py`: Line 83
- **Description**:
  `self.tg_labels = np.asarray(tg_labels, dtype=np.float32)` on a Pandas Series containing 10-dimensional NumPy vectors creates an object array that can fail casting with `ValueError: setting an array element with a sequence`.
- **Fix**:
  Use `np.stack(list(tg_labels)).astype(np.float32)`.

---

## Recommended Action Plan
1. **Immediate P0**: Patch `pipeline/losses.py` and `pipeline/trainer.py` to fix `MultiTaskLoss` calling signatures and dimensions (`BUG-01`, `BUG-02`, `BUG-04`).
2. **Immediate P0**: Patch `smoke_test.py` attention map dimension test (`BUG-06`).
3. **P1**: Update `evaluate.py` and `pipeline/inference.py` to support all configured model architectures (`BUG-07`, `BUG-08`).
4. **P1**: Fix `FGM` embedding layer detection for ModernBERT (`BUG-09`) and patch `task_b_class_aware_train.ipynb` predictions dataframe keys (`BUG-10`, `BUG-11`).
5. **P2**: Correct multilingual dictionary typo in `pipeline/augmentation.py` (`BUG-12`) and dataset text synchronization (`BUG-13`).
