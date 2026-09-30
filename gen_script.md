# LLM Rationale (ALPI Hint) Generation — Reproduction Log

Label-free **5-axis diagnostic rationale extraction** for the ALPI (Additive Latent Privileged
Information) teacher in `lgbtia-`. For each StereoQueerEval training row, an LLM decomposes the
comment (direct_hostility / indirect_subtext / context_dependence / counter_speech /
target_reference), an independent judge predicts {stereotype, hate_speech, target_identities,
target_scope, confidence}, and a QC gate compares the judge against the official gold labels.
High-quality rows get a **pre-compiled ≤64-token sanitized hint**; low-quality rows fall back to
``hint=""`` so the row trains unconditionally (never dropped).

Companion log for the synth corpus: [`../gen_script.md`](../gen_script.md) (repo root).

## Scripts

| File | Purpose |
|---|---|
| `gen/llm_rationalize.py` | Production generator; Gemini + optional Groq, rate limiting, atomic checkpoint/resume, QC gate, hint compilation |
| `pipeline/task_b_data.py` | Consumes `hint` column (ignored when empty) |

## Environment (Windows, PowerShell 5.1)

- `python -m pip install google-genai openai`  (same as root corpus log)
- **API key**: `.env` at repo root (`C:\CODE_SOMETHING\LGBT\.env`). The loader searches cwd → script
  dir → up to 3 levels up, so it finds the root `.env` from `lgbtia-` too.
- **Model**: `gemini-3.5-flash-lite` is the verified-working default (repo meta: `gemini-2.5-flash` = 404,
  `gemini-3.6-flash` also OK; Groq `gpt-oss-120b` OK for non-batch only).
- **Run from `lgbtia-`** so relative paths resolve: `--out LGBT/rationales.json` → `lgbtia-\LGBT\`.

## Output schema (`LGBT/rationales.json`, single file for all languages)

IDs are globally unique across languages (`training_EN_0001`, `training_IT_*`, `training_NL_*`,
`synth_*` — verified 0 collisions), so one flat file keyed by `StereoQueerEval_id` is safe.

```json
{
  "training_EN_0001": {
    "lang": "EN",
    "axes": {"direct_hostility": "weak", "indirect_subtext": "strong",
              "context_dependence": "strong", "counter_speech": "weak",
              "target_reference": "present"},
    "why": "...",
    "boundary": "...",
    "hint": "axes: ... | why: ... | flip: ...",          // "" if QC-failed
    "judge": {"stereotype": 0, "hate_speech": "yes_implicit",
              "target_identities": ["lgbtqia+"], "target_scope": "group",
              "target_string": "group_lgbtqia+", "confidence": 0.91},
    "quality_control": {"matches_gold_hs": true, "matches_gold_st": true,
                        "matches_gold_tg": false, "is_high_quality": true}
  }
}
```

- `hint` = sanitized (masked) + truncated to `--hint-max-tokens` (default 64). Audience: frozen mmBERT
  `[CLS]` → privileged vector; **never contains the target identity** (leakage guard).
- QC gate: `is_high_quality = matches_gold_hs OR (gold_hs == 'yes_implicit' AND confidence >= 0.60)`.
  st/tg matches are recorded for analytics only, never gate.
- Compare: judge hate_speech vs gold — `yes_implicit` vs the subtle `implicit`/`yes_implicit`
  spelling is handled by exact 3-class compare.

## Checkpointing (do NOT lose runs)

- `--resume` (default ON) loads the existing `--out` and skips already-cached ids.
- Save is **atomic**: writes `*.json.tmp` then `os.replace()` per batch → interruption can lose at
  most the in-flight batch, never the file.
- **Corrupt cache kills the run loudly** (SystemExit) instead of silently resetting to `{}` —
  that accidental wipe once destroyed a 113-item registry on the root corpus.
- `--force-restart` only for a deliberate full reset.
- Resume stats report `Cached/Pending`; QC acceptance printed per (gold_hs, lang) at the end.

## Data cleaning (2026-09-30, applied IN-PLACE to official TSVs)

The official `StereoQueerEval_{EN,IT,NL}_training.tsv` files embedded literal newlines/tabs inside
`yt_description` (and some comments) — 1,786–2,280 rows per lang. Cleaned in place in **two passes**
(original originals preserved in `LGBT\_backup_orig\`):

**Pass 1 — whitespace normalization:** collapse all whitespace runs → single space; description
truncated to 800 chars (labels untouched).

**Pass 2 — noise-line stripping:** multi-line descriptions were dominated by footer boilerplate
(social links, subscribe CTAs, channel promo, newsletter/contact/email, trailing hashtag runs).
Per-line rules strip: `[URL]`/`[EMAIL]` social & link blocks, subscribe/follow CTA lines, NL CTAs
(`abonneer`, `word lid`, `steun`, `volg ons`, `...vinden op`), IT CTAs (`abbonati`, `rimani
connesso`, `notifiche`, `piaciuto`), and trailing hashtag runs. Remaining lines are re-joined
single-space. Result: **38,763 noise lines removed**, 0 residual CTA/social, and **no description
needs truncation anymore** (all fit under the 800 cap).

| Lang | pass-1 changed | lines removed | mean len | p95 | max |
|---|---|---|---|---|---|
| EN | 1,850 | 11,026 | 293 | 663 | 798 |
| IT | 2,316 | 13,006 | 533 | 798 | 799 |
| NL | 2,082 | 14,731 | 289 | 795 | 799 |

Verification: row counts intact (2,989/2,400/2,238), `hate_speech`/`stereotype` distributions
unchanged, 0 newlines remaining, random eyeball of cleaned desc confirms news prose intact.
Generator reads these cleaned files directly; `--desc-max` (default 800) now acts as a safety net
rather than an active cut.

## Command log

### Pilot (2026-09-29) — verify schema + gate end-to-end, EN only
```powershell
# CWD = lgbtia-; wrote lgbtia-\LGBT\rationales_pilot.json
python -u gen/llm_rationalize.py --lang EN --limit 10 --batch-size 5 --out LGBT/rationales_pilot.json --rpm 14 --hint-max-tokens 64
```
Result: 10/10 generated (2 batches; 2nd batch verified `--resume` pick-up: 5 cached / 5 pending).
Pilot QC: 3×`no` kept, 2×`yes_explicit` empty_fallback in the resumed batch; overall 6/10 HQ
(`empty_fallback` = judge vs gold mismatch — EXACTLY the veil-shift case we want to fall back on).
Hints ~26 words, well under the 64-token budget. Judge targets: `none` / `group_lgbtqia+` /
`individual_lgbtqia+`.

### Full run (EN → IT → NL, resume-safe; re-run to finish any interrupted segment)
```powershell
python -u gen/llm_rationalize.py --lang EN --batch-size 5 --out LGBT/rationales.json --rpm 14 --hint-max-tokens 64
python -u gen/llm_rationalize.py --lang IT --batch-size 5 --out LGBT/rationales.json --rpm 14 --hint-max-tokens 64
python -u gen/llm_rationalize.py --lang NL --batch-size 5 --out LGBT/rationales.json --rpm 14 --hint-max-tokens 64
```
All three write to the **same** `LGBT/rationales.json`; later runs resume + append other languages.

### If synth corpus is folded into training (undecided)
Add the synth/corpus globs so synth rows also get hints. Currently `DEFAULT_INPUT_GLOBS` pulls the
official `LGBT/StereoQueerEval_*_training.tsv` only.

## Current state
- Official rows: EN 2,989 / IT 2,400 / NL 2,238 (7,627 total). Aux + synth not yet rationalized.
- `LGBT/rationales_pilot.json`: 10 EN entries (test artifact — delete before the real run or keep as fixture).
- Notebook `notebook/task_b_class_aware_train.ipynb` merges `rationales.json`, falls back to `''`
  when a row has no hint, sets `use_privileged_guidance=True` when hints exist.

## Known pitfalls
- Do NOT set `--model gemini-2.5-flash` (404). Stick to the default or `gemini-3.6-flash`.
- Run from `lgbtia-`, not the repo root, so `--out` and input globs resolve to this project.
- `sent` AFC warning from google-genai is harmless.
- Batch-level progress line requires the returned list to be keyed by `id` (already enforced via `res_map`).