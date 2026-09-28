#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Multi-Variant Data Augmentation Script for StereoQueerEval 2027:
Generates multiple high-quality, nuanced "yes_implicit" hate speech samples per "yes_explicit"
training row in a SINGLE LLM prompt (e.g., 2-3 distinct stylistic variants: sarcasm, rhetorical question, dog-whistle).
Supports Google Gemini Flash (google-genai) and Groq (openai SDK) with full rate-limit handling,
safety filter configuration, contextual title/description preservation, and JSON validation.

Outputs:
  - {out_dir}/outputs.json                       : Full metadata & audit trace (per row)
  - {tsv_dir}/SynthImplicit_{LANG}_training.tsv  : Exact 7-column SemEval-compatible dataset:
    (StereoQueerEval_id | yt_title | yt_description | yt_comment | stereotype | hate_speech | target)
    Where: hate_speech = 'yes_implicit', stereotype & target & yt_title & yt_description are strictly inherited.

Usage:
  # Generate 3 diverse implicit variants per explicit comment (e.g. 100 rows -> 300 synthetic samples in 100 API calls)
  python gen/llm_implicit_augment.py --api gemini --lang NL --limit 100 --n-variants 3

  # Using Groq (Llama-3.3-70B):
  python gen/llm_implicit_augment.py --api groq --lang IT --n-variants 2

  # Resume interrupted generation:
  python gen/llm_implicit_augment.py --resume
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any

def auto_load_dotenv():
    """Automatically loads .env from current or parent directories without external dependencies."""
    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ]
    for d in search_dirs:
        env_path = os.path.join(d, '.env')
        if os.path.isfile(env_path):
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        key, val = line.split('=', 1)
                        key = key.strip()
                        val = val.strip().strip("'").strip('"')
                        if key and key not in os.environ:
                            os.environ[key] = val
            except Exception:
                pass
            break

auto_load_dotenv()

# -----------------------------------------------------------------------------
# CONSTANTS & SCHEMAS
# -----------------------------------------------------------------------------

HEADER = [
    'StereoQueerEval_id',
    'yt_title',
    'yt_description',
    'yt_comment',
    'stereotype',
    'hate_speech',
    'target'
]

DEFAULT_INPUT_GLOBS = [
    'LGBT/StereoQueerEval_*_training.tsv',
    '../LGBT/StereoQueerEval_*_training.tsv',
    '../StereoQueerEval_*_training.tsv',
    'data/StereoQueerEval_*_training.tsv',
    'StereoQueerEval_*_training.tsv'
]

# Targeted AI Refusal Patterns (Avoid false-positives on conversational sarcasm like "I'm sorry, but...")
REFUSAL_PATTERNS = [
    r"^i cannot (?:fulfill|generate|create|comply|assist)",
    r"^as an ai\b",
    r"^i am an ai\b",
    r"\bagainst my (?:safety|content|usage) (?:guidelines|policies)\b",
    r"^i am unable to (?:generate|create|comply|assist)",
    r"cannot generate hate speech",
    r"^i apologize, but i cannot"
]

LANGUAGE_INSTRUCTIONS = {
    'EN': "English. Use authentic English YouTube comment phrasing (informal, rhetorical questions, dog-whistles, sarcasm).",
    'IT': "Italian. Use authentic Italian YouTube comment phrasing (e.g., subtle colloquialisms, 'ma che schifo di valori', faux-irony, rhetorical questions). Do NOT translate into English.",
    'NL': "Dutch. Use authentic Dutch YouTube comment phrasing (e.g., 'even normaal doen', subtle sarcasm, moral lecturing, rhetorical questions). Do NOT translate into English."
}

SYSTEM_PROMPT_TEMPLATE = """You are an expert computational linguist assisting in dataset augmentation for LGBTQ+ hate-speech detection research (SemEval StereoQueerEval).
Your task is Counterfactual Tone Perturbation: convert an EXPLICIT hate comment into {n_variants} DISTINCT and DIVERSE IMPLICIT (indirect / veiled) hate comments.

STRICT REQUIREMENTS:
1. TARGET PRESERVATION: Keep the exact same target identity (e.g., gay, trans, lesbian, bisexual, non-binary).
2. INTENT PRESERVATION: The underlying hostility, prejudice, or rejection MUST remain clearly perceptible between the lines in each variant.
3. REMOVE OVERT WORDS: Completely remove all direct slurs, swear words, explicit vulgarities, or overt violent threats.
4. STYLISTIC DIVERSITY ACROSS VARIANTS:
   - Variant 1: Sarcasm / Irony / Mocking tone
   - Variant 2: Rhetorical Question / False concern ("Is this really what we want our kids to see?", "Just asking questions")
   - Variant 3 (if requested): Dog-whistles, euphemisms, coded language ("traditional family values", "agenda", "lifestyle choice", "social contagion") or passive-aggressive moralizing
5. LANGUAGE PRESERVATION: Every variant MUST be in the exact same language ({lang_instruction}).
6. PLATFORM STYLE: Keep the natural, concise YouTube comment style (keep tokens like [url], [channel], [email] if present in source).

OUTPUT FORMAT:
You MUST reply with ONLY a raw JSON object (no markdown, no backticks, no explanations):
{{
  "variants": [
    "Implicit variation 1",
    "Implicit variation 2"
  ]
}}"""


class RetryableAPIError(Exception):
    def __init__(self, wait_hint: Optional[float] = None):
        super().__init__('Retryable API Error')
        self.wait_hint = wait_hint


class FatalAPIError(Exception):
    pass


# -----------------------------------------------------------------------------
# ARGUMENT PARSER
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--api', choices=['gemini', 'groq'], default='gemini',
                        help='LLM API provider: gemini (Google GenAI) or groq (OpenAI-compatible SDK). Default: gemini')
    parser.add_argument('--model', default=None,
                        help='Model name. Default: gemini-2.5-flash (for gemini) or llama-3.3-70b-versatile (for groq)')
    parser.add_argument('--input', action='append', default=None,
                        help='File or glob pattern of original source dataset (e.g. data/StereoQueerEval_NL_training.tsv)')
    parser.add_argument('--lang', default=None,
                        help='Filter only specific language code (e.g. EN, IT, NL)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of explicit source rows to process')
    parser.add_argument('--n-variants', type=int, default=3,
                        help='Number of distinct implicit variants to generate per explicit comment in a single prompt (default: 3)')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Pause/cooldown in seconds between each request (e.g. --delay 4.0 for Gemini Free Tier 15 RPM)')
    parser.add_argument('--rpm', type=int, default=None,
                        help='Target maximum Requests Per Minute rate limit (e.g. --rpm 14 for Free Tier)')
    parser.add_argument('--max-concurrency', type=int, default=None,
                        help='Max parallel worker threads (default: 8 for gemini, 4 for groq; use 1-2 if on Free Tier)')
    parser.add_argument('--max-retries', type=int, default=6,
                        help='Max retries per sample upon rate limit')
    parser.add_argument('--max-chars', type=int, default=1200,
                        help='Truncate over-length input comments')
    parser.add_argument('--temp', type=float, default=0.75,
                        help='Sampling temperature')
    parser.add_argument('--out', default='LGBT/outputs.json',
                        help='Output JSON metadata path')
    parser.add_argument('--tsv-dir', default='LGBT',
                        help='Output directory for SynthImplicit_{LANG}_training.tsv files')
    parser.add_argument('--resume', action='store_true',
                        help='Skip rows already present in output JSON')
    parser.add_argument('--txt-suffix', default='_training.tsv',
                        help='Suffix of output TSV files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only inspect and print explicit statistics without making API calls')
    return parser.parse_args()


ARGS = parse_args()


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_existing(out_path: str) -> Dict[str, Dict[str, Any]]:
    if not ARGS.resume or not os.path.exists(out_path):
        return {}
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data.get('items', [])
        loaded: Dict[str, Dict[str, Any]] = {}
        for it in items:
            key = it.get('source_id') or it.get('id')
            if key and (it.get('variants') or it.get('rewritten')):
                loaded[key] = it
        return loaded
    except Exception as e:
        print(f"  [Warning] Could not load resume cache from {out_path}: {e}")
        return {}


def find_input_paths() -> List[str]:
    patterns = ARGS.input or DEFAULT_INPUT_GLOBS
    found: List[str] = []
    for p in patterns:
        if os.path.isdir(p):
            found += glob.glob(os.path.join(p, 'StereoQueerEval_*_training.tsv'))
        else:
            found += glob.glob(p, recursive=True)
    return sorted(set(found))


def read_source_rows(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        lang_match = re.search(r'_([A-Z]{2})_training\.tsv$', os.path.basename(p), re.IGNORECASE)
        lang = (lang_match.group(1).upper() if lang_match else 'EN')
        if ARGS.lang and lang != ARGS.lang.upper():
            continue

        try:
            with open(p, 'r', encoding='utf-8', errors='replace', newline='') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader, None)
                if not header:
                    continue

                col_idx = {name: header.index(name) for name in header if name in HEADER}
                if 'yt_comment' not in col_idx or 'hate_speech' not in col_idx:
                    continue

                for line in reader:
                    if len(line) <= max(col_idx.values()):
                        continue
                    if line[col_idx['hate_speech']].strip() != 'yes_explicit':
                        continue

                    comment_text = line[col_idx['yt_comment']].strip()
                    if not comment_text:
                        continue

                    row_id = line[col_idx['StereoQueerEval_id']] if 'StereoQueerEval_id' in col_idx else f"{lang}-{len(rows)}"
                    title = line[col_idx['yt_title']].strip() if 'yt_title' in col_idx else ""
                    desc = line[col_idx['yt_description']].strip() if 'yt_description' in col_idx else ""
                    stereotype = line[col_idx['stereotype']].strip() if 'stereotype' in col_idx else "no"
                    target = line[col_idx['target']].strip() if 'target' in col_idx else "none"

                    rows.append({
                        'id': row_id,
                        'lang': lang,
                        'yt_title': title,
                        'yt_description': desc,
                        'stereotype': stereotype,
                        'target': target,
                        'text': comment_text[:ARGS.max_chars],
                    })
        except Exception as e:
            print(f"  [Error reading {p}]: {e}", file=sys.stderr)
    return rows


# -----------------------------------------------------------------------------
# CLIENT & MODEL INITIALIZATION
# -----------------------------------------------------------------------------

def init_client() -> Tuple[str, Any]:
    if ARGS.api == 'gemini':
        try:
            from google import genai
            key = os.environ.get('GEMINI_API_KEY')
            if not key:
                raise SystemExit('Missing GEMINI_API_KEY environment variable. Add it to .env or run: export GEMINI_API_KEY="..."')
            return ('gemini', genai.Client(api_key=key))
        except ImportError:
            raise SystemExit("Please install google-genai: pip install google-genai")

    elif ARGS.api == 'groq':
        try:
            from openai import OpenAI
            key = os.environ.get('GROQ_API_KEY')
            if not key:
                raise SystemExit('Missing GROQ_API_KEY environment variable. Add it to .env or run: export GROQ_API_KEY="..."')
            return ('groq', OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1'))
        except ImportError:
            raise SystemExit("Please install openai: pip install openai")

    raise ValueError(f"Unknown API provider: {ARGS.api}")


_CTX: Dict[str, Any] = {'client': None, 'model': None}


def get_default_model(api_kind: str) -> str:
    if ARGS.model:
        return ARGS.model
    if api_kind == 'gemini':
        return 'gemini-2.5-flash'
    return 'llama-3.3-70b-versatile'


def extract_retry_after(err: Any) -> Optional[float]:
    """Safely extracts server Retry-After header to respect rate limits."""
    headers = getattr(getattr(err, 'response', None), 'headers', None) or \
              getattr(getattr(err, 'http_response', None), 'headers', None)
    if headers:
        ra = headers.get('retry-after') or headers.get('Retry-After')
        if ra:
            try:
                return max(0.5, float(ra))
            except (ValueError, TypeError):
                pass
    return None


# -----------------------------------------------------------------------------
# API CALL & PARSING
# -----------------------------------------------------------------------------

def build_prompt_payload(row: Dict[str, Any]) -> Tuple[str, str]:
    """Constructs prompts safely without brace-formatting crashes on user content."""
    lang = row['lang']
    lang_inst = LANGUAGE_INSTRUCTIONS.get(lang, f"{lang}. Use authentic colloquial YouTube style.")
    system = SYSTEM_PROMPT_TEMPLATE.format(lang_instruction=lang_inst, n_variants=ARGS.n_variants)
    
    # Use direct string concatenation / safe f-string to prevent KeyError on braces {} in text/title/desc
    title_str = row.get('yt_title', '') or '(None)'
    desc_str = row.get('yt_description', '') or '(None)'
    comment_str = row.get('text', '')

    user = (
        f"Video Context:\n"
        f"- Title: {title_str}\n"
        f"- Description: {desc_str}\n\n"
        f"Explicit Comment to Rephrase:\n"
        f"\"{comment_str}\"\n\n"
        f"Generate {ARGS.n_variants} distinct implicit hate variations:"
    )
    return system, user


def call_llm(system_prompt: str, user_prompt: str) -> str:
    kind, cli = _CTX['client']
    model_name = _CTX['model']

    if kind == 'gemini':
        from google.genai import types
        from google.genai import errors as genai_errors

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=ARGS.temp,
                max_output_tokens=1200,
                response_mime_type="application/json",
            )
            response = cli.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config
            )
            return response.text or ""
        except genai_errors.APIError as e:
            code = getattr(e, 'code', None)
            wait = extract_retry_after(e) or 2.0
            if code in (429, 500, 502, 503, 504):
                raise RetryableAPIError(wait_hint=wait) from e
            raise FatalAPIError(f"Gemini API Error (code {code}): {e}") from e
        except Exception as e:
            wait = extract_retry_after(e) or 3.0
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource exhausted" in err_str:
                raise RetryableAPIError(wait_hint=wait) from e
            raise

    elif kind == 'groq':
        from openai import APIConnectionError, APIStatusError, RateLimitError
        try:
            response = cli.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=ARGS.temp,
                max_tokens=1200,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError) as e:
            wait = extract_retry_after(e) or 2.0
            raise RetryableAPIError(wait_hint=wait) from e
        except APIStatusError as e:
            wait = extract_retry_after(e) or 2.0
            if e.status_code in (429, 500, 502, 503, 504):
                raise RetryableAPIError(wait_hint=wait) from e
            raise FatalAPIError(f"Groq HTTP Error {e.status_code}: {e}") from e


def clean_and_validate_variants(raw_text: str, original_text: str) -> List[str]:
    if not raw_text:
        return []

    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()

    candidate_list: List[str] = []
    if cleaned.startswith('{'):
        try:
            parsed = json.loads(cleaned)
            v = parsed.get('variants') or parsed.get('implicit_variations') or parsed.get('rewritten')
            if isinstance(v, list):
                candidate_list = [str(x).strip() for x in v if str(x).strip()]
            elif isinstance(v, str) and v.strip():
                candidate_list = [v.strip()]
        except json.JSONDecodeError:
            pass
    elif cleaned.startswith('['):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                candidate_list = [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    if not candidate_list:
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        for l in lines:
            # Strip markdown list bullets like "1. ", "- ", "* " safely without eating digits in sentences
            stripped_line = re.sub(r'^\s*(?:\d+[\.\)]|[-*•])\s*', '', l).strip()
            if len(stripped_line) > 10 and not stripped_line.startswith('{') and not stripped_line.endswith('}'):
                candidate_list.append(stripped_line)

    valid_variants: List[str] = []
    lower_orig = original_text.lower().strip()

    for item in candidate_list:
        item_str = item.strip().strip('"').strip("'")
        # Sanitize tabs and newlines to keep TSV format clean and unquoted
        item_str = re.sub(r'[\t\r\n]+', ' ', item_str).strip()

        if not item_str or len(item_str) < 5:
            continue

        lower_item = item_str.lower()
        if lower_item == lower_orig:
            continue

        # Check for genuine refusal patterns
        is_refusal = any(re.search(pat, lower_item) for pat in REFUSAL_PATTERNS)
        if is_refusal:
            continue

        if item_str not in valid_variants:
            valid_variants.append(item_str)

    return valid_variants


def process_single_row(row: Dict[str, Any]) -> Tuple[str, List[str], str]:
    try:
        system_prompt, user_prompt = build_prompt_payload(row)
    except Exception as e:
        print(f"  [Error building prompt for {row.get('id')}]: {e}", file=sys.stderr)
        return row['id'], [], 'fatal'

    for attempt in range(1, ARGS.max_retries + 1):
        try:
            raw_out = call_llm(system_prompt, user_prompt)
            variants = clean_and_validate_variants(raw_out, row['text'])
            if variants:
                return row['id'], variants, 'ok'
            return row['id'], [], 'empty_or_invalid'
        except RetryableAPIError as e:
            if attempt >= ARGS.max_retries:
                break
            wait_time = e.wait_hint or min(45.0, (1.8 ** attempt)) + random.uniform(0.1, 0.5)
            time.sleep(wait_time)
        except FatalAPIError as e:
            print(f"  [FATAL] ID {row['id']}: {e}", file=sys.stderr)
            return row['id'], [], 'fatal'
        except Exception as e:
            if attempt >= ARGS.max_retries:
                print(f"  [Error] ID {row['id']} failed after {attempt} retries ({type(e).__name__}: {e})", file=sys.stderr)
                break
            time.sleep(1.5 * attempt)

    return row['id'], [], 'gave_up'


# -----------------------------------------------------------------------------
# IO & TSV EXPORT
# -----------------------------------------------------------------------------

def atomic_save_json(path: str, data: Any):
    dir_name = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_canonical_tsv(lang: str, items: List[Dict[str, Any]], out_tsv_path: str):
    """Writes clean, deterministic, unquoted TSV lines conforming to SemEval schema."""
    os.makedirs(os.path.dirname(os.path.abspath(out_tsv_path)) or '.', exist_ok=True)
    with open(out_tsv_path, 'w', encoding='utf-8', newline='') as f:
        f.write('\t'.join(HEADER) + '\n')
        # Sort items stably by source_id so order is deterministic across snapshots
        for item in sorted(items, key=lambda x: x.get('source_id') or x.get('id', '')):
            source_id = item.get('source_id') or item.get('id', '')
            variants = item.get('variants', [])
            if not variants and item.get('rewritten'):
                variants = [item['rewritten']]

            for v_idx, variant_text in enumerate(variants, start=1):
                # Clean tabs and newlines to preserve raw TSV structure
                clean_variant = re.sub(r'[\t\r\n]+', ' ', variant_text).strip()
                clean_title = re.sub(r'[\t\r\n]+', ' ', item.get('yt_title', '')).strip()
                clean_desc = re.sub(r'[\t\r\n]+', ' ', item.get('yt_description', '')).strip()
                clean_stereo = item.get('stereotype', 'no').strip()
                clean_target = item.get('target', 'none').strip()

                # Stable, deterministic Synthetic ID derived from source_id
                synth_id = f"synth_{source_id}_v{v_idx}" if len(variants) > 1 else f"synth_{source_id}"
                row_fields = [
                    synth_id,
                    clean_title,
                    clean_desc,
                    clean_variant,
                    clean_stereo,
                    'yes_implicit',
                    clean_target
                ]
                f.write('\t'.join(row_fields) + '\n')


# -----------------------------------------------------------------------------
# MAIN LIFECYCLE
# -----------------------------------------------------------------------------

def main():
    print(f"================================================================")
    print(f" StereoQueerEval 2027: Multi-Variant Implicit Hate Augmentation")
    print(f"================================================================")

    paths = find_input_paths()
    if not paths:
        sys.exit(f"No original dataset found matching patterns: {DEFAULT_INPUT_GLOBS}")

    print(f"Found input files: {paths}")
    rows = read_source_rows(paths)
    if not rows:
        sys.exit("No matching 'yes_explicit' rows found in the specified dataset.")

    from collections import Counter
    lang_counter = Counter(r['lang'] for r in rows)

    if ARGS.dry_run:
        print("\n[DRY RUN] Explicit rows available for Implicit generation:")
        for lang, count in sorted(lang_counter.items()):
            est_samples = count * ARGS.n_variants
            print(f"  - Language {lang}: {count} explicit comments -> ~{est_samples} synthetic implicit samples ({ARGS.n_variants} per prompt)")
        print(f"Total available: {len(rows)} comments -> ~{len(rows) * ARGS.n_variants} total synthetic samples")
        return

    # Load existing cache
    existing_items = load_existing(ARGS.out) if ARGS.resume else {}
    done_ids: Set[str] = set(existing_items.keys())
    todo_rows = [r for r in rows if r['id'] not in done_ids]

    if ARGS.limit:
        todo_rows = todo_rows[:ARGS.limit]

    print(f"Total Explicit: {len(rows)} | Already Done: {len(done_ids)} | To Process: {len(todo_rows)}")
    print(f"Variants per prompt: {ARGS.n_variants} (Yielding ~{len(todo_rows) * ARGS.n_variants} implicit samples)")

    if not todo_rows:
        print("All matching samples are already processed. Exiting.")
        return

    # Setup LLM Client
    kind, cli = init_client()
    _CTX['client'] = (kind, cli)
    _CTX['model'] = get_default_model(kind)
    concurrency = ARGS.max_concurrency or (8 if kind == 'gemini' else 4)

    # Compute rate pacer delay (cooldown between requests)
    effective_delay = ARGS.delay
    if ARGS.rpm and ARGS.rpm > 0:
        effective_delay = max(effective_delay, 60.0 / ARGS.rpm)
    if effective_delay > 0:
        print(f"Rate Limiter Active: {effective_delay:.2f}s cooldown pause between requests (~{60.0/effective_delay:.1f} RPM)")

    print(f"API Provider: {kind} | Model: {_CTX['model']} | Parallel Workers: {concurrency}")

    results: Dict[str, Dict[str, Any]] = dict(existing_items)
    stats = {'ok': 0, 'empty_or_invalid': 0, 'fatal': 0, 'gave_up': 0}
    row_lookup = {r['id']: r for r in rows}

    def persist_snapshot():
        total_variants_count = sum(len(it.get('variants', [1])) for it in results.values())
        meta = {
            'generator': 'llm_implicit_augment.py',
            'api': ARGS.api,
            'model': _CTX['model'],
            'n_variants_per_prompt': ARGS.n_variants,
            'updated_at': now_iso(),
            'source_comments_count': len(results),
            'total_synthetic_samples_count': total_variants_count,
            'items': [results[k] for k in sorted(results.keys())]
        }
        atomic_save_json(ARGS.out, meta)

        by_lang: Dict[str, List[Dict[str, Any]]] = {}
        for it in results.values():
            by_lang.setdefault(it['lang'], []).append(it)

        for lang, items in by_lang.items():
            tsv_path = os.path.join(ARGS.tsv_dir, f"SynthImplicit_{lang}{ARGS.txt_suffix}")
            write_canonical_tsv(lang, items, tsv_path)

        return total_variants_count

    try:
        t0 = time.time()
        processed_count = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for i in range(0, len(todo_rows), concurrency):
                chunk = todo_rows[i:i + concurrency]
                future_map = {executor.submit(process_single_row, r): r for r in chunk}

                for future in as_completed(future_map):
                    r_id, variants_list, status = future.result()
                    stats[status] = stats.get(status, 0) + 1
                    processed_count += 1

                    if status == 'ok' and variants_list:
                        src = row_lookup[r_id]
                        results[r_id] = {
                            'source_id': r_id,
                            'lang': src['lang'],
                            'yt_title': src.get('yt_title', ''),
                            'yt_description': src.get('yt_description', ''),
                            'stereotype': src['stereotype'],
                            'source_hate_speech': 'yes_explicit',
                            'generated_hate_speech': 'yes_implicit',
                            'target': src['target'],
                            'original': src['text'],
                            'variants': variants_list,
                            'variants_count': len(variants_list),
                            'generated_at': now_iso(),
                        }

                    if processed_count % 15 == 0 or processed_count == len(todo_rows):
                        total_var_saved = persist_snapshot()
                        elapsed = time.time() - t0
                        speed = processed_count / max(elapsed, 0.001)
                        print(f"  Progress: {processed_count}/{len(todo_rows)} prompts ({speed:.1f} prompts/s) | "
                              f"Total Synthetic Rows: {total_var_saved} | Ok: {stats['ok']} | Errors: {stats['empty_or_invalid'] + stats['gave_up']}",
                              flush=True)

                if effective_delay > 0 and (i + concurrency) < len(todo_rows):
                    time.sleep(effective_delay * len(chunk))

    except KeyboardInterrupt:
        print("\n[Ctrl+C detected] Gracefully saving generated items before exit...", flush=True)
    finally:
        total_var_saved = persist_snapshot()
        print(f"\n================================================================")
        print(f" Generation Complete!")
        print(f" Total Source Prompts Processed: {len(results)}")
        print(f" Total Synthetic Implicit Rows Generated: {total_var_saved}")
        print(f" JSON Registry: {ARGS.out}")
        print(f" TSV Directory: {ARGS.tsv_dir}/SynthImplicit_{{LANG}}_training.tsv")
        print(f" Execution Stats: {stats}")
        print(f"================================================================")


if __name__ == '__main__':
    main()
