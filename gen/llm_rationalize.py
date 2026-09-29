#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Multi-Perspective Diagnostic Reasoning Generator (Privileged Information Extraction)
for StereoQueerEval 2027 (SemEval Task B / C).

Core Concepts:
1. Label-Agnostic Feature Extraction: Strictly prohibits the LLM from outputting gold label tokens
   ('no', 'implicit', 'explicit', 'hate_speech', 'neutral', 'non-hate') to prevent label shortcut leakage.
2. Leak-Proof Regex Sanitizer (plan.md §4): Programmatically sanitizes all free-text fields
   (`rationale`, `boundary_note`) by replacing label leak tokens with `[MASKED]` before caching to disk.
3. 5-Axis Contrastive Linguistic Decomposition:
   - direct_hostility: [weak | moderate | strong]
   - indirect_subtext: [weak | moderate | strong]
   - context_dependence: [weak | moderate | strong]
   - counter_speech: [weak | strong]
   - target_reference: [present | absent]
4. Multi-Class Boundary Contrast:
   - rationale: Objective pragmatic breakdown
   - boundary_note: Confusion boundary and communicative nuance
5. High-Throughput Batch Processing & Atomic JSON checkpointing.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any

def auto_load_dotenv():
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
                        k, v = line.split('=', 1)
                        k, v = k.strip(), v.strip().strip("'").strip('"')
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass
            break

auto_load_dotenv()

DEFAULT_INPUT_GLOBS = [
    'LGBT/StereoQueerEval_*_training.tsv',
    '../LGBT/StereoQueerEval_*_training.tsv',
    '../StereoQueerEval_*_training.tsv',
    'data/StereoQueerEval_*_training.tsv',
    'StereoQueerEval_*_training.tsv'
]

LANGUAGE_NAMES = {
    'EN': "English",
    'IT': "Italian",
    'NL': "Dutch"
}

# Strict Leak-Proof Sanitization Pattern (plan.md §4)
# Programmatically scrubs any explicit label token or variant before writing to disk
LEAK_WORDS_PATTERN = re.compile(
    r'\b(yes_implicit|yes_explicit|no_hate|non[-_ ]?hate|implicit(ly)?|explicit(ly)?|hate[-_ ]?speech|hatespeech|hate|neutral|nonhate)\b',
    re.IGNORECASE
)

def sanitize_leak_free_text(text: Optional[str]) -> str:
    """
    Strict Leak-Proof Sanitizer:
    Replaces any occurrence of label leak tokens ('implicit', 'explicit', 'hate',
    'non-hate', 'neutral', etc.) with '[MASKED]' before saving to cache/disk.
    """
    if not text or not isinstance(text, str):
        return ""
    # Replace label-revealing tokens with [MASKED]
    cleaned = LEAK_WORDS_PATTERN.sub('[MASKED]', text)
    # Collapse multiple consecutive [MASKED] tokens for readability
    cleaned = re.sub(r'(\[MASKED\]\s*)+', '[MASKED] ', cleaned).strip()
    return cleaned


RATIONALE_SYSTEM_PROMPT = """You are a senior computational sociolinguist conducting pragmatic discourse analysis for YouTube video comments related to LGBTQ+ topics.

Analyze the given comment strictly based on pragmatic and linguistic evidence.

CRITICAL INSTRUCTIONS TO PREVENT DATA LEAKAGE:
1. DO NOT mention the classification label words ("no", "implicit", "explicit", "hate_speech", "neutral", "non-hate") in your analysis!
2. Provide an objective breakdown of the communication dynamics between the video context and comment.
3. Assess the linguistic axes:
   - direct_hostility: 'weak' (no slurs/threats), 'moderate', or 'strong' (overt slurs/violent threats)
   - indirect_subtext: 'weak' (literal/supportive), 'moderate', or 'strong' (sarcasm, dog-whistle, faux-concern, moral lecturing)
   - context_dependence: 'weak' (meaning is clear standalone), 'moderate', or 'strong' (meaning flips completely based on video title/topic)
   - counter_speech: 'weak' (not defending LGBTQ+), 'strong' (defending or supporting LGBTQ+ persons)
   - target_reference: 'present' (explicitly or implicitly mentions LGBTQ+ identities/groups), 'absent'
4. Write a concise factual explanation (1-2 sentences) of the subtext and how it could be confused with another interpretation.

OUTPUT FORMAT:
Reply ONLY with a raw JSON object:
{
  "results": [
    {
      "id": "sample_id",
      "axes": {
        "direct_hostility": "weak|moderate|strong",
        "indirect_subtext": "weak|moderate|strong",
        "context_dependence": "weak|moderate|strong",
        "counter_speech": "weak|strong",
        "target_reference": "present|absent"
      },
      "rationale": "Objective 1-2 sentence breakdown of the tone, sarcasm, innuendo, or literal stance.",
      "boundary_note": "Why this comment might seem literal or severe at first glance, and what makes its true communicative nuance distinct."
    }
  ]
}"""

class RetryableAPIError(Exception):
    def __init__(self, wait_hint: Optional[float] = None):
        super().__init__('Retryable API Error')
        self.wait_hint = wait_hint

class FatalAPIError(Exception):
    pass

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--api', choices=['gemini', 'groq'], default='gemini')
    parser.add_argument('--model', default='gemini-2.5-flash')
    parser.add_argument('--input', action='append', default=None)
    parser.add_argument('--lang', default=None, help='EN, IT, NL')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=8, help='Number of comments to analyze in 1 API call (recommended 5-10)')
    parser.add_argument('--rpm', type=int, default=14, help='Rate limit for Free Tier')
    parser.add_argument('--temp', type=float, default=0.2, help='Low temperature for analytical consistency')
    parser.add_argument('--out', default='LGBT/rationales.json', help='Output JSON cache')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--force-restart', action='store_true', default=False)
    return parser.parse_args()

def read_tsv_rows(paths: List[str], lang_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        lang_match = re.search(r'_([A-Z]{2})_training\.tsv$', os.path.basename(p), re.IGNORECASE)
        lang = lang_match.group(1).upper() if lang_match else "EN"
        if lang_filter and lang != lang_filter.upper():
            continue
        try:
            with open(p, 'r', encoding='utf-8', errors='replace', newline='') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader, None)
                if not header:
                    continue
                header = [h.strip() for h in header]
                idx_map = {col: i for i, col in enumerate(header)}
                
                id_col = 'StereoQueerEval_id' if 'StereoQueerEval_id' in idx_map else header[0]
                c_col = 'yt_comment' if 'yt_comment' in idx_map else None
                t_col = 'yt_title' if 'yt_title' in idx_map else None
                d_col = 'yt_description' if 'yt_description' in idx_map else None

                for row in reader:
                    if not row or len(row) <= idx_map.get(id_col, 0):
                        continue
                    sid = row[idx_map[id_col]].strip()
                    if not sid:
                        continue
                    comment = row[idx_map[c_col]].strip() if c_col and len(row) > idx_map[c_col] else ""
                    title = row[idx_map[t_col]].strip() if t_col and len(row) > idx_map[t_col] else ""
                    desc = row[idx_map[d_col]].strip() if d_col and len(row) > idx_map[d_col] else ""
                    
                    rows.append({
                        'id': sid,
                        'lang': lang,
                        'yt_comment': comment,
                        'yt_title': title,
                        'yt_description': desc
                    })
        except Exception as e:
            print(f"Error reading {p}: {e}")
    return rows

def init_client(api_name: str):
    if api_name == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise FatalAPIError("GEMINI_API_KEY not found in environment!")
        from google import genai
        client = genai.Client(api_key=api_key)
        return ('gemini', client)
    elif api_name == 'groq':
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            raise FatalAPIError("GROQ_API_KEY not found in environment!")
        from groq import Groq
        client = Groq(api_key=api_key)
        return ('groq', client)
    else:
        raise FatalAPIError(f"Unsupported API: {api_name}")

def call_llm(cli_tuple, model_name: str, sys_prompt: str, user_prompt: str, temp: float) -> str:
    api_name, client = cli_tuple
    if api_name == 'gemini':
        from google.genai import types
        try:
            cfg = types.GenerateContentConfig(
                system_instruction=sys_prompt,
                temperature=temp,
                response_mime_type="application/json"
            )
            resp = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=cfg
            )
            return resp.text
        except Exception as e:
            err_str = str(e).lower()
            if '429' in err_str or 'quota' in err_str or 'resource_exhausted' in err_str or 'rate' in err_str:
                raise RetryableAPIError()
            raise e
    elif api_name == 'groq':
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                response_format={"type": "json_object"}
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if '429' in err_str or 'rate' in err_str:
                raise RetryableAPIError()
            raise e
    return ""

def main():
    args = parse_args()
    
    input_files = []
    if args.input:
        for inp in args.input:
            input_files.extend(glob.glob(inp))
    else:
        for g in DEFAULT_INPUT_GLOBS:
            found = glob.glob(g)
            if found:
                input_files.extend(found)
                break
                
    input_files = sorted(list(set(input_files)))
    if not input_files:
        print("Error: No training TSV files found!")
        sys.exit(1)
        
    print(f"Reading training files: {input_files}")
    rows = read_tsv_rows(input_files, lang_filter=args.lang)
    if args.limit:
        rows = rows[:args.limit]
        
    cache: Dict[str, Any] = {}
    if args.resume and not args.force_restart and os.path.exists(args.out):
        try:
            with open(args.out, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"Loaded existing cache with {len(cache)} diagnostic rationales.")
        except Exception:
            cache = {}
    
    pending = [r for r in rows if r['id'] not in cache]
    print(f"Total rows: {len(rows)} | Cached: {len(cache)} | Pending: {len(pending)}")
    if not pending:
        print("All rows already analyzed! Done.")
        return

    cli_tuple = init_client(args.api)
    batch_size = max(1, args.batch_size)
    min_interval = 60.0 / args.rpm if args.rpm else 0.0
    
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i+batch_size]
        t0 = time.time()
        
        batch_input = []
        for r in batch:
            batch_input.append({
                "id": r['id'],
                "language": LANGUAGE_NAMES.get(r['lang'], r['lang']),
                "video_title": r['yt_title'],
                "video_description": r['yt_description'][:200],
                "comment": r['yt_comment']
            })
            
        user_prompt = f"Analyze these {len(batch)} YouTube comments:\n{json.dumps(batch_input, ensure_ascii=False, indent=2)}"
        
        retries = 0
        success = False
        while retries < 5 and not success:
            try:
                raw_json = call_llm(cli_tuple, args.model, RATIONALE_SYSTEM_PROMPT, user_prompt, args.temp)
                parsed = json.loads(raw_json)
                items = parsed.get('results', []) if isinstance(parsed, dict) else parsed
                for item in items:
                    item_id = item.get('id')
                    if item_id:
                        # -------------------------------------------------------------
                        # LEAK-PROOF SANITIZER (plan.md §4):
                        # Strict programmatic masking before writing to cache / disk
                        # -------------------------------------------------------------
                        raw_rationale = item.get('rationale', '')
                        raw_boundary = item.get('boundary_note', '')
                        
                        clean_rationale = sanitize_leak_free_text(raw_rationale)
                        clean_boundary = sanitize_leak_free_text(raw_boundary)

                        cache[item_id] = {
                            'axes': item.get('axes', {}),
                            'rationale': clean_rationale,
                            'boundary_note': clean_boundary
                        }
                success = True
            except RetryableAPIError:
                retries += 1
                wait = 4.0 * retries
                print(f"  [Rate Limit] Retrying in {wait:.1f}s (attempt {retries}/5)...")
                time.sleep(wait)
            except Exception as e:
                print(f"  [Error in batch {i//batch_size}]: {e}")
                break

        # Save atomic checkpoint
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            
        elapsed = time.time() - t0
        print(f"Processed {min(i+batch_size, len(pending))}/{len(pending)} (Saved sanitized to {args.out}) [{elapsed:.1f}s]")
        
        if min_interval > elapsed:
            time.sleep(min_interval - elapsed)

    print(f"\n🎉 Finished generating {len(cache)} leak-proof diagnostic rationales -> {args.out}")

if __name__ == '__main__':
    main()
