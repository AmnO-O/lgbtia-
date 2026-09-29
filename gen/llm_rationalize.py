#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Multi-Perspective Diagnostic Reasoning Generator (Privileged Information Extraction)
for StereoQueerEval 2027 (SemEval Task B / C).

Core Concepts:
1. Label-Agnostic Feature Extraction: Strictly prohibits the LLM from outputting gold label tokens
   ('no', 'implicit', 'explicit') to prevent label shortcut leakage.
2. 5-Axis Contrastive Linguistic Decomposition:
   - direct_hostility: [weak | moderate | strong]
   - indirect_subtext: [weak | moderate | strong]
   - context_dependence: [weak | moderate | strong]
   - counter_speech: [weak | moderate | strong]
   - target_reference: [present | absent]
3. Multi-Class Boundary Contrast:
   - why_hostility_present_or_absent: Short factual rationale of the pragmatic subtext
   - confusion_boundary: Why this comment could be mistaken for the nearest alternative class
4. High-Throughput Batch Processing & Robust JSON-checkpointing.
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

RATIONALE_SYSTEM_PROMPT = """You are a senior computational sociolinguist conducting pragmatic discourse analysis for YouTube video comments related to LGBTQ+ topics.

Analyze the given comment strictly based on pragmatic and linguistic evidence.

CRITICAL INSTRUCTIONS TO PREVENT DATA LEAKAGE:
1. DO NOT mention the classification label words ("no", "implicit", "explicit", "hate_speech") in your analysis!
2. Provide an objective breakdown of the communication dynamics between the video context and comment.
3. Assess the linguistic axes:
   - direct_hostility: 'weak' (no slurs/threats), 'moderate', or 'strong' (overt slurs/violent threats)
   - indirect_subtext: 'weak' (literal/neutral/supportive), 'moderate', or 'strong' (sarcasm, dog-whistle, faux-concern, moral lecturing)
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
      "boundary_note": "Why this comment might seem neutral or overt at first glance, and what makes its true tone distinct."
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
                col_idx = {h: i for i, h in enumerate(header)}
                for line in reader:
                    if len(line) <= max(col_idx.values()):
                        continue
                    rows.append({
                        'id': line[col_idx['StereoQueerEval_id']].strip(),
                        'lang': lang,
                        'yt_title': line[col_idx['yt_title']].strip() if 'yt_title' in col_idx else '',
                        'yt_description': line[col_idx['yt_description']].strip() if 'yt_description' in col_idx else '',
                        'yt_comment': line[col_idx['yt_comment']].strip() if 'yt_comment' in col_idx else '',
                        'target': line[col_idx['target']].strip() if 'target' in col_idx else 'none',
                        'stereotype': line[col_idx['stereotype']].strip() if 'stereotype' in col_idx else 'no',
                        'hate_speech': line[col_idx['hate_speech']].strip() if 'hate_speech' in col_idx else 'no'
                    })
        except Exception as e:
            print(f"Error reading {p}: {e}", file=sys.stderr)
    return rows

def init_client(api: str):
    if api == 'gemini':
        from google import genai
        key = os.environ.get('GEMINI_API_KEY')
        if not key:
            raise SystemExit('Missing GEMINI_API_KEY in environment or .env')
        return ('gemini', genai.Client(api_key=key))
    elif api == 'groq':
        from openai import OpenAI
        key = os.environ.get('GROQ_API_KEY')
        if not key:
            raise SystemExit('Missing GROQ_API_KEY in environment or .env')
        return ('groq', OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1'))
    raise ValueError(f"Unknown API: {api}")

def call_llm(cli_tuple, model_name: str, system_prompt: str, user_prompt: str, temp: float) -> str:
    kind, cli = cli_tuple
    if kind == 'gemini':
        from google.genai import types
        from google.genai import errors as genai_errors
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temp,
                max_output_tokens=3000,
                response_mime_type="application/json"
            )
            response = cli.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config
            )
            return response.text or ""
        except genai_errors.APIError as e:
            if getattr(e, 'code', None) in (429, 500, 502, 503, 504):
                raise RetryableAPIError() from e
            raise FatalAPIError(f"Gemini error: {e}") from e
    elif kind == 'groq':
        try:
            resp = cli.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                response_format={"type": "json_object"}
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e).lower():
                raise RetryableAPIError() from e
            raise

def main():
    args = parse_args()
    candidate_paths = args.input or DEFAULT_INPUT_GLOBS
    all_files = sorted({p for pat in candidate_paths for p in glob.glob(pat, recursive=True) if os.path.isfile(p)})
    if not all_files:
        sys.exit(f"No source files found in {candidate_paths}")
    
    rows = read_tsv_rows(all_files, args.lang)
    if not rows:
        sys.exit("No data rows loaded.")
    if args.limit:
        rows = rows[:args.limit]
    
    # Load cache
    cache: Dict[str, Any] = {}
    if args.resume and not args.force_restart and os.path.exists(args.out):
        try:
            with open(args.out, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"Loaded existing cache with {len(cache)} analyzed rationales from {args.out}")
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
                        cache[item_id] = {
                            'axes': item.get('axes', {}),
                            'rationale': item.get('rationale', ''),
                            'boundary_note': item.get('boundary_note', '')
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
        print(f"Processed {min(i+batch_size, len(pending))}/{len(pending)} (Saved to {args.out}) [{elapsed:.1f}s]")
        
        if min_interval > elapsed:
            time.sleep(min_interval - elapsed)

    print(f"\n🎉 Finished generating {len(cache)} diagnostic rationales -> {args.out}")

if __name__ == '__main__':
    main()
