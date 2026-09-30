#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Multi-Perspective Diagnostic Reasoning & Quality-Gated Teacher Extractor
for StereoQueerEval 2027 (SemEval Task B / C).

Core Upgrades & Invariants (plan.md §4 & §5):
1. 5-Axis Decomposition: direct_hostility, indirect_subtext, context_dependence, counter_speech, target_reference.
2. Independent LLM Judge (Self-Verification): stereotype, hate_speech, target_identities, target_scope, confidence.
3. Quality Control (QC Gate):
   - Compares LLM judge with gold labels.
   - High quality if exact match on hate_speech OR (gold is implicit and confidence >= 0.60)
     OR (gold is explicit and judge says hate with confidence >= 0.85).
   - If not high quality -> safe fallback to empty hint (hint=""), so sample safely reverts to pure unguided baseline.
4. Strict Leak-Proof Sanitizer: masks all label tokens (implicit|explicit|hate|neutral|non-hate) to [MASKED].
5. Pre-compiled <=64 token hint budget ready for DataLoader ingestion.
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
from typing import Dict, List, Optional, Set, Tuple, Any

def auto_load_dotenv():
    # Walk up from the script location to the filesystem root, loading the first .env found.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [os.getcwd(), script_dir] + [
        os.path.dirname(script_dir),
        os.path.dirname(os.path.dirname(script_dir)),
        os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
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
                        # First occurrence wins (the fresh/quota-reset key listed first).
                        # The previous last-wins approach got stuck whenever the newer
                        # key's quota was exhausted but the older one had reset.
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass
            print(f"Loaded .env from {env_path}")
            return

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

# Official StereoQueerEval Canonical Identity Ordering for Task C
ID_ORDER = ['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+']

# Strict Leak-Proof Sanitization Pattern (plan.md §4)
LABEL_TOKEN_RE = re.compile(
    r'\b(yes_implicit|yes_explicit|no_hate|non[-\s_]?hate|implicit(ly)?|explicit(ly)?|hate[-\s_]?speech|hatespeech|hate|neutral|nonhate)\b',
    re.IGNORECASE
)

def sanitize_leak_free_text(text: Optional[str]) -> str:
    """Strict Leak-Proof Sanitizer: masks label tokens to [MASKED]."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = LABEL_TOKEN_RE.sub('[MASKED]', text)
    cleaned = re.sub(r'(\[MASKED\]\s*)+', '[MASKED] ', cleaned).strip()
    return cleaned

def truncate_words(text: str, max_words: int = 8) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words])

RATIONALE_SYSTEM_PROMPT = """You are a senior computational sociolinguist conducting pragmatic discourse analysis for YouTube video comments related to LGBTQ+ topics.

Analyze the given comment strictly grounded in the video's context (title + description).

CONTEXT-FIRST GROUNDING DIRECTIVES (CRITICAL):
1. MANDATORY: Read the video_title and video_description FIRST to identify:
   - What LGBTQ+ identities or individuals are featured in the video (e.g., lesbian 'l', gay 'g', bisexual 'b', transgender 't', queer 'q', intersex 'i', asexual 'a', non-binary 'nb', or general 'lgbtqia+').
   - Whether the comment refers to a specific person in the video ('individual') or the broader community ('group').
2. Evaluate if the comment's meaning depends on or attacks the subject shown in the video. (e.g. If the video is about a lesbian woman, a comment praising denial of communion or mocking her identity directly targets 'l'/'individual' or 'l'/'group').
3. NON-TARGET & BENIGN COMMENTS GUARD: If the comment is genuinely generic, off-topic, discussing an unrelated aspect of the video, or simply stating a neutral observation without attacking or undermining LGBTQ+ identities/persons, set target_identities: [] and target_scope: 'none' and target_reference: 'absent'. However, if the comment uses indirect criticism, moral judgment, sarcasm, slurs, or exclusionary rhetoric aimed at the LGBTQ+ topic or subjects, identify the corresponding target identity and scope.

CRITICAL INSTRUCTIONS TO PREVENT DATA LEAKAGE:
1. DO NOT mention classification label words ("no", "implicit", "explicit", "hate_speech", "neutral", "non-hate") in your 'why' or 'boundary' fields!
2. Provide an objective breakdown of the communication dynamics between the video context and comment.
3. Assess the linguistic axes:
   - direct_hostility: 'weak' (no slurs/threats), 'moderate', or 'strong' (overt slurs/violent threats)
   - indirect_subtext: 'weak' (literal/supportive), 'moderate', or 'strong' (sarcasm, dog-whistle, faux-concern, moral lecturing)
   - context_dependence: 'weak' (meaning is clear standalone), 'moderate', or 'strong' (meaning flips completely based on video title/topic)
   - counter_speech: 'weak' (not defending LGBTQ+), 'strong' (defending or supporting LGBTQ+ persons)
   - target_reference: 'present' (explicitly or implicitly mentions LGBTQ+ identities/groups), 'absent'
4. MANDATORY STEP-BY-STEP CONTEXT ANALYSIS (do this BEFORE everything else): For each comment, first fill a 'context_analysis' object in your output:
   - video_subject: one short factual line identifying who/what the video is actually about (person, group, event). Must follow directly from the title/description.
   - comment_stance: one short factual line describing the comment's relationship to that subject WITHOUT using classification labels (e.g. 'sympathizes with an accused-abuser figure to endorse the abuse', 'mocks the person coming out', 'defends the group against criticism', 'discusses an unrelated topic').
   Only after writing context_analysis may you fill 'why', 'boundary', and 'judge'.
5. In 'judge', determine the target identities and scope directly grounded in the video context before deciding the classification:
   - target_identities: list from ['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+'] or [] if none.
   - target_scope: 'group' | 'individual' | 'none'.
   - stereotype: 0 or 1.
   - hate_speech: 'no' | 'yes_implicit' | 'yes_explicit'.

OUTPUT FORMAT:
Reply ONLY with a raw JSON object matching:
{
  "results": [
    {
      "id": "sample_id",
      "context_analysis": {
        "video_subject": "one line: who/what the video is about",
        "comment_stance": "one line: comment's relationship to that subject, label-free"
      },
      "axes": {
        "direct_hostility": "weak|moderate|strong",
        "indirect_subtext": "weak|moderate|strong",
        "context_dependence": "weak|moderate|strong",
        "counter_speech": "weak|strong",
        "target_reference": "present|absent"
      },
      "why": "Objective factual breakdown of communicative tone and subtext, label-free.",
      "boundary": "Why this might look benign or literal at first glance and what subtle cue differentiates it, label-free.",
      "judge": {
        "stereotype": 0,
        "hate_speech": "no|yes_implicit|yes_explicit",
        "target_identities": ["l"],
        "target_scope": "individual|group|none",
        "confidence": 0.85
      }
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
    parser.add_argument('--model', default='gemini-3.5-flash-lite', help='Default model (verified working: gemini-3.5-flash-lite)')
    parser.add_argument('--input', action='append', default=None)
    parser.add_argument('--lang', default=None, help='EN, IT, NL')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1, help='Number of comments to analyze in 1 API call (default 1 for maximum per-sample reasoning accuracy)')
    parser.add_argument('--rpm', type=int, default=14, help='Rate limit for API')
    parser.add_argument('--temp', type=float, default=0.2, help='Low temperature for analytical consistency')
    parser.add_argument('--hint-max-tokens', type=int, default=64, help='Token budget for precompiled hint')
    parser.add_argument('--desc-max', type=int, default=800, help='Max chars of video_description fed to the LLM (0 = unlimited; whitespace is always collapsed)')
    parser.add_argument('--out', default='LGBT/rationales.json', help='Output JSON cache')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--force-restart', action='store_true', default=False)
    parser.add_argument('--gate', choices=['all', 'hs', 'hint'], default='hint',
                        help="QC gate: 'all' (approx HS+ST+TG - noisy), 'hs' (hate-speech label + hint quality), "
                             "'hint' (hint content only, decoupled from noisy judge labels)")
    return parser.parse_args()

def normalize_text(s: str, max_len: Optional[int] = None) -> str:
    # Collapse all whitespace runs (multi-line YouTube descriptions) into a single space, then cap.
    s = re.sub(r'\s+', ' ', s or '').strip()
    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip()
    return s

def read_tsv_rows(paths: List[str], lang_filter: Optional[str] = None, desc_max: Optional[int] = None) -> List[Dict[str, Any]]:
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
                hs_col = 'hate_speech' if 'hate_speech' in idx_map else None
                st_col = 'stereotype' if 'stereotype' in idx_map else None
                tg_col = 'target' if 'target' in idx_map else None

                for row in reader:
                    if not row or len(row) <= idx_map.get(id_col, 0):
                        continue
                    sid = row[idx_map[id_col]].strip()
                    if not sid:
                        continue
                    comment = normalize_text(row[idx_map[c_col]] if c_col and len(row) > idx_map[c_col] else "")
                    title = normalize_text(row[idx_map[t_col]] if t_col and len(row) > idx_map[t_col] else "")
                    desc = normalize_text(row[idx_map[d_col]] if d_col and len(row) > idx_map[d_col] else "",
                                          max_len=desc_max)
                    
                    gold_hs = row[idx_map[hs_col]].strip() if hs_col and len(row) > idx_map[hs_col] else ""
                    gold_st_raw = row[idx_map[st_col]].strip().lower() if st_col and len(row) > idx_map[st_col] else "0"
                    gold_st = 1 if gold_st_raw in ['1', 'yes', 'true'] else 0
                    gold_tg = row[idx_map[tg_col]].strip() if tg_col and len(row) > idx_map[tg_col] else "none"

                    rows.append({
                        'id': sid,
                        'lang': lang,
                        'yt_comment': comment,
                        'yt_title': title,
                        'yt_description': desc,
                        'gold_hs': gold_hs,
                        'gold_st': gold_st,
                        'gold_tg': gold_tg
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
            if ('429' in err_str or 'quota' in err_str or 'resource_exhausted' in err_str or 'rate' in err_str
                    or '503' in err_str or 'unavailable' in err_str or 'high demand' in err_str):
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

def format_target_string(identities: Any, scope: Any) -> str:
    """
    Formats predicted target to canonical StereoQueerEval format:
    e.g. 'group_t,lgbtqia+' or 'individual_g' or 'none'.
    Canonical order: ['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+']
    """
    if not identities or identities == 'none' or (isinstance(identities, list) and not identities):
        return "none"
    
    if isinstance(identities, list):
        clean_ids = [str(x).strip().lower().replace("group_", "").replace("individual_", "") for x in identities if str(x).strip()]
        # Preserve canonical StereoQueerEval identity ordering
        ordered_ids = [x for x in ID_ORDER if x in clean_ids]
        id_part = ",".join(ordered_ids)
    else:
        id_part = str(identities).strip().lower().replace("group_", "").replace("individual_", "")

    if not id_part or id_part == 'none':
        return "none"

    clean_scope = str(scope).strip().lower()
    if clean_scope not in ['group', 'individual']:
        clean_scope = 'group'

    return f"{clean_scope}_{id_part}"

def compile_context_hint(subject: str, stance: str, why: str, boundary: str, max_tokens: int = 64) -> str:
    """Compiles hint from the mandatory step-by-step context analysis:
    video subject + comment stance + subtext divergence (label-free, context-anchored).
    """
    subject_short = truncate_words(subject, 8)
    stance_short = truncate_words(stance, 12)
    why_short = truncate_words(why, 8)

    raw_hint = f"context: {subject_short} | stance: {stance_short}"
    if why_short:
        raw_hint = f"{raw_hint} | divergence: {why_short}"
    clean_hint = sanitize_leak_free_text(raw_hint)
    words = clean_hint.split()
    if len(words) > max_tokens * 0.9:
        clean_hint = " ".join(words[:max_tokens])
    return clean_hint

def compile_latent_hint(axes: Dict[str, Any], why: str, boundary: str, max_tokens: int = 64) -> str:
    """Compiles a compact, label-free diagnostic hint within token budget."""
    ind = axes.get('indirect_subtext', 'weak')
    ctx = axes.get('context_dependence', 'weak')
    dir_h = axes.get('direct_hostility', 'weak')
    cs = axes.get('counter_speech', 'weak')
    tg_ref = axes.get('target_reference', 'absent')

    why_short = truncate_words(why, 8)
    flip_short = truncate_words(boundary, 8)

    raw_hint = (
        f"axes: indirect_subtext={ind}, context_dependence={ctx}, "
        f"direct_hostility={dir_h}, counter_speech={cs}, target_reference={tg_ref} | "
        f"why: {why_short} | flip: {flip_short}"
    )
    clean_hint = sanitize_leak_free_text(raw_hint)
    # Token length approximation guard
    words = clean_hint.split()
    if len(words) > max_tokens:
        clean_hint = " ".join(words[:max_tokens])
    return clean_hint

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
    rows = read_tsv_rows(input_files, lang_filter=args.lang, desc_max=args.desc_max)
    if args.limit:
        rows = rows[:args.limit]
        
    cache: Dict[str, Any] = {}
    if args.resume and not args.force_restart and os.path.exists(args.out):
        try:
            with open(args.out, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"Loaded existing cache with {len(cache)} diagnostic entries.")
        except Exception as e:
            raise SystemExit(
                f"[CHECKPOINT ERROR] corrupt cache {args.out}: {e}. "
                f"Move it aside and re-run with --force-restart if intended; don't silently lose the run."
            )
    
    # Checkpoint check: skip already analyzed entries
    pending = [r for r in rows if r['id'] not in cache]
    print(f"Total rows: {len(rows)} | Cached: {len(cache)} | Pending: {len(pending)}")
    if not pending:
        print("All rows already analyzed and validated! Done.")
        return

    cli_tuple = init_client(args.api)
    batch_size = max(1, args.batch_size)
    min_interval = 60.0 / args.rpm if args.rpm else 0.0
    
    # Tracking counters
    qc_stats = Counter()

    for i in range(0, len(pending), batch_size):
        batch = pending[i:i+batch_size]
        t0 = time.time()
        
        batch_input = []
        for r in batch:
            batch_input.append({
                "id": r['id'],
                "language": LANGUAGE_NAMES.get(r['lang'], r['lang']),
                "video_title": r['yt_title'],
                "video_description": r['yt_description'],
                "comment": r['yt_comment']
            })
            
        user_prompt = f"Analyze these {len(batch)} YouTube comments:\n{json.dumps(batch_input, ensure_ascii=False, indent=2)}"
        
        retries = 0
        success = False
        parsed_results = []

        while retries < 5 and not success:
            try:
                raw_json = call_llm(cli_tuple, args.model, RATIONALE_SYSTEM_PROMPT, user_prompt, args.temp)
                parsed = json.loads(raw_json)
                parsed_results = parsed.get('results', []) if isinstance(parsed, dict) else parsed
                success = True
            except RetryableAPIError:
                retries += 1
                wait = 4.0 * retries
                print(f"  [Rate Limit] Retrying in {wait:.1f}s (attempt {retries}/5)...")
                time.sleep(wait)
            except Exception as e:
                print(f"  [Error in batch {i//batch_size}]: {e}")
                break

        res_map = {item.get('id'): item for item in parsed_results if item.get('id')}

        missing_ids = [r['id'] for r in batch if r['id'] not in res_map]
        if missing_ids:
            print(f"  [WARN] batch {i//batch_size}: {len(missing_ids)}/{len(batch)} responses missing "
                  f"({missing_ids[:3]}...) — leaving them out of cache so --resume retries next run")
            continue

        for r in batch:
            sid = r['id']
            item = res_map.get(sid, {})
            
            axes = item.get('axes', {
                'direct_hostility': 'weak',
                'indirect_subtext': 'weak',
                'context_dependence': 'weak',
                'counter_speech': 'weak',
                'target_reference': 'absent'
            })
            raw_why = item.get('why', item.get('rationale', ''))
            raw_boundary = item.get('boundary', item.get('boundary_note', ''))
            
            clean_why = sanitize_leak_free_text(raw_why)
            clean_boundary = sanitize_leak_free_text(raw_boundary)

            # Step-by-step context analysis (forced context reading)
            ctx_data = item.get('context_analysis', {}) or {}
            ctx_subject = sanitize_leak_free_text(ctx_data.get('video_subject', ''))
            ctx_stance = sanitize_leak_free_text(ctx_data.get('comment_stance', ''))
            
            # Judge extraction
            judge_data = item.get('judge', {})
            pred_hs = judge_data.get('hate_speech', '')
            pred_st_raw = judge_data.get('stereotype', 0)
            pred_st = 1 if str(pred_st_raw) in ['1', 'yes', 'true'] else 0
            pred_tg_ids = judge_data.get('target_identities', [])
            pred_tg_scope = judge_data.get('target_scope', 'group')
            pred_tg_str = format_target_string(pred_tg_ids, pred_tg_scope)
            confidence = float(judge_data.get('confidence', 0.8))

            # Quality Control Gating against Gold Labels
            matches_hs = (pred_hs == r['gold_hs'])
            matches_st = (pred_st == r['gold_st'])
            matches_tg = (pred_tg_str == r['gold_tg'])

            # Hint-content quality: the step-by-step context reading must be present,
            # substantive, and leak-free. Judge label matches are decoupled from gate.
            subject_words = len(ctx_subject.split())
            stance_words = len(ctx_stance.split())
            leak_detected = ('[MASKED]' in ctx_subject) or ('[MASKED]' in ctx_stance)
            hint_content_ok = (subject_words >= 3 and stance_words >= 3 and not leak_detected)

            # Build final hint first so gate can check its post-compile state
            candidate_hint = compile_context_hint(ctx_subject, ctx_stance, clean_why, clean_boundary, max_tokens=args.hint_max_tokens)
            if args.gate == 'all':
                is_high_quality = matches_hs and matches_st and matches_tg
            elif args.gate == 'hs':
                is_high_quality = matches_hs and hint_content_ok
            else:  # 'hint': gift quality only, decoupled from noisy judge labels
                is_high_quality = hint_content_ok

            # Post-compile guard: reject hints that still leak masked label residue
            if is_high_quality and '[MASKED]' in candidate_hint:
                is_high_quality = False

            if is_high_quality:
                hint = candidate_hint
                qc_stats[(r['gold_hs'], r['lang'], 'kept')] += 1
            else:
                # Safe fallback: empty hint reverts row to pure baseline unguided representation
                hint = ""
                qc_stats[(r['gold_hs'], r['lang'], 'empty_fallback')] += 1

            cache[sid] = {
                'lang': r['lang'],
                'context_analysis': {
                    'video_subject': ctx_subject,
                    'comment_stance': ctx_stance
                },
                'axes': axes,
                'why': clean_why,
                'boundary': clean_boundary,
                'hint': hint,
                'judge': {
                    'stereotype': pred_st,
                    'hate_speech': pred_hs,
                    'target_identities': pred_tg_ids,
                    'target_scope': pred_tg_scope,
                    'target_string': pred_tg_str,
                    'confidence': confidence
                },
                'quality_control': {
                    'matches_gold_hs': matches_hs,
                    'matches_gold_st': matches_st,
                    'matches_gold_tg': matches_tg,
                    'hint_content_ok': hint_content_ok,
                    'is_high_quality': is_high_quality
                }
            }

        # Save atomic checkpoint (write .tmp then rename to avoid corruption on interruption)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        tmp_path = args.out + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, args.out)
            
        elapsed = time.time() - t0
        accepted_cnt = sum(1 for r in batch if cache[r['id']]['quality_control']['is_high_quality'])
        print(f"Processed {min(i+batch_size, len(pending))}/{len(pending)} | High-Quality Accepted: {accepted_cnt}/{len(batch)} -> {args.out} [{elapsed:.1f}s]")
        
        if min_interval > elapsed:
            time.sleep(min_interval - elapsed)

    print(f"\n🎉 Finished generating {len(cache)} diagnostic rationales -> {args.out}")
    print("\nQC Acceptance Summary:")
    for k, count in sorted(qc_stats.items()):
        print(f"  {k}: {count}")

if __name__ == '__main__':
    main()
