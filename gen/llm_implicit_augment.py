#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate "implicit hate" data from yes_explicit rows of the original dataset via
the Gemini Flash API (google-genai) or Groq (openai SDK), with rate-limit handling.

Outputs:
  - LGBT/outputs.json                       : raw JSON (per row)
  - LGBT/SynthImplicit_{LANG}_training.tsv  : exact 7-column dataset format
    (StereoQueerEval_id | yt_title | yt_description | yt_comment | stereotype
     | hate_speech | target), hate_speech = yes_implicit, stereotype/target inherited.

Usage:
  $env:GEMINI_API_KEY='...'   (or GROQ_API_KEY with --api groq)
  python scripts/llm_implicit_augment.py --api gemini --limit 20
  python scripts/llm_implicit_augment.py --api groq  --model llama-3.3-70b-versatile
  python scripts/llm_implicit_augment.py --resume   # skip ids already processed
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

HEADER = ['StereoQueerEval_id', 'yt_title', 'yt_description', 'yt_comment',
          'stereotype', 'hate_speech', 'target']

DEFAULT_INPUTS = [
    'LGBT/StereoQueerEval_*_training.tsv',
    '../LGBT/StereoQueerEval_*_training.tsv',
    '../StereoQueerEval_*_training.tsv',
]

PROMPT = """You are helping build training data for a hate-speech research task (implicit vs explicit detection).

I give you an EXISTING explicit hate comment from a real YouTube dataset (in {lang}). Rephrase it so the aggression becomes IMPLICIT (veiled / indirect), while keeping:
- the SAME target (person/group being attacked),
- the SAME hostile intent clearly recognizable between the lines,
- the SAME language ({lang}),
- a natural YouTube-comment style (short, informal, may keep tokens like [url], [channel], [email]).

Implicit techniques to choose from: irony, sarcasm, rhetorical question, euphemism, dog-whistle, presupposition, "just asking questions", hypothetical framing.

Forbidden: slurs, direct insults, overt profanity, changing the target, softening into non-hate.

Reply with ONLY a valid JSON object, no markdown, no extra text:
{{"rewritten": "your implicit-hate version"}}

Explicit source:
{text}"""


class RetryableAPIError(Exception):
    def __init__(self, wait_hint=None):
        super().__init__('retryable API error')
        self.wait_hint = wait_hint


class FatalAPIError(Exception):
    pass


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--api', choices=['gemini', 'groq'], default='gemini',
                    help='provider (default gemini)')
    ap.add_argument('--model', default=None,
                    help='model name (default: gemini-2.5-flash / llama-3.3-70b-versatile)')
    ap.add_argument('--input', action='append', default=None,
                    help='file/glob of the source dataset (default: StereoQueerEval_*_training.tsv)')
    ap.add_argument('--lang', default=None,
                    help='process only this language (EN/IT/NL...)')
    ap.add_argument('--limit', type=int, default=None,
                    help='max rows to process (testing; default all)')
    ap.add_argument('--max-concurrency', type=int, default=None,
                    help='max concurrent requests (gemini 8, groq 4)')
    ap.add_argument('--max-retries', type=int, default=8)
    ap.add_argument('--max-chars', type=int, default=1200,
                    help='truncate over-length comments before sending')
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--out', default=r'LGBT/outputs.json',
                    help='output JSON file (default LGBT/outputs.json)')
    ap.add_argument('--tsv-dir', default='LGBT',
                    help='directory for SynthImplicit_{LANG}_training.tsv')
    ap.add_argument('--resume', action='store_true',
                    help='skip ids already present in --out')
    ap.add_argument('--txt-suffix', default='_training.tsv',
                    help='suffix of the TSV output files')
    ap.add_argument('--dry-run', action='store_true',
                    help='only count yes_explicit rows, no API calls')
    return ap.parse_args()


ARGS = parse_args()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_existing(out):
    if not ARGS.resume:
        return set()
    try:
        with open(out, encoding='utf-8') as f:
            data = json.load(f)
        return {it['id'] for it in data.get('items', []) if it.get('id')}
    except (OSError, json.JSONDecodeError):
        return set()


def find_input_paths():
    pats = ARGS.input or DEFAULT_INPUTS
    found = []
    for p in pats:
        if os.path.isdir(p):
            found += glob.glob(os.path.join(p, 'StereoQueerEval_*_training.tsv'))
        else:
            found += glob.glob(p, recursive=True)
    return sorted(set(found))


def read_source_rows(paths):
    rows = []
    for p in paths:
        lang_m = re.search(r'_([A-Z]{2})_training\.tsv$', os.path.basename(p))
        lang = ARGS.lang or (lang_m.group(1) if lang_m else 'UNK')
        with open(p, encoding='utf-8', errors='replace', newline='') as f:
            rd = csv.reader(f, delimiter='\t')
            header = next(rd, None)
            if not header:
                continue
            idx = {name: header.index(name) for name in
                   ['yt_comment', 'stereotype', 'hate_speech', 'target']
                   if name in header}
            if 'yt_comment' not in idx or 'hate_speech' not in idx:
                continue
            if 'StereoQueerEval_id' in header:
                idx['id'] = header.index('StereoQueerEval_id')
            for line in rd:
                if len(line) <= max(idx.values()):
                    continue
                if line[idx['hate_speech']] != 'yes_explicit':
                    continue
                text = line[idx['yt_comment']].strip()
                if not text:
                    continue
                rows.append({
                    'id': line[idx['id']] if 'id' in idx else f'{lang}-{len(rows)}',
                    'lang': lang,
                    'stereotype': line[idx['stereotype']],
                    'target': line[idx['target']] if 'target' in idx else 'none',
                    'text': text[:ARGS.max_chars],
                })
    return rows


def build_prompt(lang, text):
    return PROMPT.format(lang=lang, text=text)


def client():
    if ARGS.api == 'gemini':
        from google import genai
        key = os.environ.get('GEMINI_API_KEY')
        if not key:
            raise SystemExit('Missing GEMINI_API_KEY')
        return ('gemini', genai.Client(api_key=key))
    from openai import OpenAI
    key = os.environ.get('GROQ_API_KEY')
    if not key:
        raise SystemExit('Missing GROQ_API_KEY')
    return ('groq', OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1'))


_CTX = {'client': None, 'model': None}


def get_model():
    if ARGS.model:
        return ARGS.model
    return 'gemini-2.5-flash' if ARGS.api == 'gemini' else 'llama-3.3-70b-versatile'


def retry_hint(e):
    headers = getattr(getattr(e, 'response', None), 'headers', None) or \
              getattr(getattr(e, 'http_response', None), 'headers', None)
    if headers:
        ra = headers.get('retry-after')
        if ra:
            try:
                return max(0.2, float(ra))
            except ValueError:
                pass
    return None


def call_api(prompt):
    kind, cli = _CTX['client']
    model = _CTX['model']
    try:
        if kind == 'gemini':
            return cli.models.generate_content(
                model=model, contents=prompt,
                config={'max_output_tokens': 500, 'temperature': ARGS.temp}).text
        r = cli.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=ARGS.temp, max_tokens=500)
        return r.choices[0].message.content
    except Exception as e:
        if kind == 'gemini':
            from google.genai import errors as genai_errors
            if isinstance(e, genai_errors.APIError):
                code = getattr(e, 'code', None)
                if code in (429, 500, 502, 503, 504):
                    raise RetryableAPIError(retry_hint(e)) from e
                raise FatalAPIError(f'Gemini code {code}: {e}') from e
            raise
        from openai import APIConnectionError, APIStatusError, RateLimitError
        if isinstance(e, (RateLimitError, APIConnectionError)):
            raise RetryableAPIError(retry_hint(e)) from e
        if isinstance(e, APIStatusError):
            sc = e.status_code
            if sc in (429, 500, 502, 503, 504):
                raise RetryableAPIError(retry_hint(e)) from e
            raise FatalAPIError(f'HTTP {sc}: {e}') from e
        raise


def parse_output(raw):
    if not raw:
        return None
    t = raw.strip()
    t = re.sub(r'^```(?:json)?\s*', '', t)
    t = re.sub(r'\s*```$', '', t).strip()
    if t.startswith('{'):
        try:
            d = json.loads(t)
            v = d.get('rewritten')
            if isinstance(v, str) and v.strip():
                return v.strip()
        except json.JSONDecodeError:
            pass
    if len(t) >= 2 and t[0] == t[-1] and t[0] in '\'"':
        t = t[1:-1]
    return t or None


def process_row(row):
    prompt = build_prompt(row['lang'], row['text'])
    for attempt in range(1, ARGS.max_retries + 1):
        try:
            out = parse_output(call_api(prompt))
            return row['id'], out if out else '<EMPTY>', 'ok' if out else 'empty'
        except RetryableAPIError as e:
            if attempt >= ARGS.max_retries:
                break
            wait = e.wait_hint or min(60.0, 1.5 ** attempt) + random.uniform(0, 0.5)
            time.sleep(wait)
        except FatalAPIError as e:
            print(f'  [FATAL] {row["id"]}: {e}', file=sys.stderr)
            return row['id'], None, 'fatal'
    return row['id'], None, 'gave_up'


def atomic_json(path, data):
    d = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_tsv(lang, items, out_txt):
    os.makedirs(os.path.dirname(os.path.abspath(out_txt)) or '.', exist_ok=True)
    rank = {}
    for it in items:
        rank[it['id']] = len(rank)
    with open(out_txt, 'w', encoding='utf-8', newline='') as f:
        f.write('\t'.join(HEADER) + '\n')
        for it in sorted(items, key=lambda x: rank[x['id']]):
            new_id = f"synth_implicit_{lang}_{rank[it['id']] + 1:04d}"
            row = [new_id, '', '', it['rewritten'], it['stereotype'],
                   'yes_implicit', it['target']]
            f.write('\t'.join(row) + '\n')


def main():
    paths = find_input_paths()
    if not paths:
        sys.exit('No original dataset found (StereoQueerEval_*_training.tsv)')
    rows = read_source_rows(paths)
    if not rows:
        sys.exit('No yes_explicit rows found')
    from collections import Counter
    by_lang = Counter(r['lang'] for r in rows)

    if ARGS.dry_run:
        print('  Dry-run: yes_explicit rows ready for implicit generation')
        for lg, n in sorted(by_lang.items()):
            print(f'    {lg}: {n}')
        print('  Total:', len(rows))
        return

    done_ids = load_existing(ARGS.out) if ARGS.resume else set()
    todo = [r for r in rows if r['id'] not in done_ids]
    if ARGS.limit:
        todo = todo[:ARGS.limit]
        for lg in list(by_lang):
            if lg not in {r['lang'] for r in todo}:
                del by_lang[lg]
    print(f'  Already done: {len(done_ids)} | To process: {len(todo)} | '
          f'By language: {dict(sorted(by_lang.items()))}')
    if not todo:
        sys.exit('Nothing left to process (--resume)')

    kind, cli = client()
    _CTX['client'] = (kind, cli)
    _CTX['model'] = get_model()
    print(f'  API: {ARGS.api} | model: {_CTX["model"]} | '
          f'concurrency: {ARGS.max_concurrency or (8 if kind == "gemini" else 4)}')

    results = {it['id']: it for it in load_existing(ARGS.out)
               if it.get('rewritten') and it.get('rewritten') != '<EMPTY>'} if ARGS.resume else {}
    counts = {'ok': 0, 'empty': 0, 'fatal': 0, 'gave_up': 0}
    chunk = ARGS.max_concurrency or (8 if kind == 'gemini' else 4)

    def snapshot():
        meta = {
            'generator': 'llm_implicit_augment.py',
            'api': ARGS.api,
            'model': _CTX['model'],
            'created_at': now_iso(),
            'count': len(results),
            'items': [results[k] for k in sorted(results)],
        }
        atomic_json(ARGS.out, meta)
        by_lang = {}
        for it in results.values():
            by_lang.setdefault(it['lang'], []).append(it)
        for lg, items in by_lang.items():
            write_tsv(lg, items,
                      os.path.join(ARGS.tsv_dir,
                                   f'SynthImplicit_{lg}{ARGS.txt_suffix}'))
        return len(results)

    try:
        completed = 0
        with ThreadPoolExecutor(max_workers=chunk) as ex:
            for i in range(0, len(todo), chunk):
                batch = todo[i:i + chunk]
                futs = {ex.submit(process_row, r): r for r in batch}
                for fut in as_completed(futs):
                    rid, out, status = fut.result()
                    counts[status] = counts.get(status, 0) + 1
                    if out and out != '<EMPTY>':
                        src = next((r for r in batch if r['id'] == rid), None)
                        if src:
                            results[rid] = {
                                'id': rid,
                                'lang': src['lang'],
                                'stereotype': src['stereotype'],
                                'source_hate_speech': 'yes_explicit',
                                'generated_hate_speech': 'yes_implicit',
                                'target': src['target'],
                                'original': src['text'],
                                'rewritten': out,
                                'generated_at': now_iso(),
                            }
                    completed += 1
                    if completed % 20 == 0:
                        snapshot()
                        print(f'  ...{len(results)} items | ok={counts.get("ok",0)} '
                              f'empty={counts.get("empty",0)} '
                              f'fatal={counts.get("fatal",0)} '
                              f'gave_up={counts.get("gave_up",0)}', flush=True)
    except KeyboardInterrupt:
        print('\n  Ctrl+C: saving partial results...', flush=True)
    finally:
        if results:
            n = snapshot()
            print(f'  Saved {n} items -> {ARGS.out} (+ SynthImplicit_*_training.tsv)')
        print('  Done. Summary:', counts)


if __name__ == '__main__':
    main()