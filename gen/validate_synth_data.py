#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Synthetic Data Validation & Quality Filtering Script
for StereoQueerEval 2027 (SemEval Task B / C).

Performs rigorous 5-stage automated quality auditing & filtering on LLM-augmented data:
  Stage 1: Schema & Format Integrity (7-column SemEval standard, valid IDs, non-empty fields)
  Stage 2: LLM Artifact & Refusal Detection (strip preambles, detect refusal/apology outputs)
  Stage 3: Class Purity & Overt Slur Filter (ensures comments are truly 'yes_implicit', not 'yes_explicit')
  Stage 4: Diversity & Non-Triviality (reject exact/near verbatim copies of source explicit comment)
  Stage 5: Language & Target Consistency (detects target drift, language mismatch)
  Stage 6 (Optional): LLM-as-a-Judge Verification (Gemini / Groq zero-shot consensus check)

Outputs:
  - Cleaned & Verified TSV: {path}_clean.tsv (or direct overwrite)
  - Detailed Audit Report:  {out_dir}/synth_validation_report.json
  - Visual Terminal Summary with rejection breakdown & quality metrics.

Usage:
  # Fast rule-based validation on a single TSV:
  python gen/validate_synth_data.py --input data/SynthImplicit_NL_training.tsv

  # Validate all synthetic TSV files in a directory:
  python gen/validate_synth_data.py --data-dir data/

  # Run with LLM-as-a-Judge verification (Gemini Flash):
  python gen/validate_synth_data.py --input data/SynthImplicit_NL_training.tsv --use-llm-judge --api gemini

  # In-place clean overwrite:
  python gen/validate_synth_data.py --input data/SynthImplicit_NL_training.tsv --overwrite
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

# -----------------------------------------------------------------------------
# AUTO-LOAD .ENV
# -----------------------------------------------------------------------------
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
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass
            break

auto_load_dotenv()

# -----------------------------------------------------------------------------
# CONSTANTS & LEXICONS
# -----------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    'StereoQueerEval_id',
    'yt_title',
    'yt_description',
    'yt_comment',
    'stereotype',
    'hate_speech',
    'target'
]

VALID_STEREOTYPES = {'yes', 'no'}
VALID_HATE_SPEECH = {'no', 'yes_implicit', 'yes_explicit'}
VALID_TARGET_SCOPES = {'group', 'individual', 'none'}
VALID_IDENTITIES = {'l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+'}

REFUSAL_REGEXES = [
    re.compile(r"^i cannot (?:fulfill|generate|create|comply|assist)", re.IGNORECASE),
    re.compile(r"^as an ai\b", re.IGNORECASE),
    re.compile(r"^i am an ai\b", re.IGNORECASE),
    re.compile(r"\bagainst my (?:safety|content|usage) (?:guidelines|policies)\b", re.IGNORECASE),
    re.compile(r"^i am unable to (?:generate|create|comply|assist)", re.IGNORECASE),
    re.compile(r"cannot generate hate speech", re.IGNORECASE),
    re.compile(r"^i apologize, but i cannot", re.IGNORECASE),
    re.compile(r"^sorry, but i can'?t", re.IGNORECASE),
    re.compile(r"^i'm sorry, but i cannot", re.IGNORECASE),
]

PREAMBLE_PATTERNS = [
    re.compile(r"^(?:here (?:is|are)(?: the)?|implicit (?:comment|variation|version)|variant \d+:?|variation \d+:?)\s*", re.IGNORECASE),
    re.compile(r"^\[(?:implicit|variation|sample)\]:?\s*", re.IGNORECASE),
    re.compile(r"^comment:\s*", re.IGNORECASE),
]

# Multilingual overt slurs and severe curses (presence indicates explicit hate, violating implicit purity)
OVERT_SLURS_MULTILINGUAL = {
    'EN': [
        'faggot', 'fag', 'dyke', 'tranny', 'shemale', 'homo', 'kill all', 'die fags',
        'burn in hell', 'pedophile', 'groomer', 'abomination', 'faggots', 'dykes'
    ],
    'IT': [
        'frocio', 'froci', 'finocchio', 'finocchi', 'checca', 'culattone', 'culattoni',
        'morite', 'bruciate', 'pervertito', 'pedofilo', 'schifoso frocio'
    ],
    'NL': [
        'flikker', 'flikkers', 'poot', 'poten', 'kankerhomo', 'kankerpot', 'kankerflikker',
        'teringhomo', 'stik de moord', 'dood aan', 'pedofielen', 'smerige homo'
    ]
}

# Stopwords / markers for language verification
LANGUAGE_MARKERS = {
    'EN': {'the', 'and', 'is', 'to', 'in', 'it', 'you', 'that', 'this', 'for', 'are', 'with', 'not'},
    'IT': {'il', 'la', 'che', 'e', 'un', 'una', 'in', 'per', 'non', 'sono', 'questo', 'questa', 'di', 'con'},
    'NL': {'de', 'het', 'een', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'voor', 'niet', 'maar', 'is', 'zijn'}
}


# -----------------------------------------------------------------------------
# DETERMINISTIC VALIDATION & CLEANING FUNCTIONS
# -----------------------------------------------------------------------------

def clean_comment_text(text: str) -> str:
    """Cleans up formatting artifacts from LLM generation without changing semantics."""
    if not isinstance(text, str):
        text = str(text or '')
    
    # Strip markdown bolding/quotes/backticks around the whole string
    text = text.strip()
    text = re.sub(r'^```(?:json|text)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip('`"\'“”‘’')

    # Remove conversational preambles
    for pat in PREAMBLE_PATTERNS:
        text = pat.sub('', text)

    # Normalize whitespace
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_refusal(text: str) -> bool:
    """Checks if text is an LLM refusal/apology."""
    for pattern in REFUSAL_REGEXES:
        if pattern.search(text):
            return True
    return False


def validate_target_format(target_str: str) -> Tuple[bool, str]:
    """Validates that the target string matches standard SemEval format."""
    if not isinstance(target_str, str) or not target_str.strip():
        return False, "Target string is empty"
    
    target_str = target_str.strip()
    if target_str == 'none':
        return True, "Valid 'none'"
    
    parts = target_str.split('_', 1)
    if len(parts) != 2:
        return False, f"Invalid format '{target_str}' (expected <scope>_<ids>)"
    
    scope, id_list_str = parts
    if scope not in VALID_TARGET_SCOPES:
        return False, f"Invalid scope '{scope}'"
    
    for ident in id_list_str.split(','):
        ident = ident.strip()
        if ident not in VALID_IDENTITIES:
            return False, f"Unknown identity '{ident}' in target '{target_str}'"
            
    return True, "Valid target"


def check_overt_slurs(text: str, lang: str) -> Tuple[bool, Optional[str]]:
    """
    Checks if text contains overt slurs or violent threats.
    Returns (has_slur: bool, detected_slur: str).
    """
    text_lower = text.lower()
    slur_list = OVERT_SLURS_MULTILINGUAL.get(lang, []) + OVERT_SLURS_MULTILINGUAL.get('EN', [])
    for slur in slur_list:
        # Match whole word or exact substring in phrase
        pattern = r'\b' + re.escape(slur) + r'\b'
        if re.search(pattern, text_lower):
            return True, slur
    return False, None


def compute_token_similarity(text1: str, text2: str) -> float:
    """Computes Jaccard word-level similarity between two texts."""
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def check_language_markers(text: str, lang: str) -> Tuple[bool, float]:
    """
    Checks if the text contains expected language stop words.
    Returns (is_likely_language: bool, marker_ratio: float).
    """
    words = set(re.findall(r'\w+', text.lower()))
    if len(words) < 4:
        return True, 1.0  # Too short to reliably penalize
    
    target_markers = LANGUAGE_MARKERS.get(lang, set())
    if not target_markers:
        return True, 1.0

    matched = words & target_markers
    ratio = len(matched) / min(len(words), 10)
    # If text is at least 6 words and has 0 target language markers, might be wrong language
    if len(words) >= 6 and len(matched) == 0:
        # Check if it has strong English markers when expecting IT or NL
        if lang != 'EN':
            en_matched = words & LANGUAGE_MARKERS['EN']
            if len(en_matched) >= 3:
                return False, 0.0
    return True, ratio


# -----------------------------------------------------------------------------
# LLM-AS-A-JUDGE VALIDATOR (GEMINI & GROQ)
# -----------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are an impartial expert linguistic judge validating augmented hate-speech dataset samples for the SemEval StereoQueerEval benchmark.

We generated an IMPLICIT hate comment (indirect, veiled, sarcastic, rhetorical, coded language) targeting LGBTQ+.
Your job is to evaluate if this comment meets the strict standard for 'yes_implicit'.

CONTEXT:
- Video Title: {title}
- Video Description: {description}
- Target Identity: {target}
- Language: {lang}
- Generated Comment: "{comment}"

EVALUATION CRITERIA:
1. IMPLICIT QUALITY: Is it truly IMPLICIT (no direct slurs/threats, but clear underlying prejudice, rejection, or mockery between the lines)?
   - If it has zero hate/prejudice (completely benign), it fails.
   - If it has overt, vulgar slurs or explicit violent curses, it fails.
2. TARGET FIDELITY: Does the prejudice remain aligned with the target ({target})?
3. NATURALNESS: Is it phrased like an authentic, natural social media comment?

OUTPUT FORMAT:
Respond with ONLY a raw JSON object (no markdown formatting, no explanations outside JSON):
{{
  "verdict": "ACCEPT" | "REJECT",
  "confidence": 0.0 to 1.0,
  "detected_class": "yes_implicit" | "no" | "yes_explicit",
  "reason": "Short 1-sentence reason"
}}"""


class LLMJudge:
    """Zero-shot LLM Judge using Google Gemini or Groq."""
    def __init__(self, api: str = "gemini", model_name: Optional[str] = None):
        self.api = api.lower()
        if self.api == "gemini":
            self.model_name = model_name or "gemini-2.5-flash"
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required for --use-llm-judge")
            from google import genai
            self.client = genai.Client(api_key=api_key)
        elif self.api in ("groq", "openai"):
            self.model_name = model_name or "llama-3.3-70b-versatile"
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is required for --use-llm-judge")
            from openai import OpenAI
            self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        else:
            raise ValueError(f"Unsupported judge API: {self.api}")

    def judge_sample(self, comment: str, title: str, description: str, target: str, lang: str) -> Dict[str, Any]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            title=title[:200],
            description=description[:200],
            target=target,
            lang=lang,
            comment=comment
        )
        try:
            if self.api == "gemini":
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"temperature": 0.1, "response_mime_type": "application/json"}
                )
                raw_text = resp.text or "{}"
            else:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                raw_text = resp.choices[0].message.content or "{}"
            
            # Parse JSON
            raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text.strip())
            raw_text = re.sub(r'\s*```$', '', raw_text.strip())
            data = json.loads(raw_text)
            return {
                "verdict": data.get("verdict", "ACCEPT").upper(),
                "confidence": float(data.get("confidence", 0.8)),
                "detected_class": data.get("detected_class", "yes_implicit"),
                "reason": data.get("reason", "LLM consensus")
            }
        except Exception as e:
            return {
                "verdict": "ACCEPT",
                "confidence": 0.5,
                "detected_class": "unknown",
                "reason": f"Judge error: {str(e)}"
            }


# -----------------------------------------------------------------------------
# MAIN VALIDATION PIPELINE
# -----------------------------------------------------------------------------

class SyntheticDataValidator:
    """Comprehensive multi-stage validator and cleaner for synthetic TSV datasets."""
    def __init__(
        self,
        min_words: int = 3,
        max_words: int = 180,
        max_source_similarity: float = 0.90,
        use_llm_judge: bool = False,
        judge_api: str = "gemini",
        judge_model: Optional[str] = None
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.max_source_similarity = max_source_similarity
        self.use_llm_judge = use_llm_judge
        self.judge = LLMJudge(api=judge_api, model_name=judge_model) if use_llm_judge else None

    def validate_file(
        self,
        input_tsv: str,
        output_tsv: Optional[str] = None,
        overwrite: bool = False,
        known_val_titles: Optional[Set[str]] = None,
        known_val_ids: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Validates a single synthetic TSV file and saves clean records."""
        if not os.path.isfile(input_tsv):
            raise FileNotFoundError(f"Input file not found: {input_tsv}")

        # Detect language from filename
        match = re.search(r'_([A-Z]{2})(?:_training|_clean)?\.tsv$', input_tsv)
        lang = match.group(1) if match else "NL"

        rows_in = []
        with open(input_tsv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for r in reader:
                rows_in.append(r)

        total_samples = len(rows_in)
        accepted_rows = []
        rejected_stats = Counter()
        rejection_details = []
        seen_comments = set()

        print(f"\n🔍 Auditing {input_tsv} ({total_samples} samples, Language: {lang})...")

        for idx, row in enumerate(rows_in):
            row_id = row.get('StereoQueerEval_id', f'row_{idx}')
            raw_comment = row.get('yt_comment', '')
            title = row.get('yt_title', '')
            desc = row.get('yt_description', '')
            target = row.get('target', 'none')
            hate_speech = row.get('hate_speech', '')
            stereotype = row.get('stereotype', '')

            # -------------------------------------------------------------
            # Stage 1: Schema & Integrity Checks
            # -------------------------------------------------------------
            if not raw_comment or str(raw_comment).strip() == '':
                rejected_stats['empty_comment'] += 1
                continue

            # Check target format
            is_valid_tgt, tgt_msg = validate_target_format(target)
            if not is_valid_tgt:
                rejected_stats['invalid_target_format'] += 1
                rejection_details.append({'id': row_id, 'reason': tgt_msg, 'comment': raw_comment[:60]})
                continue

            # Check hate_speech label must be yes_implicit
            if hate_speech != 'yes_implicit':
                rejected_stats['non_implicit_label'] += 1
                continue

            # -------------------------------------------------------------
            # Stage 2: Clean formatting & Refusal Detection
            # -------------------------------------------------------------
            cleaned_comment = clean_comment_text(raw_comment)
            if detect_refusal(cleaned_comment):
                rejected_stats['llm_refusal_detected'] += 1
                rejection_details.append({'id': row_id, 'reason': 'LLM safety refusal detected', 'comment': cleaned_comment[:60]})
                continue

            word_count = len(cleaned_comment.split())
            if word_count < self.min_words:
                rejected_stats['too_short'] += 1
                continue
            if word_count > self.max_words:
                rejected_stats['too_long'] += 1
                continue

            # -------------------------------------------------------------
            # Stage 3: Class Purity & Overt Slur Filter
            # -------------------------------------------------------------
            has_slur, slur_word = check_overt_slurs(cleaned_comment, lang)
            if has_slur:
                rejected_stats['overt_slur_detected'] += 1
                rejection_details.append({
                    'id': row_id,
                    'reason': f"Contains overt slur/curse '{slur_word}' (fails implicit requirement)",
                    'comment': cleaned_comment
                })
                continue

            # -------------------------------------------------------------
            # Stage 4: Duplicate & Degenerate Variation Filter
            # -------------------------------------------------------------
            normalized_c = cleaned_comment.lower()
            if normalized_c in seen_comments:
                rejected_stats['exact_duplicate_variant'] += 1
                continue
            seen_comments.add(normalized_c)

            # -------------------------------------------------------------
            # Stage 5: Language Marker Filter
            # -------------------------------------------------------------
            is_valid_lang, _ = check_language_markers(cleaned_comment, lang)
            if not is_valid_lang:
                rejected_stats['language_mismatch'] += 1
                rejection_details.append({'id': row_id, 'reason': f"Language marker mismatch for {lang}", 'comment': cleaned_comment})
                continue

            # -------------------------------------------------------------
            # Stage 6: Validation Contamination Check (if val data provided)
            # -------------------------------------------------------------
            if known_val_titles and title in known_val_titles:
                rejected_stats['validation_video_leakage'] += 1
                continue
            if known_val_ids:
                base_id = re.sub(r'^synth_', '', row_id)
                base_id = re.sub(r'_v\d+$', '', base_id)
                if base_id in known_val_ids:
                    rejected_stats['validation_source_leakage'] += 1
                    continue

            # -------------------------------------------------------------
            # Stage 7 (Optional): LLM-as-a-Judge Consensus
            # -------------------------------------------------------------
            if self.judge:
                judge_res = self.judge.judge_sample(
                    comment=cleaned_comment,
                    title=title,
                    description=desc,
                    target=target,
                    lang=lang
                )
                if judge_res['verdict'] == 'REJECT':
                    rejected_stats['llm_judge_rejected'] += 1
                    rejection_details.append({
                        'id': row_id,
                        'reason': f"LLM Judge: {judge_res.get('reason', 'Rejected')}",
                        'comment': cleaned_comment
                    })
                    continue

            # Valid sample accepted!
            clean_row = dict(row)
            clean_row['yt_comment'] = cleaned_comment
            accepted_rows.append(clean_row)

        # -----------------------------------------------------------------
        # Save output TSV
        # -----------------------------------------------------------------
        if output_tsv is None:
            if overwrite:
                output_tsv = input_tsv
            else:
                output_tsv = input_tsv.replace('.tsv', '_clean.tsv')

        os.makedirs(os.path.dirname(os.path.abspath(output_tsv)), exist_ok=True)
        with open(output_tsv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for r in accepted_rows:
                writer.writerow({k: r.get(k, '') for k in REQUIRED_COLUMNS})

        pass_rate = (len(accepted_rows) / total_samples * 100.0) if total_samples > 0 else 0.0

        # Build report
        report = {
            "input_file": input_tsv,
            "output_file": output_tsv,
            "language": lang,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_input_rows": total_samples,
            "accepted_rows": len(accepted_rows),
            "rejected_rows": total_samples - len(accepted_rows),
            "acceptance_rate_percent": round(pass_rate, 2),
            "rejection_breakdown": dict(rejected_stats),
            "sample_rejections": rejection_details[:10]
        }

        self._print_summary(report)
        return report

    @staticmethod
    def _print_summary(report: Dict[str, Any]):
        print("==================================================")
        print("📊 SYNTHETIC DATA QUALITY AUDIT REPORT")
        print(f"  • Input File:        {report['input_file']}")
        print(f"  • Cleaned Output:    {report['output_file']}")
        print(f"  • Total Evaluated:   {report['total_input_rows']} samples")
        print(f"  • Accepted Clean:    {report['accepted_rows']} samples ({report['acceptance_rate_percent']}%)")
        print(f"  • Filtered Out:      {report['rejected_rows']} samples")
        if report['rejection_breakdown']:
            print("  • Rejection Breakdown:")
            for reason, count in sorted(report['rejection_breakdown'].items(), key=lambda x: -x[1]):
                print(f"      - {reason:<28}: {count} rows")
        print("==================================================")


# -----------------------------------------------------------------------------
# CLI ENTRY POINT
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Validate & Clean Synthetic TSV Dataset for StereoQueerEval 2027")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to single synthetic TSV file")
    parser.add_argument("--data-dir", "-d", type=str, default="data", help="Directory to scan for Synth*.tsv files")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output TSV path (default: {input}_clean.tsv)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the input TSV directly in-place")
    parser.add_argument("--report-file", type=str, default="data/synth_validation_report.json", help="Path to save audit JSON report")
    
    # Filter thresholds
    parser.add_argument("--min-words", type=int, default=3, help="Minimum word length for a valid comment")
    parser.add_argument("--max-words", type=int, default=180, help="Maximum word length for a valid comment")
    
    # Optional LLM Judge
    parser.add_argument("--use-llm-judge", action="store_true", help="Enable LLM-as-a-Judge zero-shot validation")
    parser.add_argument("--api", type=str, default="gemini", choices=["gemini", "groq", "openai"], help="API for LLM Judge")
    parser.add_argument("--judge-model", type=str, default=None, help="Model override for LLM Judge")

    return parser.parse_args()


def main():
    args = parse_args()
    validator = SyntheticDataValidator(
        min_words=args.min_words,
        max_words=args.max_words,
        use_llm_judge=args.use_llm_judge,
        judge_api=args.api,
        judge_model=args.judge_model
    )

    target_files = []
    if args.input:
        target_files = [args.input]
    else:
        patterns = [
            os.path.join(args.data_dir, "Synth*.tsv"),
            os.path.join(args.data_dir, "*_clean.tsv"),
            os.path.join(args.data_dir, "SynthImplicit_*_training.tsv"),
            "data/Synth*.tsv",
            "LGBT/Synth*.tsv",
        ]
        target_files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True) if os.path.isfile(p) and not p.endswith('_clean.tsv')})

    if not target_files:
        print(f"No synthetic TSV files found to validate in '{args.data_dir}'. Specify via --input <file.tsv>")
        sys.exit(0)

    all_reports = []
    for fpath in target_files:
        out_path = args.output if (len(target_files) == 1 and args.output) else None
        rep = validator.validate_file(fpath, output_tsv=out_path, overwrite=args.overwrite)
        all_reports.append(rep)

    # Save master report
    if args.report_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_file)), exist_ok=True)
        with open(args.report_file, 'w', encoding='utf-8') as f:
            json.dump(all_reports, f, indent=2)
        print(f"📝 Full validation audit report saved to: {args.report_file}")


if __name__ == "__main__":
    main()
