#!/usr/bin/env python3
"""
Evaluation harness for the resume-extraction pipeline
(app/api/routes/candidates.py: process_and_match).

Runs the REAL extraction path -- PDF text extraction, then Ollama
email/detail extraction, then the app's own regex/synthetic fallbacks --
against every PDF in eval/resumes/, and diffs the output against
hand-labeled JSON in eval/ground_truth/.

Ground truth is never generated here. See eval/ground_truth/README.md
for the schema.

Usage (from rt_backend/):
    python eval/run_eval.py
    python eval/run_eval.py --verbose
    python eval/run_eval.py --skills-threshold 0.6 --years-tolerance 1.0
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# So `app.*` imports resolve when this is run as `python eval/run_eval.py` from rt_backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.routes.candidates import extract_text_from_pdf, clean_text
from app.services.ollama_service import OllamaClient

EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# extract_candidate_details()'s prompt never pins down JSON key names -- the LLM
# picks its own. `fullName` is the only key the app itself reads (candidates.py:622).
# These aliases cover the likely variants so eval doesn't fail on key-naming drift
# rather than actual extraction quality.
FIELD_ALIASES = {
    "phone": ["phone", "phoneNumber", "phone_number", "contactNumber", "Phone Number", "Phone"],
    "skills": ["skills", "Skills"],
    "yearsOfExperience": [
        "yearsOfExperience", "years_of_experience", "yearsExperience",
        "experience", "Years of experience",
    ],
    "education": [
        "education", "Education", "highestDegree", "highestEducation",
        "Education (highest degree)",
    ],
}

CANONICAL_FIELDS = ["fullName", "email", "phone", "skills", "yearsOfExperience", "education"]


def _unwrap(v: Any) -> Any:
    """extract_candidate_details() has no fixed JSON schema, so the model
    sometimes nests a field one level deeper than asked
    (e.g. {"education": {"highestDegree": "B.Tech"}}). Pull the first
    non-empty scalar out in that case."""
    if isinstance(v, dict):
        for inner in v.values():
            if inner not in (None, "", []):
                return inner
        return None
    return v


def resolve_field(details: dict, canonical: str) -> Optional[Any]:
    for key in FIELD_ALIASES[canonical]:
        if key in details:
            val = _unwrap(details[key])
            if val not in (None, "", []):
                return val
    return None


def run_pipeline(pdf_path: Path, client: OllamaClient) -> Dict[str, Any]:
    """Mirrors the extraction path in process_and_match()."""
    text = clean_text(extract_text_from_pdf(str(pdf_path)))
    stem = pdf_path.stem

    # email: LLM -> regex fallback -> synthetic fallback
    email = client.extract_email_from_resume(text)
    email_source = "llm"
    if not email:
        matches = re.findall(EMAIL_REGEX, text)
        if matches:
            email, email_source = matches[0], "regex_fallback"
        else:
            email, email_source = f"{stem.lower()}@example.com", "synthetic_fallback"

    # name: LLM fullName -> filename default (no regex fallback exists for name)
    details = client.extract_candidate_details(text) or {}
    if details.get("fullName"):
        name, name_source = details["fullName"], "llm"
    else:
        name, name_source = stem, "filename_default"

    return {
        "fullName": name,
        "email": email,
        "phone": resolve_field(details, "phone"),
        "skills": resolve_field(details, "skills"),
        "yearsOfExperience": resolve_field(details, "yearsOfExperience"),
        "education": resolve_field(details, "education"),
        "_sources": {"fullName": name_source, "email": email_source},
    }


# ---------- normalization ----------

def _norm_text(v: Optional[Any]) -> Optional[str]:
    if v is None:
        return None
    s = re.sub(r"\s+", " ", str(v)).strip().lower()
    return s or None


def _norm_phone(v: Optional[Any]) -> Optional[str]:
    if v is None:
        return None
    digits = re.sub(r"\D", "", str(v))
    return digits[-10:] if len(digits) >= 10 else (digits or None)


def _norm_years(v: Optional[Any]) -> Optional[float]:
    if v is None:
        return None
    m = re.search(r"[\d.]+", str(v))
    return float(m.group()) if m else None


def _norm_skills(v: Optional[Any]) -> set:
    if not v:
        return set()
    parts = re.split(r"[,;/]", v) if isinstance(v, str) else list(v)
    return {p.strip().lower() for p in parts if p and str(p).strip()}


# ---------- comparators ----------

def compare_text(pred, truth) -> bool:
    return _norm_text(pred) == _norm_text(truth)


def compare_phone(pred, truth) -> bool:
    return _norm_phone(pred) == _norm_phone(truth)


def compare_years(pred, truth, tol: float) -> bool:
    p, t = _norm_years(pred), _norm_years(truth)
    if p is None or t is None:
        return p == t
    return abs(p - t) <= tol


def compare_skills(pred, truth, threshold: float) -> Tuple[bool, float]:
    p, t = _norm_skills(pred), _norm_skills(truth)
    if not p and not t:
        return True, 1.0
    if not p or not t:
        return False, 0.0
    jaccard = len(p & t) / len(p | t)
    return jaccard >= threshold, jaccard


FIELD_COMPARATORS = {
    "fullName": lambda p, t, args: compare_text(p, t),
    "email": lambda p, t, args: compare_text(p, t),
    "phone": lambda p, t, args: compare_phone(p, t),
    "education": lambda p, t, args: compare_text(p, t),
    "yearsOfExperience": lambda p, t, args: compare_years(p, t, args.years_tolerance),
    "skills": lambda p, t, args: compare_skills(p, t, args.skills_threshold)[0],
}


def main():
    parser = argparse.ArgumentParser(description="Resume extraction accuracy eval")
    parser.add_argument("--resumes-dir", default=str(Path(__file__).parent / "resumes"))
    parser.add_argument("--ground-truth-dir", default=str(Path(__file__).parent / "ground_truth"))
    parser.add_argument("--skills-threshold", type=float, default=0.7,
                         help="Jaccard similarity required to count skills as a match")
    parser.add_argument("--years-tolerance", type=float, default=0.5,
                         help="Allowed +/- difference (years) for yearsOfExperience")
    parser.add_argument("--verbose", action="store_true", help="Print per-resume, per-field detail")
    args = parser.parse_args()

    resumes_dir = Path(args.resumes_dir)
    gt_dir = Path(args.ground_truth_dir)

    resumes = sorted(p for p in resumes_dir.glob("*.pdf"))
    if not resumes:
        print(f"No PDFs found in {resumes_dir}. Add resumes there and matching ground-truth JSON in {gt_dir}.")
        return

    client = OllamaClient()

    field_correct = {f: 0 for f in CANONICAL_FIELDS}
    field_total = {f: 0 for f in CANONICAL_FIELDS}
    email_fallback_counts = {"llm": 0, "regex_fallback": 0, "synthetic_fallback": 0}
    email_fallback_correct = {"llm": 0, "regex_fallback": 0, "synthetic_fallback": 0}
    name_default_count = 0
    name_default_correct = 0
    skipped = []
    evaluated = 0

    for pdf_path in resumes:
        gt_path = gt_dir / f"{pdf_path.stem}.json"
        if not gt_path.exists():
            skipped.append(f"{pdf_path.name} (no ground truth)")
            continue

        ground_truth = json.loads(gt_path.read_text())

        try:
            pred = run_pipeline(pdf_path, client)
        except Exception as e:
            print(f"[ERROR] Pipeline failed on {pdf_path.name}: {e}")
            skipped.append(f"{pdf_path.name} (pipeline error)")
            continue

        evaluated += 1
        sources = pred["_sources"]

        if args.verbose:
            print(f"\n=== {pdf_path.name} ===")

        field_results = {}
        for field in CANONICAL_FIELDS:
            truth_val = ground_truth.get(field)
            pred_val = pred[field]
            correct = FIELD_COMPARATORS[field](pred_val, truth_val, args)
            field_results[field] = correct
            field_total[field] += 1
            if correct:
                field_correct[field] += 1
            if args.verbose:
                mark = "OK" if correct else "MISS"
                print(f"  [{mark:4}] {field:18} pred={pred_val!r:40} truth={truth_val!r}")

        email_fallback_counts[sources["email"]] += 1
        if field_results["email"]:
            email_fallback_correct[sources["email"]] += 1

        if sources["fullName"] == "filename_default":
            name_default_count += 1
            if field_results["fullName"]:
                name_default_correct += 1

    if skipped:
        print(f"Skipped: {', '.join(skipped)}")

    if evaluated == 0:
        print("Nothing evaluated.")
        return

    print(f"\nEvaluated {evaluated} resume(s)\n")

    print(f"{'Field':<20}{'Correct':<10}{'Total':<10}{'Accuracy':<10}")
    print("-" * 50)
    total_correct = 0
    total_fields = 0
    for field in CANONICAL_FIELDS:
        c, t = field_correct[field], field_total[field]
        total_correct += c
        total_fields += t
        acc = f"{c / t * 100:.1f}%" if t else "n/a"
        print(f"{field:<20}{c:<10}{t:<10}{acc:<10}")
    print("-" * 50)
    overall = f"{total_correct / total_fields * 100:.1f}%" if total_fields else "n/a"
    print(f"{'OVERALL':<20}{total_correct:<10}{total_fields:<10}{overall:<10}")

    print("\nEmail fallback breakdown:")
    print(f"{'Source':<20}{'Fired':<10}{'Correct':<10}{'Recovery rate':<15}")
    print("-" * 55)
    for source in ["llm", "regex_fallback", "synthetic_fallback"]:
        fired = email_fallback_counts[source]
        correct = email_fallback_correct[source]
        rate = f"{correct / fired * 100:.1f}%" if fired else "n/a"
        print(f"{source:<20}{fired:<10}{correct:<10}{rate:<15}")

    if name_default_count:
        rate = f"{name_default_correct / name_default_count * 100:.1f}%"
        print(
            f"\nName fell back to the filename default in {name_default_count}/{evaluated} resume(s) "
            f"(matched ground truth {rate} of the time when it did -- i.e. filename happened to equal the name)"
        )


if __name__ == "__main__":
    main()
