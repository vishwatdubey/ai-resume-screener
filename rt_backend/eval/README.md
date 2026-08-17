# Resume extraction eval harness

Measures field-level accuracy of the real extraction pipeline
(`app/api/routes/candidates.py::process_and_match` + `OllamaClient`)
against hand-labeled ground truth.

## Setup

1. Drop test resume PDFs into `eval/resumes/`.
2. For each one, hand-write `eval/ground_truth/<same-stem>.json` -- see
   `eval/ground_truth/README.md` for the schema. **Not generated for you;
   label these yourself from the actual PDF.**
3. Make sure Ollama is running (`ollama serve`) and the model in
   `OLLAMA_MODEL` / `.env` is pulled.
4. This repo's `requirements.txt` is currently missing two packages the
   pipeline itself imports (`pdfplumber`, `requests`) -- install them if
   you haven't already: `pip install pdfplumber requests`.

## Run

```bash
cd rt_backend
python eval/run_eval.py               # summary table
python eval/run_eval.py --verbose     # + per-resume, per-field detail
```

Tunables:
- `--skills-threshold` (default 0.7) -- Jaccard similarity to count a skills-list match
- `--years-tolerance` (default 0.5) -- allowed years-of-experience drift

## What it reports

- Per-field accuracy (fullName, email, phone, skills, yearsOfExperience, education) and overall
- Email fallback breakdown: how often the LLM path / regex fallback / synthetic
  fallback fired, and the accuracy conditional on each path firing
- Whether the name field fell back to the bare filename, and how often that
  coincidentally matched ground truth

Comparisons are normalized (case/whitespace for text, digit-suffix for phone,
±tolerance for years, Jaccard set overlap for skills) so formatting
differences don't count as extraction misses.

## How many resumes before the number means anything

For a single binary field's accuracy, the 95% confidence interval width is
roughly `±1.96 * sqrt(p(1-p)/n)`. At `n=20` and `p≈0.8`, that's about
**±17 points** -- "80% accurate" could really be anywhere from ~63% to ~97%.
That's not a number worth quoting in an interview.

- **n≈20**: fine as a smoke test / sanity check while iterating on the pipeline, not for reporting.
- **n≈30-40**: minimum floor for a directionally meaningful number (~±13-15pt CI).
- **n≈50+**: what you'd want before stating a specific percentage you're prepared to defend.

Also stratify, not just scale up: PDF text extraction quality varies a lot
by resume template (single-column vs. two-column vs. tables/graphics), so
50 resumes that are all the same template style will overstate how well
this generalizes. Spread your sample across a few visibly different resume
formats.
