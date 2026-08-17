# Ground truth format

One JSON file per resume, named to match the PDF's filename stem exactly:

```
eval/resumes/john_doe.pdf   ->  eval/ground_truth/john_doe.json
```

Schema (matches the canonical fields the pipeline attempts to extract --
see `EXAMPLE.template.json`):

```json
{
  "fullName": "Jane Doe",
  "email": "jane.doe@example.com",
  "phone": "+1 415-555-0132",
  "skills": ["Python", "React", "SQL"],
  "yearsOfExperience": 3.5,
  "education": "B.Tech Computer Science"
}
```

Rules:
- If a field genuinely isn't present anywhere in the resume, set it to `null`
  (not `""` or `[]`) -- the harness treats "extractor also returned nothing"
  as a correct match for that field, and treats the extractor inventing a
  value where none exists as a miss (hallucination).
- `skills` should be a JSON array of strings, not a comma-separated string.
- `yearsOfExperience` should be a number (the harness accepts fuzzy years,
  ±0.5 by default, to absorb "3" vs "3 years" vs "~3.5" differences).
- Don't worry about matching phone formatting to the resume's exact style --
  the harness normalizes to the last 10 digits before comparing.

Label these by hand, by reading the actual resume PDF. Don't use an LLM to
generate them -- that defeats the point of having independent ground truth.
