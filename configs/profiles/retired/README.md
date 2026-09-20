# Retired calibration profiles

These three profiles were fitted during M2 on `evals/data/m1-suite.jsonl`, a 141-decision set
whose labels were produced by `qwen/qwen3.8-27b` and then scored by the backbone. They are kept
for history and are **not** loaded by any default profile registry.

They are retired for three independent reasons, each visible in the files themselves:

1. **The labels are not ground truth.** `JEV_AT_HOME_SPEC.md` ADR-06 states that teacher-model
   answers are not evaluation ground truth. A model family grading its own output cannot measure
   its own calibration.
2. **The fitted temperatures are degenerate.** `document-relevance-v1` and `rubric-assessment-v1`
   both fitted `temperature: 0.1`, the practical floor of the search range, which drives every
   distribution to a one-hot. The recorded reliability bins show mean confidences of
   `0.9999999999999993` and `1.0`.
3. **The evidence is far below the sufficiency threshold.** `sample_count` is 6 and 4. The
   specification requires at least 59 error-free accepted observations before an exact binomial
   upper bound can reach 0.05, and `samples_sufficient_for_gate` is already `false` in each file.

`has_validated_evidence()` in `src/jah/calibration.py` rejects all three, so they could never have
enabled automatic acceptance. Retiring them removes the weaker claim that they measured anything
at all.

Profiles fitted on the public human-labeled benchmark live in `configs/profiles/public/`.
