# Public human-labeled evaluation data

## Purpose and boundaries

Build a reproducible public benchmark for the Choice, Boolean, and Score primitives.
Keep `evals/data/m1-suite.jsonl` and its 141 model-adjudicated decisions as synthetic
regression evidence. Public benchmark results are separate from approval of
`support-routing-v1`, `document-relevance-v1`, and `rubric-assessment-v1`.

Imported labels must retain their real provenance. Do not invent annotator IDs,
claim two independent reviewers when upstream evidence does not establish that,
or relabel upstream human annotations as local adjudication. Public benchmark
readiness does not satisfy the specification's production release gate.

## Dataset choices

| Primitive | Source | Adaptation |
| --- | --- | --- |
| Choice | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets), 13,083 examples, CC BY 4.0 | Freeze a documented set of at most 16 intents before model evaluation. Keep original intent labels; do not silently map them into the four support teams. |
| Score | [ASAP 2.0](https://github.com/scrosseye/ASAP_2.0), 24,278 essays, CC BY 4.0 | Preserve the original score scale, rubric, assignment, and source context. Use a distinct essay-scoring workload. |
| Boolean | [WikiQA](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/) candidate | Verify original distribution and usage terms before import. Use explicit answer-sentence judgments, including negatives, under a distinct answer-relevance workload. If unsuitable, document the replacement before use. |

SciFact is not the initial Boolean source: missing relevance judgments are not
negative labels. BoolQ is a possible separate yes/no comprehension benchmark, but
does not measure answer relevance and must not be presented as doing so.

## Implementation sequence

1. Pin upstream revisions/downloads and SHA-256 checksums in a source manifest.
   Record source URLs, licenses, attribution, annotation documentation, and known
   restrictions. Store downloaded data and generated bulk datasets outside Git;
   commit adapters, configuration, tests, and compact manifests/reports.
2. Add an explicit upstream-human annotation status and source metadata to the
   dataset contract. Permit it only in an explicitly selected benchmark suite.
   Keep strict independent-human requirements for release validation.
3. Add deterministic importers for the three sources. Preserve raw labels, stable
   row IDs, source groups, task/template IDs, and upstream split identity. Exclude
   label/rater metadata from model input. Reject malformed or unsupported rows.
4. Preserve upstream test boundaries. Split only eligible training data into
   development/calibration/training partitions. Keep source and duplicate groups
   together; quarantine boundary conflicts rather than leak held-out data. Record
   exclusions, label distributions, per-workload split counts, and dataset hashes.
5. Validate the resulting suite has at least 1,000 decisions across the three
   primitives. Publish sample sufficiency separately for every workload; total
   dataset size is not evidence of selective-error certification.
6. Test adapters, label mappings, provenance, deterministic splits, leakage
   rejection, and compatibility with the existing evaluation runner. Generate
   and validate the real public suite without opening locked tests for model
   selection. Provide runnable commands and record what was actually verified.

## Statistical and release requirements

Fit temperatures and thresholds only on calibration data. Evaluate a frozen
policy on independent held-out data; do not certify a threshold using the same
observations that selected it. At zero errors, the one-sided 95% exact binomial
upper bound first falls below 5% at 59 independent accepted cases. More samples
are needed with errors; repeated questions/essays from a source are not automatically
independent. Use cluster-aware analysis where appropriate.

Report coverage, accepted counts/errors, upper bounds, calibration quality, and
the comparison against raw scoring per workload. Public datasets may have appeared
in model pretraining and have different prevalence/domains than application traffic.
Keep imported profiles review-only until the applicable approval evidence exists.

Production acceptance still requires representative application data, verified
annotation quality, frozen workload-specific gates, and sufficient held-out evidence.
Do not mark M1/M2 release criteria passed merely because this import succeeds.

## Completion criteria

- Versioned source configuration and repeatable import/validation commands.
- At least 1,000 imported human-labeled decisions with all three primitives,
  or an explicit source/licensing blocker rather than fabricated substitute labels.
- Original test boundaries and source grouping preserved in evaluation.
- Tests and lint pass; generated manifest records actual counts and hashes.
- The existing synthetic regression suite and acceptance-safety behavior remain intact.

## Delivery order

Commit and push this plan to `main` first, then implement it. Stage only this
document for the planning commit so existing uncommitted M2 work is preserved.
