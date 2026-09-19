# M0 compatibility and data-contract result

Status: **passed** on 2026-09-20 after remediation.

## Frozen inputs

- Workload: `support-routing-v1`, a four-way Choice decision covering billing, technical, account, and other.
- Operating policy: review-only for M0. Incorrect automatic routing is treated as high cost, while unnecessary review is low cost; automatic acceptance remains disabled.
- Planning load: 1 request/second with 1-32 questions per request. This is a frozen planning input, not an M0 capacity claim.
- Backbone: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` under Apache-2.0.
- Precision: BF16.
- Prompt: `decision-prompt-v2`, rendered through the model's chat template with thinking disabled.
- Labels: bare Latin uppercase labels (`A`..`P`); all four used labels are verified as distinct single tokens after the answer boundary.
- Dataset: 50 synthetic feasibility examples with single-author two-pass review. This set is excluded from any future locked test split.

## Verified environment

- Windows 11, Python 3.12.9.
- 63.69 GiB host RAM.
- NVIDIA GeForce RTX 4090, 24,564 MiB VRAM, compute capability 8.9.
- NVIDIA driver 616.64.
- PyTorch 2.10.0+cu128; CUDA is available and a CUDA tensor operation passed.
- Transformers 5.17.0.
- The run was on native Windows 11, not the specification's Linux/WSL2 baseline. A WSL2 re-run remains part of M1 reproducibility work.

## Measured result

The authoritative values are in `artifacts/m0/compatibility.json`. The passing run was generated from source commit `b79fc7df9943bafce5a0ca558ed0278113b0c688`, processed all 50 decisions, and achieved:

| Measure | Result |
| --- | ---: |
| Accuracy | 0.76 |
| Macro-F1 | 0.773 |
| Majority baseline accuracy | 0.26 |
| Accuracy margin over majority | +0.50 |
| Minimum label mass | 0.640 |
| Median label mass | 0.941 |
| Cold-start forward pass | 658.3 ms |
| Measured warmup passes | 5 |
| Median inference | 154.7 ms/decision |
| P95 inference | 177.4 ms/decision |
| Peak allocated VRAM | 9.32 GB |
| Largest compiled prompt | 254 tokens |

This passes the frozen M0 feasibility gate: accuracy at least 0.50, at least 0.05 above the majority baseline, and label mass at least 0.50 on every case. It establishes that the pinned backbone fits on the target GPU, exposes usable next-token logits, and has measurable signal for the first workload.

Evidence hashes:

- Compatibility report: `c1fb89de976ee84b190f2b9677a9c102c1c01750a005008786d081de8d3e6039`
- Predictions: `18d38e1047432b3040730352fece170e1714202c9e86315ac57bc552fe6e2d33`
- Order-bias report: `499f702dc8cd0c9127b69b981b595b8f4f9f2b3a8655806441b0037467d8958a`

## Order-bias diagnostic

The diagnostic in `artifacts/m0/order_bias.json` used the same source commit and the single-item reference scoring path. Identical prompts were measured once and reused so the comparison could not be contaminated by repeated-run or batching variance.

| Run | Accuracy | Macro-F1 | `other` share | Flip rate vs. R0 |
| --- | ---: | ---: | ---: | ---: |
| R0 baseline | 0.760 | 0.773 | 0.48 | 0.00 |
| R1 all cyclic rotations | 0.755 | 0.768 | 0.47 | 0.06 |
| R2 `other` in position A | 0.760 | 0.775 | 0.44 | 0.04 |
| R3 shortened `other` wording | 0.820 | 0.830 | 0.38 | 0.10 |
| R4 content-free, all rotations | 0.240 | 0.097 | 1.00 | 0.52 |

R1 does not show a last-position effect: `other` was selected on 44% of cases in position A and 48% in positions B, C, and D. R3 reduced `other` selection by 10 percentage points and improved macro-F1 by 0.056. The recorded branch is therefore **wording bias**, and the M1 prompt action is to rewrite the catch-all description and tune it on the development split only. R4 selected the semantic `other` option in every rotation while distributing the winning letter evenly, reinforcing that conclusion.

## Compute budgets and stop conditions

- M1: at most 20 full development runs, 10 prompt versions, and 2 cumulative RTX 4090 GPU-hours. Stop prompt iteration after three consecutive versions improve macro-F1 by less than 0.01.
- Open M4 if the best frozen development macro-F1 remains below 0.85. If it remains below 0.75, review the model, labels, and workload before spending the adapter budget.
- M4: at most 12 train/evaluate trials, 24 cumulative RTX 4090 GPU-hours, and 2 hours per trial. Stop after three consecutive trials fail to improve macro-F1 by at least 0.01 over the frozen baseline.
- Stop the project if the best frozen baseline plus adapter remains below 0.85 development macro-F1.

## Interpretation and limitations

M0 is a feasibility result, not a release-quality claim. The 50 examples are synthetic and received two passes from one author; they are not independently adjudicated. An independent human pass is still required before M1 labeling starts, followed by the specified independently annotated 1,000-example suite and locked test split. The current error concentration remains technical requests being routed to `other`, which is an explicit target for M1 prompt and evaluation work.

Transformers used correct reference PyTorch implementations for `causal_conv1d` and `flash-linear-attention`; optional optimized kernels were not installed. The compatibility report records `nvidia-smi` process snapshots before and after measurement, and other GPU-capable processes were present. Latency remains a correctness-reference measurement, not an optimized throughput claim.
