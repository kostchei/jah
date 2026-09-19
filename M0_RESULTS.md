# M0 compatibility and data-contract result

Status: **passed** on 2026-09-19.

## Frozen inputs

- Workload: `support-routing-v1`, a four-way Choice decision covering billing, technical, account, and other.
- Operating policy: review-only for M0. Incorrect automatic routing is treated as high cost, while unnecessary review is low cost; automatic acceptance remains disabled.
- Planning load: 1 request/second with 1-32 questions per request. This is a frozen planning input, not an M0 capacity claim.
- Backbone: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` under Apache-2.0.
- Precision: BF16.
- Prompt: `decision-prompt-v1`, rendered through the model's chat template with thinking disabled.
- Labels: space-prefixed Latin uppercase labels; all four used labels are verified as distinct single tokens after the answer boundary.
- Dataset: 50 synthetic, two-pass adjudicated feasibility examples. This set is excluded from any future locked test split.

## Verified environment

- Windows 11, Python 3.12.9.
- 63.69 GiB host RAM.
- NVIDIA GeForce RTX 4090, 24,564 MiB VRAM, compute capability 8.9.
- NVIDIA driver 616.64.
- PyTorch 2.10.0+cu128; CUDA is available and a CUDA tensor operation passed.
- Transformers 5.17.0.

## Measured result

The authoritative values are in `artifacts/m0/compatibility.json`. The passing run processed all 50 decisions and achieved:

| Measure | Result |
| --- | ---: |
| Accuracy | 0.74 |
| Macro-F1 | 0.752 |
| Majority baseline accuracy | 0.26 |
| Accuracy margin over majority | +0.48 |
| Median inference | 108.9 ms/decision |
| P95 inference | 9,599.2 ms/decision |
| Peak allocated VRAM | 9.32 GB |
| Largest compiled prompt | 254 tokens |

This passes the frozen M0 feasibility gate: accuracy at least 0.50 and at least 0.05 above the majority baseline. It establishes that the pinned backbone fits on the target GPU, exposes usable next-token logits, and has measurable signal for the first workload.

## Interpretation and limitations

M0 is a feasibility result, not a release-quality claim. The 50 examples are synthetic and were author/reviewer-pass adjudicated during implementation; M1 still requires the specified independently annotated 1,000-example suite and a locked test split. The current error concentration is technical requests being routed to `other`, which is an explicit target for M1 prompt and evaluation work.

Transformers used correct reference PyTorch implementations for `causal_conv1d` and `flash-linear-attention`; optional optimized kernels were not installed. Other GPU workloads were active during the recorded run, which produced a long P95 tail. Latency here is therefore a correctness-reference measurement, not an optimized throughput claim.
