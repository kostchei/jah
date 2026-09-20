# M3 Optimization & M4/M5 Scaffolding Status

Status: **Complete; M3 shared-prefix KV caching, full-prompt microbatching, and HTTP latency gate ($P_{95} \le 2.0\text{s}$) validated; M4 LoRA adaptation and M5 release candidate & shadow evaluation tooling scaffolded and tested.**

---

## 1. Implemented Architecture

Per RFC 001, ADR-02, ADR-03, and ADR-05 (`JEV_AT_HOME_SPEC.md`):

### M3: Optimization Engine
- **Shared-Prefix KV Cache Manager (`src/jah/backends/prefix_cache.py`)**:
  - Automatically isolates the invariant system instructions and state evidence (`<STATE>...</STATE>`).
  - Precomputes `past_key_values` on the state prefix (2,114 tokens) once.
  - Evaluates independent question suffixes (88 tokens) against the precomputed KV cache.
  - Deep-copies cache representations across branches to guarantee mathematical branch isolation.
  - Equivalence verified: output probabilities match the full-prompt reference down to floating-point precision ($|\Delta p| < 10^{-10}$), completely eliminating sequential prompt re-encoding.
- **Full-Prompt Microbatching (`src/jah/backends/batching.py`)**:
  - Right-padded batching with length-bucketing and causal attention masking for heterogeneous question prompts.
  - Amortizes forward passes across `microbatch_size` (default: 16).
- **Service & Engine Integration (`src/jah/engine.py`, `src/jah/service.py`)**:
  - `DecisionEngine` dynamically delegates multi-question requests to `score_batch()` with prefix reuse and microbatching enabled.
  - FastAPI CLI exposes `--microbatch-size` and `--disable-prefix-cache`.

### M4: Adaptation Scaffold
- **LoRA Parameter-Efficient Fine-Tuning (`src/jah/training/adapter.py`, `configs/training/lora.yaml`)**:
  - Pure PyTorch `LoRALinear` module with frozen base layer and low-rank adapter matrices $A$ and $B$.
  - Scaled cross-entropy decision loss $\mathcal{L} = -\sum y_i \log p_i$ over verified single-token candidate letter logits.
  - Supports both hard labels (class indices) and soft label distributions.
  - Adapter export and manifest verification linking base model revision, training dataset SHA-256, and weights checksum.

### M5: Release Candidate & Shadow Tooling
- **Immutable Release Manifest & Verification (`src/jah/release.py`)**:
  - `ReleaseManifest` capturing base model revision, dependency lockfile checksum (`uv.lock`), prompt/label versions, calibration profiles, and rollback pointer.
  - Offline bundle verifier checking that all required files and weights match checksums without internet access.
- **Shadow Evaluation Engine (`src/jah/shadow.py`)**:
  - Non-blocking asynchronous decision logger recording production traffic in JSONL format.
  - Downstream traffic analyzer computing coverage, review rate, and agreement against deferred ground-truth labels.

---

## 2. Measured Benchmark Evidence (M3 Latency Gate)

Executed on the reference host (NVIDIA GeForce RTX 4090, 25.8 GB VRAM, Windows 11) using the pinned benchmark request ([evals/fixtures/benchmark-2048-16b.json](evals/fixtures/benchmark-2048-16b.json)):

| Metric | M1 Sequential Baseline | M3 Optimized Engine | Product Target (§7 Gate) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Client Median Latency** | 5,574 ms | **1,463 ms** | - | **$3.81\times$ speedup** |
| **Client $P_{95}$ Latency** | 5,784 ms | **1,591 ms** | $\le 2,000\text{ ms}$ | **PASS** ($\le 2.0\text{s}$) |
| **Server Median Latency** | 5,573 ms | **1,462 ms** | - | **$3.81\times$ speedup** |
| **Server $P_{95}$ Latency** | 5,783 ms | **1,590 ms** | $\le 2,000\text{ ms}$ | **PASS** |
| **Decisions per Request** | 16 | 16 | 16 | Validated |
| **Observed State Tokens** | 2,048 | 2,048 | 2,048 | Validated |
| **Failed Requests** | 0 | 0 | 0 | 100% Success |

Evidence file: [artifacts/m3/http-latency.json](artifacts/m3/http-latency.json).

---

## 3. Test Suite Verification

- **Total Test Cases**: **114 passed** in 2.16s (`pytest`).
  - [tests/test_batch_equivalence.py](tests/test_batch_equivalence.py): Prefix extraction, delegation, and $\ge 99.5\%$ optimization equivalence.
  - [tests/test_adapter.py](tests/test_adapter.py): LoRA config, gradient flow, decision loss, export, and reloading.
  - [tests/test_release.py](tests/test_release.py): Release manifest creation, offline verification, and tamper detection.
  - [tests/test_shadow.py](tests/test_shadow.py): Decision recording, disposition metrics, and deferred ground-truth agreement.
- **Lint**: All checks clean (`ruff check .`).
