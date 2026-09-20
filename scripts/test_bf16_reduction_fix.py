"""Test whether forcing FP32 accumulation in BF16 matmuls restores batch equivalence.

The divergence diagnosis showed that batching changes final logits by 0.0625 to 0.125 even when
every row has identical length and no padding is emitted. Those step sizes are one to two BF16
units in the last place at logit magnitudes of 16 to 32, which points at reduced-precision
accumulation inside the batched GEMM rather than at any masking or indexing defect.

PyTorch exposes that accumulation choice directly. This script measures the same comparison with
it on and off.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from jah.compiler import compile_request
from jah.evaluation import load_backend
from jah.schemas import EvaluateRequest
from jah.scoring import last_unpadded_logits
from jah.workload import load_yaml


def label_logits(final, question):
    return {
        option_id: float(final[token_id].item())
        for option_id, token_id in zip(question.option_ids, question.label_token_ids, strict=True)
    }


def softmax(values):
    import math

    peak = max(values.values())
    exponentials = {k: math.exp(v - peak) for k, v in values.items()}
    total = sum(exponentials.values())
    return {k: v / total for k, v in exponentials.items()}


def forward(backend, prompts):
    encoded = backend.tokenizer(
        prompts, padding=True, return_tensors="pt", add_special_tokens=False
    )
    encoded = {k: v.to(backend.device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = backend.model(**encoded, use_cache=False, return_dict=True)
    return last_unpadded_logits(outputs.logits, encoded["attention_mask"])


def measure(backend, questions, microbatch_size: int = 8):
    singles = [label_logits(forward(backend, [q.prompt])[0], q) for q in questions]
    all_finals = []
    for i in range(0, len(questions), microbatch_size):
        chunk = questions[i : i + microbatch_size]
        finals_chunk = forward(backend, [q.prompt for q in chunk])
        for idx in range(len(chunk)):
            all_finals.append(finals_chunk[idx])
    worst_logit = 0.0
    worst_probability = 0.0
    flips = 0
    for index, question in enumerate(questions):
        batched = label_logits(all_finals[index], question)
        reference = singles[index]
        worst_logit = max(worst_logit, max(abs(reference[k] - batched[k]) for k in reference))
        rp, bp = softmax(reference), softmax(batched)
        worst_probability = max(worst_probability, max(abs(rp[k] - bp[k]) for k in rp))
        if max(rp, key=rp.get) != max(bp, key=bp.get):
            flips += 1
    return {
        "max_logit_deviation": worst_logit,
        "max_probability_deviation": worst_probability,
        "argmax_flips": flips,
        "decisions": len(questions),
    }


def main() -> int:
    root = Path.cwd()
    model_config = load_yaml(root / "configs/models/qwen3.5-4b.yaml")
    backend = load_backend("direct", model_config)
    fixtures = root / "evals/fixtures/equivalence"

    questions = []
    for name in ("wikiqa-boolean-08.json", "wikiqa-boolean-16.json", "asap-mixed-13.json"):
        request = EvaluateRequest.model_validate_json(
            (fixtures / name).read_text(encoding="utf-8")
        )
        questions.extend(compile_request(request, tokenizer=backend.tokenizer))

    results = {}
    for label, reduced in (("bf16_reduced_reduction_default_on", True), ("fp32_reduction", False)):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
        results[label] = measure(backend, questions)
        results[label]["allow_bf16_reduced_precision_reduction"] = reduced

    print(json.dumps(results, indent=2, sort_keys=True))
    output = root / "artifacts/m3/bf16-reduction-experiment.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
