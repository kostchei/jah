"""Standalone training runner for LoRA decision head adaptation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from jah.compiler import compile_request
from jah.m1_dataset import build_split_manifest, load_m1_dataset
from jah.training.adapter import (
    LoRAConfig,
    compute_decision_loss,
    export_adapter,
    inject_lora,
)
from jah.workload import load_yaml


def train_adapter(
    *,
    dataset_path: Path | str,
    output_dir: Path | str,
    model_config_path: Path | str = "configs/models/qwen3.5-4b.yaml",
    training_config_path: Path | str = "configs/training/lora.yaml",
    split_seed: str = "jah-public-v1",
    workload: str | None = None,
    max_samples: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    gradient_accumulation_steps: int = 4,
    max_input_tokens: int = 2048,
    device: str = "cuda",
    seed: int = 42,
    log_interval: int = 25,
) -> dict[str, Any]:
    """Train a LoRA adapter on decision scoring data and export the bundle."""
    dataset_path = Path(dataset_path).resolve()
    output_dir = Path(output_dir).resolve()
    model_config_path = Path(model_config_path).resolve()
    training_config_path = Path(training_config_path).resolve()

    examples = load_m1_dataset(dataset_path)
    manifest = build_split_manifest(examples, dataset_path=dataset_path, seed=split_seed)
    train_examples = [e for e in examples if manifest["assignments"][e.example_id] == "train"]

    if workload:
        train_examples = [e for e in train_examples if e.workload_id == workload]

    if max_samples and max_samples < len(train_examples):
        # Sample proportionally across workloads if multiple are present
        rng = random.Random(seed)
        workloads = sorted({e.workload_id for e in train_examples})
        if len(workloads) > 1:
            per_workload = max_samples // len(workloads)
            sampled = []
            for w in workloads:
                w_examples = [e for e in train_examples if e.workload_id == w]
                rng.shuffle(w_examples)
                sampled.extend(w_examples[:per_workload])
            train_examples = sampled
        else:
            rng.shuffle(train_examples)
            train_examples = train_examples[:max_samples]

    if not train_examples:
        raise ValueError(f"no training examples found for workload={workload}")

    model_config = load_yaml(model_config_path)
    model_id = model_config["model_id"]
    revision = model_config["revision"]

    lora_dict = load_yaml(training_config_path)
    if epochs is not None:
        lora_dict["epochs"] = epochs
    if batch_size is not None:
        lora_dict["batch_size"] = batch_size
    if learning_rate is not None:
        lora_dict["learning_rate"] = learning_rate

    lora_config = LoRAConfig(**lora_dict)

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        low_cpu_mem_usage=True,
    ).to(dev)

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    if use_cuda:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_modules = inject_lora(model, lora_config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"Injected {len(lora_modules)} LoRA modules. "
        f"Trainable parameters: {total_trainable:,} / {total_params:,} "
        f"({total_trainable / total_params * 100:.3f}%)",
        flush=True,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=lora_config.learning_rate)
    model.train()

    loss_history: list[dict[str, Any]] = []
    total_steps = 0
    start_time = time.perf_counter()

    for epoch in range(lora_config.epochs):
        epoch_rng = random.Random(seed + epoch)
        shuffled = list(train_examples)
        epoch_rng.shuffle(shuffled)

        optimizer.zero_grad()
        accumulated_loss = 0.0

        for idx, example in enumerate(shuffled):
            try:
                compiled = compile_request(
                    example.request, tokenizer=tokenizer, max_input_tokens=8192
                )
                for q_compiled in compiled:
                    if q_compiled.label_token_ids is None:
                        continue
                    if q_compiled.input_tokens and q_compiled.input_tokens > max_input_tokens:
                        continue
                    q_id = q_compiled.question_id
                    reference = example.reference_answers[q_id]

                    if reference not in q_compiled.option_ids:
                        continue
                    target_idx = q_compiled.option_ids.index(reference)

                    encoded = tokenizer(
                        q_compiled.prompt,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    encoded = {k: v.to(dev) for k, v in encoded.items()}

                    outputs = model(**encoded, use_cache=False)
                    last_logits = outputs.logits[:, -1, :]

                    loss = compute_decision_loss(
                        last_logits,
                        [target_idx],
                        q_compiled.label_token_ids,
                    )
                    loss_scaled = loss / gradient_accumulation_steps
                    loss_scaled.backward()
                    accumulated_loss += loss.item()
            except torch.OutOfMemoryError:
                if use_cuda:
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                accumulated_loss = 0.0
                continue

            if (idx + 1) % gradient_accumulation_steps == 0 or (idx + 1) == len(shuffled):
                if lora_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, lora_config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                total_steps += 1

                step_loss = accumulated_loss / gradient_accumulation_steps
                loss_history.append(
                    {
                        "step": total_steps,
                        "epoch": epoch + 1,
                        "example_idx": idx + 1,
                        "loss": round(step_loss, 5),
                    }
                )
                accumulated_loss = 0.0

                if total_steps % log_interval == 0 or (idx + 1) == len(shuffled):
                    print(
                        f"Epoch {epoch + 1}/{lora_config.epochs} | "
                        f"Step {total_steps} | Example {idx + 1}/{len(shuffled)} | "
                        f"Loss: {step_loss:.4f}",
                        flush=True,
                    )

    elapsed = time.perf_counter() - start_time
    print(f"Training completed in {elapsed:.1f}s across {total_steps} optimizer steps.", flush=True)

    # Export adapter bundle
    adapter_manifest = export_adapter(
        lora_modules,
        lora_config,
        output_dir,
        base_model=model_id,
        base_revision=revision,
        dataset_sha256=manifest["dataset_sha256"],
    )

    report = {
        "schema_version": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total_optimizer_steps": total_steps,
        "epochs": lora_config.epochs,
        "train_samples": len(train_examples),
        "trainable_parameters": total_trainable,
        "total_parameters": total_params,
        "initial_loss": loss_history[0]["loss"] if loss_history else None,
        "final_loss": loss_history[-1]["loss"] if loss_history else None,
        "loss_history": loss_history,
        "manifest": adapter_manifest,
    }

    report_path = output_dir / "training-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/data/public/suite.jsonl")
    parser.add_argument("--suite-config", default="configs/evals/public-suite.yaml")
    parser.add_argument("--split-seed", default="jah-public-v1")
    parser.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    parser.add_argument("--training-config", default="configs/training/lora.yaml")
    parser.add_argument("--workload", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="artifacts/adapters/qwen3.5-4b-public-v1")
    parser.add_argument("--log-interval", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_adapter(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        split_seed=args.split_seed,
        workload=args.workload,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_input_tokens=args.max_input_tokens,
        device=args.device,
        log_interval=args.log_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
