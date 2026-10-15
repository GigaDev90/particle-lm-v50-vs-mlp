#!/usr/bin/env python3
"""Train and compare a single ~30M GPT baseline on fixed tokenized data.

Example:
  python train.py --benchmark --data-dir ../../work/data/tinystories --microbatch 8
  python train.py --train --data-dir ../../work/data/tinystories --run-dir runs/baseline
  python train.py --train --resume --run-dir runs/baseline

The default 100M-token budget is a comparison run, not a claim of convergence.
Checkpoints are trusted local PyTorch files and must not be loaded from strangers.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from model import GPT, ModelConfig


EFFECTIVE_BATCH_TOKENS = 32768
VALIDATION_TOKENS = 131072


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Fixed sampling and seeds are reproducible on the same software/hardware;
    # fast GPU kernels are not promised to be bitwise deterministic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def precision_for(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast_for(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()


def make_optimizer(model, device, learning_rate, weight_decay):
    # Tied embedding/head is counted once by named_parameters().
    parameters = list(model.parameters())
    groups = [
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), fused=device.type == "cuda")


class TokenData:
    def __init__(self, directory, context_length, seed):
        self.directory = Path(directory).resolve()
        self.context_length = context_length
        self.train = self._open("train.bin")
        self.val = self._open("val.bin")
        self.sampler = np.random.default_rng(seed)
        fixed = np.random.default_rng(12345)
        self.val_offsets = fixed.integers(0, len(self.val) - context_length,
                                          size=VALIDATION_TOKENS // context_length)
        metadata = self.directory / "metadata.json"
        self.metadata = json.loads(metadata.read_text()) if metadata.exists() else {}
        self.identity = {
            "directory": str(self.directory),
            "train_tokens": len(self.train), "val_tokens": len(self.val),
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest() if metadata.exists() else None,
        }

    def _open(self, name):
        path = self.directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; prepare the tokenized dataset first.")
        if path.stat().st_size % np.dtype(np.uint16).itemsize:
            raise ValueError(f"{path} is not a complete uint16 token file.")
        result = np.memmap(path, mode="r", dtype=np.uint16)
        if len(result) <= self.context_length:
            raise ValueError(f"{path} has too few tokens for one sequence.")
        return result

    def batch(self, offsets, split, device):
        source = self.train if split == "train" else self.val
        block = np.stack([source[int(i):int(i) + self.context_length + 1] for i in offsets]).astype(np.int64)
        # Copy avoids read-only mmap tensors and gives contiguous CUDA transfers.
        x = torch.from_numpy(block[:, :-1].copy())
        y = torch.from_numpy(block[:, 1:].copy())
        if device.type == "cuda":
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def training_batch(self, size, device):
        offsets = self.sampler.integers(0, len(self.train) - self.context_length, size=size)
        return self.batch(offsets, "train", device)


@torch.no_grad()
def evaluate(model, data, microbatch, device, dtype):
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    count = 0
    for start in range(0, len(data.val_offsets), microbatch):
        x, y = data.batch(data.val_offsets[start:start + microbatch], "val", device)
        with autocast_for(device, dtype):
            _, loss = model(x, y)
        weighted_loss += loss.item() * y.numel()
        count += y.numel()
    model.train(was_training)
    loss = weighted_loss / count
    return {"val_loss": loss, "val_perplexity": math.exp(loss), "val_tokens": count}


def cosine_lr(completed_steps, total_steps, peak, floor, warmup):
    warmup = min(warmup, max(1, total_steps // 10))
    if completed_steps < warmup:
        return peak * (completed_steps + 1) / warmup
    ratio = min(1.0, (completed_steps - warmup) / max(1, total_steps - warmup - 1))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * ratio))


def benchmark(args, device, dtype):
    seed_everything(args.seed)
    raw_model = GPT().to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizer = make_optimizer(raw_model, device, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    data = TokenData(args.data_dir, raw_model.config.context_length, args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timed_losses = []
    times = []
    model.train()
    for index in range(args.benchmark_warmup + args.benchmark_steps):
        synchronize(device)
        started = time.perf_counter()
        x, y = data.training_batch(args.microbatch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device, dtype):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= args.benchmark_warmup:
            times.append(elapsed)
            timed_losses.append(loss.item())
    tokens = args.microbatch * raw_model.config.context_length * len(times)
    result = {
        "mode": "benchmark", "parameters": raw_model.parameter_count(),
        "microbatch": args.microbatch, "context_length": raw_model.config.context_length,
        "precision": str(dtype), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "compiled": args.compile,
        "timed_steps": len(times), "timed_tokens": tokens,
        "elapsed_seconds": sum(times), "tokens_per_second": tokens / sum(times),
        "seconds_per_microbatch": sum(times) / len(times),
        "final_loss": timed_losses[-1],
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9 if device.type == "cuda" else None,
        "note": "Includes forward, backward, gradient clipping and optimizer every microbatch; training accumulates gradients.",
    }
    print(json.dumps(result, indent=2), flush=True)


def train(args, device, dtype):
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pt"
    checkpoint = None
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("model.py", "train.py")
    }
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        original_manifest = json.loads((run_dir / "manifest.json").read_text())
        if original_manifest["source_sha256"]["model.py"] != source_hashes["model.py"]:
            raise ValueError("model.py differs from the original run. Restore run-dir/source/model.py to resume, or start an architecture experiment in a new run directory.")
        saved = checkpoint["training_config"]
        # The original training horizon, data, and optimizer settings survive a
        # restart; --tokens cannot silently extend a resumed experiment.
        for field in ("seed", "tokens", "microbatch", "lr", "min_lr", "weight_decay", "warmup_steps", "data_dir", "compile", "eval_every"):
            setattr(args, field, saved[field])
    elif checkpoint_path.exists() or (run_dir / "metrics.jsonl").exists():
        raise FileExistsError(f"Run directory already contains a run. Use --resume or a new --run-dir: {run_dir}")

    seed_everything(args.seed)
    raw_model = GPT(ModelConfig(**checkpoint["model_config"]) if checkpoint else ModelConfig()).to(device)
    context_length = raw_model.config.context_length
    microbatch_tokens = args.microbatch * context_length
    if EFFECTIVE_BATCH_TOKENS % microbatch_tokens:
        raise ValueError(f"microbatch × context must divide {EFFECTIVE_BATCH_TOKENS} tokens.")
    accumulation_steps = EFFECTIVE_BATCH_TOKENS // microbatch_tokens
    total_steps = math.ceil(args.tokens / EFFECTIVE_BATCH_TOKENS)
    target_steps = min(total_steps, args.max_steps) if args.max_steps is not None else total_steps
    data = TokenData(args.data_dir, context_length, args.seed)
    optimizer = make_optimizer(raw_model, device, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    completed_steps = 0
    best_loss = float("inf")
    training_seconds = 0.0
    previous_wall_seconds = 0.0
    last_validation = {}
    if checkpoint:
        if checkpoint["data_identity"] != data.identity:
            raise ValueError("Dataset identity differs from the checkpoint. Restore the original data before resuming.")
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        completed_steps = checkpoint["completed_steps"]
        best_loss = checkpoint["best_val_loss"]
        training_seconds = checkpoint["training_seconds"]
        previous_wall_seconds = checkpoint["wall_seconds"]
        last_validation = checkpoint.get("last_validation", {})
        data.sampler.bit_generator.state = checkpoint["sampler_state"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    model = torch.compile(raw_model) if args.compile else raw_model
    training_config = {
        "seed": args.seed, "tokens": args.tokens, "microbatch": args.microbatch,
        "lr": args.lr, "min_lr": args.min_lr, "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps, "data_dir": str(data.directory),
        "compile": args.compile, "eval_every": args.eval_every,
        "effective_batch_tokens": EFFECTIVE_BATCH_TOKENS,
        "accumulation_steps": accumulation_steps, "total_steps": total_steps,
        "actual_token_budget": total_steps * EFFECTIVE_BATCH_TOKENS,
    }
    manifest = {
        "model_config": raw_model.config_dict(), "parameters": raw_model.parameter_count(),
        "training_config": training_config, "data_identity": data.identity,
        "data_metadata": data.metadata, "precision": str(dtype), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "compiled": args.compile, "validation_seed": 12345,
        "validation_tokens": VALIDATION_TOKENS,
        "reproducibility": "Fixed seeds/data order; GPU kernels can vary numerically across hardware/software.",
        "source_sha256": source_hashes,
    }
    if not checkpoint:
        source_directory = run_dir / "source"
        source_directory.mkdir(exist_ok=True)
        for name in ("model.py", "train.py"):
            (source_directory / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        tokenizer_path = data.directory / "tokenizer.json"
        if tokenizer_path.exists():
            (run_dir / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    atomic_json(run_dir / ("resume_manifest.json" if checkpoint else "manifest.json"), manifest)
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}; saving after the current optimizer step.", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def status(state="running", **extra):
        wall = previous_wall_seconds + time.perf_counter() - start_time
        tokens = completed_steps * EFFECTIVE_BATCH_TOKENS
        result = {
            "state": state, "pid": os.getpid(), "run_dir": str(run_dir),
            "parameters": raw_model.parameter_count(), "completed_steps": completed_steps,
            "total_steps": total_steps, "tokens": tokens, "target_tokens": args.tokens,
            "actual_token_budget": total_steps * EFFECTIVE_BATCH_TOKENS,
            "progress_fraction": completed_steps / total_steps,
            "training_seconds": training_seconds, "wall_seconds": wall,
            "training_tokens_per_second": tokens / training_seconds if training_seconds else None,
            "wall_tokens_per_second": tokens / wall if wall else None,
            "estimated_remaining_training_seconds": (total_steps - completed_steps) * training_seconds / completed_steps if completed_steps else None,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
            "updated_unix": time.time(), **last_validation, **extra,
        }
        atomic_json(run_dir / "status.json", result)
        return result

    def append_metric(value):
        with (run_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")

    def save(is_best=False):
        value = {
            "format_version": 1, "model": raw_model.state_dict(),
            "model_config": raw_model.config_dict(), "training_config": training_config,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "completed_steps": completed_steps, "tokens": completed_steps * EFFECTIVE_BATCH_TOKENS,
            "best_val_loss": best_loss, "last_validation": last_validation,
            "training_seconds": training_seconds,
            "wall_seconds": previous_wall_seconds + time.perf_counter() - start_time,
            "data_identity": data.identity, "sampler_state": data.sampler.bit_generator.state,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        atomic_checkpoint(checkpoint_path, value)
        if is_best:
            atomic_checkpoint(run_dir / "best.pt", value)

    print(json.dumps({"event": "resume" if checkpoint else "start", **manifest}), flush=True)
    status("validating" if not checkpoint else "running")
    if not checkpoint:
        try:
            last_validation = evaluate(model, data, args.microbatch, device, dtype)
            best_loss = last_validation["val_loss"]
            value = status("running", event="validation")
            append_metric(value)
            print(json.dumps(value), flush=True)
            save(is_best=True)
        except Exception as error:
            status("failed", error=f"{type(error).__name__}: {error}")
            raise

    model.train()
    last_status_time = time.perf_counter()
    overflow_retries = 0
    try:
        while completed_steps < target_steps and not stop_requested:
            synchronize(device)
            step_started = time.perf_counter()
            # FP16 can overflow at its initial dynamic loss scale. Retry the
            # identical samples at a smaller scale without advancing the run.
            retry_state = None
            if scaler.is_enabled():
                retry_state = (data.sampler.bit_generator.state, torch.get_rng_state(), torch.cuda.get_rng_state_all())
            learning_rate = cosine_lr(completed_steps, total_steps, args.lr, args.min_lr, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for _ in range(accumulation_steps):
                x, y = data.training_batch(args.microbatch, device)
                with autocast_for(device, dtype):
                    _, loss = model(x, y)
                loss_sum += loss.detach()
                scaler.scale(loss / accumulation_steps).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                if scaler.is_enabled() and overflow_retries < 20:
                    scaler.update(new_scale=scaler.get_scale() / 2)
                    data.sampler.bit_generator.state = retry_state[0]
                    torch.set_rng_state(retry_state[1])
                    torch.cuda.set_rng_state_all(retry_state[2])
                    synchronize(device)
                    training_seconds += time.perf_counter() - step_started
                    overflow_retries += 1
                    print(json.dumps({"event": "fp16_overflow_retry", "step": completed_steps + 1, "scale": scaler.get_scale()}), flush=True)
                    continue
                # A persistent overflow or BF16 failure indicates a broken run.
                raise FloatingPointError(f"Non-finite gradient at step {completed_steps + 1}.")
            scaler.step(optimizer)
            scaler.update()
            overflow_retries = 0
            synchronize(device)
            training_seconds += time.perf_counter() - step_started
            completed_steps += 1
            train_loss = loss_sum.item() / accumulation_steps
            if not math.isfinite(train_loss):
                raise FloatingPointError("Training loss is not finite.")

            validate_now = completed_steps % args.eval_every == 0 or completed_steps == target_steps
            if validate_now:
                status("validating")
                last_validation = evaluate(model, data, args.microbatch, device, dtype)
                improved = last_validation["val_loss"] < best_loss
                best_loss = min(best_loss, last_validation["val_loss"])
                value = status("running", event="validation", train_loss=train_loss, lr=learning_rate)
                append_metric(value)
                print(json.dumps(value), flush=True)
                save(is_best=improved)
                last_status_time = time.perf_counter()
            elif completed_steps % 10 == 0 or time.perf_counter() - last_status_time >= 15:
                value = status(train_loss=train_loss, lr=learning_rate)
                append_metric({"event": "train", **value})
                print(json.dumps(value), flush=True)
                last_status_time = time.perf_counter()

        save()
        result = status("completed" if completed_steps >= total_steps else "paused", best_val_loss=best_loss)
        append_metric({"event": "finish", **result})
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        # latest.pt remains the last consistent optimizer checkpoint on failure.
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


@torch.no_grad()
def sample(args, device, dtype):
    from tokenizers import Tokenizer

    run_dir = Path(args.run_dir).resolve()
    original_manifest = json.loads((run_dir / "manifest.json").read_text())
    model_hash = hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest()
    if model_hash != original_manifest["source_sha256"]["model.py"]:
        raise ValueError("model.py differs from this checkpoint's architecture. Restore run-dir/source/model.py before sampling.")
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data_directory = Path(checkpoint["training_config"]["data_dir"])
    tokenizer_path = Path(args.run_dir).resolve() / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = data_directory / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoded = tokenizer.encode(args.prompt).ids
    if not encoded:
        raise ValueError("The sampling prompt must contain at least one token.")
    seed_everything(args.seed)
    tokens = torch.tensor([encoded], device=device, dtype=torch.long)
    for _ in range(args.sample_tokens):
        with autocast_for(device, dtype):
            logits, _ = model(tokens[:, -model.config.context_length:])
        probabilities = torch.softmax(logits[:, -1, :].float() / args.temperature, dim=-1)
        tokens = torch.cat((tokens, torch.multinomial(probabilities, num_samples=1)), dim=1)
    print(tokenizer.decode(tokens[0].tolist()), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true", help="Measure actual forward/backward/optimizer throughput; saves no model.")
    mode.add_argument("--train", action="store_true", help="Train the ~30M baseline from scratch, or resume it.")
    mode.add_argument("--sample", action="store_true", help="Generate text from an existing checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume run-dir/latest.pt with its original data, settings and training horizon.")
    parser.add_argument("--data-dir", default="data/fineweb-edu-1b-v1", help="Directory containing uint16 train.bin, val.bin, metadata.json and tokenizer.json.")
    parser.add_argument("--run-dir", default="runs/baseline", help="Output directory for manifest, metrics, status, latest.pt and best.pt.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--microbatch", type=int, default=4, help="Sequences per forward pass; effective batch stays 32,768 tokens.")
    parser.add_argument("--tokens", type=int, default=100_000_000, help="Training token budget, rounded up to a full optimizer step.")
    parser.add_argument("--max-steps", type=int, help="Pause at this update while retaining the full token-budget schedule.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=250, help="Validation and checkpoint interval in optimizer steps.")
    parser.add_argument("--compile", action="store_true", help="Opt into torch.compile; first invocation includes compilation overhead.")
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-steps", type=int, default=5)
    parser.add_argument("--checkpoint", default="best.pt", choices=("best.pt", "latest.pt"))
    parser.add_argument("--prompt", default="Once upon a time, there was a little")
    parser.add_argument("--sample-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if args.microbatch <= 0 or args.tokens <= 0 or args.eval_every <= 0 or args.benchmark_steps <= 0:
        parser.error("microbatch, tokens, eval-every and benchmark-steps must be positive.")
    if args.warmup_steps < 0 or args.benchmark_warmup < 0 or args.temperature <= 0:
        parser.error("warmup counts must be non-negative and temperature must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive.")
    if args.resume and not args.train:
        parser.error("--resume is only meaningful with --train.")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = precision_for(device)
    if args.benchmark:
        benchmark(args, device, dtype)
    elif args.train:
        train(args, device, dtype)
    else:
        sample(args, device, dtype)


if __name__ == "__main__":
    main()
