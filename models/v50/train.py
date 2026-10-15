#!/usr/bin/env python3
"""V50: exact V15 training with half-sized initial resident write matrices.

Only construction differs: the unchanged V15 forward consumes write weights
initialized at half their usual value. No runtime field multiplier is present.
The optimizer, raw-position clipping, sampler, LR horizon and CE remain V15.
Standalone runs freeze model.py, train.py and unchanged v15.py.
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

if __package__:
    from ..models.v50 import GPT, ModelConfig
else:
    from model import GPT, ModelConfig


def source_files():
    if __package__:
        models = Path(__file__).resolve().parents[1] / "models"
        return {"train.py": Path(__file__), "model.py": models / "v50.py",
                "v15.py": models / "v15.py"}
    return {name: Path(__file__).with_name(name)
            for name in ("train.py", "model.py", "v15.py")}


def source_hashes_now():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files().items()}


EFFECTIVE_BATCH_TOKENS = 32768
VALIDATION_TOKENS = 131072
BASELINE_PARAMETERS = 29676544
DEFAULT_DATA = "data/fineweb-edu-1b-v1"


def model_config_from_args(args):
    values = json.loads(Path(args.model_config).read_text()) if args.model_config else {}
    return ModelConfig(**values)


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


def validate_position_direction_lr_scale(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("position_direction_lr_scale must be finite and positive")
    return float(value)


def optimizer_variation(learning_rate, weight_decay, position_direction_lr_scale):
    scale = validate_position_direction_lr_scale(position_direction_lr_scale)
    return {
        "name": "unit_sphere_tangent_adam",
        "position_parameter_suffix": ".field.raw_positions",
        "position_direction_lr_scale": scale,
        "position_peak_lr": learning_rate * scale,
        "position_weight_decay": 0.0,
        "position_initialization": "unit_norm",
        "position_clip_reference": "initial_raw_norm",
        "position_moments": "ambient_adam_no_transport",
        "base_peak_lr": learning_rate,
        "base_weight_decay": weight_decay,
    }


@torch.no_grad()
def normalize_resident_positions(model):
    """Preserve directions/RNG; return fixed raw norms for clipping comparison."""
    reference_norms = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".field.raw_positions"):
            norms = parameter.norm(dim=-1, keepdim=True)
            if not torch.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Resident position initialization must be finite and nonzero")
            reference_norms[name] = norms.clone()
            parameter.div_(norms)
    if not reference_norms:
        raise ValueError("model has no trainable resident-position parameters")
    return reference_norms


class TangentAdamW(torch.optim.AdamW):
    """Native AdamW elsewhere; ambient Adam moments with tangent sphere steps.

    This is an explicit projected adaptive update, not a Riemannian optimizer
    with vector transport. No radial weight decay or angle cap is applied.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        position_groups = [g for g in self.param_groups if g["group_name"] == "positions"]
        held = []
        for group in position_groups:
            if group["weight_decay"] != 0:
                raise ValueError("Unit-sphere positions must have zero weight decay")
            for parameter in group["params"]:
                norms = parameter.norm(dim=-1)
                if not torch.isfinite(parameter).all() or not torch.allclose(norms, torch.ones_like(norms), atol=2e-6, rtol=0):
                    raise FloatingPointError("Resident position is non-finite or not unit length")
                gradient = parameter.grad
                if gradient is not None and (gradient.is_sparse or not torch.isfinite(gradient).all()):
                    raise FloatingPointError("Position gradient must be dense and finite")
                held.append((parameter, gradient))
        # Let PyTorch apply its unchanged (including fused CUDA) AdamW update to
        # every non-position parameter. Skipping positions also preserves their
        # custom state metadata without invoking native AdamW on it.
        for parameter, _ in held:
            parameter.grad = None
        try:
            super().step()
        finally:
            for parameter, gradient in held:
                parameter.grad = gradient
        turn_angles, all_norms = [], []
        for group in position_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                state = self.state[parameter]
                if gradient is None:
                    turn_angles.append(torch.zeros_like(parameter[..., 0], dtype=torch.float64))
                    all_norms.append(parameter.norm(dim=-1))
                    continue
                if "step" not in state:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"].add_(1)
                step = int(state["step"].item())
                state["exp_avg"].lerp_(gradient, 1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denominator = (state["exp_avg_sq"].sqrt() / math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                adaptive = (state["exp_avg"] / (1 - beta1 ** step)) / denominator
                before = torch.nn.functional.normalize(parameter, dim=-1)
                tangent = adaptive - (adaptive * before).sum(dim=-1, keepdim=True) * before
                candidate = before - group["lr"] * tangent
                candidate = torch.nn.functional.normalize(candidate, dim=-1)
                # An initial zero-gradient/no-momentum resident has exactly no
                # update. Avoid introducing normalization-only round-off drift.
                candidate = torch.where(tangent.square().sum(-1, keepdim=True) == 0, parameter, candidate)
                if not torch.isfinite(candidate).all() or not torch.isfinite(state["exp_avg"]).all() or not torch.isfinite(state["exp_avg_sq"]).all():
                    raise FloatingPointError("Non-finite unit-sphere Adam update")
                old_direction = torch.nn.functional.normalize(parameter.double(), dim=-1)
                new_direction = torch.nn.functional.normalize(candidate.double(), dim=-1)
                dot = (old_direction * new_direction).sum(dim=-1)
                perpendicular = new_direction - dot.unsqueeze(-1) * old_direction
                turn_angles.append(torch.rad2deg(torch.atan2(perpendicular.norm(dim=-1), dot)))
                parameter.copy_(candidate)
                all_norms.append(parameter.norm(dim=-1))
        angles = torch.cat([a.flatten() for a in turn_angles])
        norms = torch.cat([n.flatten() for n in all_norms])
        self._position_step_diagnostics = {
            "mean_turn_degrees": angles.mean().item(),
            "p95_turn_degrees": torch.quantile(angles, 0.95).item(),
            "max_turn_degrees": angles.max().item(),
            "raw_norm_min": norms.min().item(),
            "raw_norm_max": norms.max().item(),
        }
        return loss

    def position_step_diagnostics(self):
        return dict(getattr(self, "_position_step_diagnostics", {}))


def make_optimizer(model, device, learning_rate, weight_decay, position_direction_lr_scale=0.885,
                   position_reference_norms=None):
    scale = validate_position_direction_lr_scale(position_direction_lr_scale)
    if position_reference_norms is None:
        position_reference_norms = normalize_resident_positions(model)
    named = list(model.named_parameters())
    position_names = {name for name, p in named if name.endswith(".field.raw_positions")}
    if not position_names or set(position_reference_norms) != position_names:
        raise ValueError("Position reference norms must cover exactly the resident parameters")
    positions = [p for name, p in named if name in position_names]
    others = [p for name, p in named if name not in position_names]
    groups = [
        {"params": [p for p in others if p.ndim >= 2], "weight_decay": weight_decay,
         "group_name": "decay", "lr_scale": 1.0},
        {"params": [p for p in others if p.ndim < 2], "weight_decay": 0.0,
         "group_name": "no_decay", "lr_scale": 1.0},
        {"params": positions, "weight_decay": 0.0,
         "group_name": "positions", "lr_scale": scale, "lr": learning_rate * scale, "fused": False,
         "position_optimizer": "unit_sphere_tangent_adam",
         "position_clip_reference": "initial_raw_norm"},
    ]
    optimizer = TangentAdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8,
                            fused=device.type == "cuda")
    for name, parameter in named:
        if name in position_names:
            norms = position_reference_norms[name].to(device=parameter.device, dtype=parameter.dtype)
            if norms.shape != parameter.shape[:-1] + (1,) or not torch.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Position reference norms must be positive finite per-resident scalars")
            if not torch.allclose(parameter.norm(dim=-1), torch.ones_like(parameter[..., 0]), atol=2e-6, rtol=0):
                raise ValueError("Positions must be normalized before constructing the optimizer")
            optimizer.state[parameter]["position_reference_norm"] = norms.clone()
    return optimizer


@torch.no_grad()
def clip_gradients(model, optimizer, max_norm):
    """Clip in original raw-position units, then apply that scale to all grads.

    For positions only, evaluate the norm of grad / initial_raw_norm. This is
    equivalent to dividing before shared clipping and restoring afterward, but
    avoids a round-trip division/multiplication of the stored gradients.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    position_ids = {id(p) for g in optimizer.param_groups if g["group_name"] == "positions" for p in g["params"]}
    gradients, norm_gradients = [], []
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradients.append(parameter.grad)
            norm_gradients.append(parameter.grad / optimizer.state[parameter]["position_reference_norm"]
                                  if id(parameter) in position_ids else parameter.grad)
    if not gradients:
        return torch.tensor(0.0)
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2) for g in norm_gradients]), 2)
    coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    torch._foreach_mul_(gradients, coefficient)
    return total_norm


def set_learning_rate(optimizer, base_learning_rate):
    """Apply the unchanged full-horizon schedule to every optimizer group."""
    for group in optimizer.param_groups:
        group["lr"] = base_learning_rate * group.get("lr_scale", 1.0)


def restore_training_args(args, saved):
    """Resume the recorded recipe; CLI settings cannot change its optimizer."""
    for field in ("seed", "tokens", "microbatch", "lr", "min_lr", "weight_decay",
                  "warmup_steps", "data_dir", "compile", "eval_every", "position_direction_lr_scale"):
        setattr(args, field, saved[field])
    validate_position_direction_lr_scale(args.position_direction_lr_scale)
    expected = {"position_optimizer": "unit_sphere_tangent_adam", "position_weight_decay": 0.0,
                "position_initialization": "unit_norm", "position_clip_reference": "initial_raw_norm"}
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint direction optimizer recipe is incompatible")
    if saved.get("position_peak_lr") != args.lr * args.position_direction_lr_scale:
        raise ValueError("Checkpoint position peak LR contradicts its saved scale")


def load_optimizer_state(optimizer, state):
    """Reject incompatible groups or missing/corrupt fixed reference norms."""
    keys = ("group_name", "lr_scale", "weight_decay", "betas", "eps",
            "position_optimizer", "position_clip_reference")
    expected = [tuple(group.get(key) for key in keys) for group in optimizer.param_groups]
    recorded = [tuple(group.get(key) for key in keys) for group in state["param_groups"]]
    if recorded != expected:
        raise ValueError("Checkpoint optimizer groups differ from the recorded direction recipe")
    for group, saved_group in zip(optimizer.param_groups, state["param_groups"]):
        if len(group["params"]) != len(saved_group["params"]):
            raise ValueError("Checkpoint optimizer parameter count differs")
        if group["group_name"] != "positions":
            continue
        for parameter, saved_id in zip(group["params"], saved_group["params"]):
            saved_state = state["state"].get(saved_id, {})
            norms = saved_state.get("position_reference_norm")
            if norms is None or norms.shape != parameter.shape[:-1] + (1,) or not torch.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Checkpoint position reference norms are missing or invalid")
            original = optimizer.state[parameter]["position_reference_norm"]
            if not torch.equal(original.cpu(), norms.cpu()):
                raise ValueError("Checkpoint position reference norms differ from seeded initialization")
            if "step" in saved_state:
                for name in ("exp_avg", "exp_avg_sq"):
                    moment = saved_state.get(name)
                    if moment is None or moment.shape != parameter.shape or not torch.isfinite(moment).all():
                        raise ValueError("Checkpoint position moments are missing or invalid")
    optimizer.load_state_dict(state)


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
    loss = weighted_loss / count
    result = {"val_loss": loss, "val_perplexity": math.exp(loss), "val_tokens": count}
    raw_model = getattr(model, "_orig_mod", model)
    if hasattr(raw_model, "particle_diagnostics"):
        # A fixed sequence makes these observational diagnostics independent
        # of how validation windows are partitioned into microbatches.
        diagnostic_x, _ = data.batch(data.val_offsets[:1], "val", device)
        with autocast_for(device, dtype):
            result["particle_layers"] = raw_model.particle_diagnostics(diagnostic_x)
        result["diagnostic_validation_offset"] = int(data.val_offsets[0])
    elif hasattr(raw_model, "field"):
        with autocast_for(device, dtype):
            query = raw_model.encode(x[:1])[:, :32]
            result["field_diagnostics"] = raw_model.field.diagnostics(query)
            evolved = raw_model.field(query)
            result["field_diagnostics"]["final_query_rms"] = evolved.float().square().mean().sqrt().item()
            result["field_diagnostics"]["trajectory_displacement_rms"] = (evolved.float() - query.float()).square().mean().sqrt().item()
    model.train(was_training)
    return result


def cosine_lr(completed_steps, total_steps, peak, floor, warmup):
    warmup = min(warmup, max(1, total_steps // 10))
    if completed_steps < warmup:
        return peak * (completed_steps + 1) / warmup
    ratio = min(1.0, (completed_steps - warmup) / max(1, total_steps - warmup - 1))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * ratio))


def benchmark(args, device, dtype):
    seed_everything(args.seed)
    raw_model = GPT(model_config_from_args(args))
    position_reference_norms = normalize_resident_positions(raw_model)
    raw_model = raw_model.to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizer = make_optimizer(raw_model, device, args.lr, args.weight_decay, args.position_direction_lr_scale, position_reference_norms)
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
        clip_gradients(raw_model, optimizer, 1.0)
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
        "model_config": raw_model.config_dict(),
        "optimizer_variation": optimizer_variation(args.lr, args.weight_decay, args.position_direction_lr_scale),
        "source_sha256": source_hashes_now(),
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
    if args.result_json:
        atomic_json(args.result_json, result)
    print(json.dumps(result, indent=2), flush=True)


def train(args, device, dtype):
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pt"
    checkpoint = None
    source_hashes = source_hashes_now()
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        original_manifest = json.loads((run_dir / "manifest.json").read_text())
        if original_manifest["source_sha256"] != source_hashes:
            raise ValueError("Model or trainer source differs from the original run. Restore run-dir/source files to resume, or start a new experiment.")
        saved = checkpoint["training_config"]
        # The original training horizon, data, and optimizer settings survive a
        # restart; --tokens cannot silently extend a resumed experiment.
        restore_training_args(args, saved)
    elif checkpoint_path.exists() or (run_dir / "metrics.jsonl").exists() or (run_dir / "initial.pt").exists():
        raise FileExistsError(f"Run directory already contains a run. Use --resume or a new --run-dir: {run_dir}")

    seed_everything(args.seed)
    raw_model = GPT(ModelConfig(**checkpoint["model_config"]) if checkpoint else model_config_from_args(args))
    position_reference_norms = normalize_resident_positions(raw_model)
    raw_model = raw_model.to(device)
    context_length = raw_model.config.context_length
    microbatch_tokens = args.microbatch * context_length
    if EFFECTIVE_BATCH_TOKENS % microbatch_tokens:
        raise ValueError(f"microbatch × context must divide {EFFECTIVE_BATCH_TOKENS} tokens.")
    accumulation_steps = EFFECTIVE_BATCH_TOKENS // microbatch_tokens
    total_steps = math.ceil(args.tokens / EFFECTIVE_BATCH_TOKENS)
    data = TokenData(args.data_dir, context_length, args.seed)
    optimizer = make_optimizer(raw_model, device, args.lr, args.weight_decay, args.position_direction_lr_scale, position_reference_norms)
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
        load_optimizer_state(optimizer, checkpoint["optimizer"])
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
        "position_direction_lr_scale": args.position_direction_lr_scale,
        "position_weight_decay": 0.0,
        "position_optimizer": "unit_sphere_tangent_adam",
        "position_initialization": "unit_norm",
        "position_clip_reference": "initial_raw_norm",
        "position_peak_lr": args.lr * args.position_direction_lr_scale,
        "warmup_steps": args.warmup_steps, "data_dir": str(data.directory),
        "compile": args.compile, "eval_every": args.eval_every,
        "effective_batch_tokens": EFFECTIVE_BATCH_TOKENS,
        "accumulation_steps": accumulation_steps, "total_steps": total_steps,
        "actual_token_budget": total_steps * EFFECTIVE_BATCH_TOKENS,
    }
    manifest = {
        "model_config": raw_model.config_dict(), "parameters": raw_model.parameter_count(),
        "training_config": training_config, "data_identity": data.identity,
        "optimizer_variation": optimizer_variation(args.lr, args.weight_decay, args.position_direction_lr_scale),
        "data_metadata": data.metadata, "precision": str(dtype), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "compiled": args.compile, "validation_seed": 12345,
        "validation_tokens": VALIDATION_TOKENS,
        "reproducibility": "Fixed seeds/data order; GPU kernels can vary numerically across hardware/software.",
        "source_sha256": source_hashes,
        "field_update": {"kind": "original_v15", "scale": 1.0,
                         "learned": False, "post_sum_activation": None},
        "write_initialization": {"scale": 0.5, "matrices": 8,
                                 "runtime_multiplier": False, "extra_random_draws": 0},
        "shared_initialization": {
            "schema": "v50-v15-constructor-replay-v1",
            "source": "V15 constructor replay with the same seed and configuration",
            "source_step": 0, "source_tokens": 0,
            "copied_parameters": 0,
            "shared_parameters": raw_model.parameter_count() - sum(block.field.down.weight.numel() for block in raw_model.blocks),
            "scaled_parameters": sum(block.field.down.weight.numel() for block in raw_model.blocks),
            "method": "Original V15 constructor replayed with the same seed; only field.down.weight tensors are halved afterward; no extra RNG draws",
            "routing": "Original learned full-width projection into four normalized routing spaces",
            "trained_checkpoint_loaded": False,
        },
        "baseline_reference": {
            "parameters": BASELINE_PARAMETERS,
            "parameter_difference": raw_model.parameter_count() - BASELINE_PARAMETERS,
            "parameter_difference_percent": 100 * (raw_model.parameter_count() / BASELINE_PARAMETERS - 1),
            "comparison": "Halve only the initial resident write weights, then use the original V15 forward and training recipe. V50 and V48 start with the same mathematical field, but differ in write gradients and potentially shared clipping. This is not a pure learning-rate ablation.",
            "limitation": "One seed and a fixed validation sample; an improvement is not a statistical robustness claim.",
        },
    }
    if not checkpoint:
        source_directory = run_dir / "source"
        source_directory.mkdir(exist_ok=True)
        for name, source_path in source_files().items():
            (source_directory / name).write_bytes(source_path.read_bytes())
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
            "position_direction_lr_scale": args.position_direction_lr_scale,
            "position_lr": next(group["lr"] for group in optimizer.param_groups if group["group_name"] == "positions"),
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
        if completed_steps == 0:
            initial_path = run_dir / "initial.pt"
            if not initial_path.exists():
                os.link(checkpoint_path, initial_path)
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
        while completed_steps < total_steps and not stop_requested and (args.max_steps is None or completed_steps < args.max_steps):
            synchronize(device)
            step_started = time.perf_counter()
            # FP16 can overflow at its initial dynamic loss scale. Retry the
            # identical samples at a smaller scale without advancing the run.
            retry_state = None
            if scaler.is_enabled():
                retry_state = (data.sampler.bit_generator.state, torch.get_rng_state(), torch.cuda.get_rng_state_all())
            learning_rate = cosine_lr(completed_steps, total_steps, args.lr, args.min_lr, args.warmup_steps)
            set_learning_rate(optimizer, learning_rate)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for _ in range(accumulation_steps):
                x, y = data.training_batch(args.microbatch, device)
                with autocast_for(device, dtype):
                    _, loss = model(x, y)
                loss_sum += loss.detach()
                scaler.scale(loss / accumulation_steps).backward()
            scaler.unscale_(optimizer)
            gradient_norm = clip_gradients(raw_model, optimizer, 1.0)
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
            with (run_dir / "position_updates.jsonl").open("a") as position_log:
                position_log.write(json.dumps({"step": completed_steps, "base_lr": learning_rate,
                    "position_lr": learning_rate * args.position_direction_lr_scale,
                    **optimizer.position_step_diagnostics()}, allow_nan=False) + "\n")
            train_loss = loss_sum.item() / accumulation_steps
            if not math.isfinite(train_loss):
                raise FloatingPointError("Training loss is not finite.")

            validate_now = completed_steps % args.eval_every == 0 or completed_steps == total_steps or completed_steps == args.max_steps
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
        result = status("completed" if completed_steps >= total_steps else "paused", best_val_loss=best_loss,
                        pause_reason="requested step limit" if completed_steps == args.max_steps else None)
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
    if source_hashes_now() != original_manifest["source_sha256"]:
        raise ValueError("V50 source differs from the frozen checkpoint dependencies")
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
    mode.add_argument("--train", action="store_true", help="Train the particle model from scratch, or resume it.")
    mode.add_argument("--sample", action="store_true", help="Generate text from an existing checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume run-dir/latest.pt with its original data, settings and training horizon.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA, help="Directory containing the unchanged baseline token files.")
    parser.add_argument("--run-dir", default="runs/particle-v50", help="Output directory for manifest, metrics, status, latest.pt and best.pt.")
    parser.add_argument("--model-config", help="JSON ModelConfig overrides for a new run or benchmark; resume always uses its saved configuration.")
    parser.add_argument("--max-steps", type=int, help="Pause after this total optimizer update count, preserving the full token budget and LR schedule.")
    parser.add_argument("--result-json", help="Save benchmark measurements to this JSON path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--microbatch", type=int, default=4, help="Sequences per forward pass; effective batch stays 32,768 tokens.")
    parser.add_argument("--tokens", type=int, default=100_000_000, help="Training token budget, rounded up to a full optimizer step.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--position-direction-lr-scale", type=float, default=0.8671654817246768,
                        help="Unit-sphere tangent Adam LR relative to the base schedule; calibrated independently of validation loss.")
    parser.add_argument("--eval-every", type=int, default=100, help="Validation and checkpoint interval; matches the baseline.")
    parser.add_argument("--compile", action="store_true", help="Opt into torch.compile; first invocation includes compilation overhead.")
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-steps", type=int, default=5)
    parser.add_argument("--checkpoint", default="best.pt", choices=("best.pt", "latest.pt"))
    parser.add_argument("--prompt", default="Once upon a time, there was a little")
    parser.add_argument("--sample-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    try:
        validate_position_direction_lr_scale(args.position_direction_lr_scale)
    except ValueError as error:
        parser.error(str(error))
    if args.microbatch <= 0 or args.tokens <= 0 or args.eval_every <= 0 or args.benchmark_steps <= 0:
        parser.error("microbatch, tokens, eval-every and benchmark-steps must be positive.")
    if args.warmup_steps < 0 or args.benchmark_warmup < 0 or args.temperature <= 0:
        parser.error("warmup counts must be non-negative and temperature must be positive.")
    if args.resume and not args.train:
        parser.error("--resume is only meaningful with --train.")
    if args.max_steps is not None and (args.max_steps <= 0 or not args.train):
        parser.error("--max-steps must be positive and used with --train.")
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
