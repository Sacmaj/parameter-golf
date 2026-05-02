from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


W9_RELATIVE_PATH = (
    "records/track_non_record_16mb/"
    "2026-05-01_psyarbor_lab_rebus_calculations_w9/train_gpt.py"
)
BUCKET_ORDER = (
    "embedding_output",
    "core_sequence",
    "router_controller",
    "adaptation_plasticity",
    "norm_positional",
    "supervisor_state",
    "other",
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    iterations: int
    pretrain_episodes: int
    mode: str
    episodic_objective: str = "lm_only"
    episodic_trainable: str = "base_and_outer"
    arg_overrides: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic W9 distribution-shift diagnostic. Each episode has a hidden "
            "token-shift domain shared by support and query. The metric is held-out "
            "query CE after support adaptation, not FineWeb BPB."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--w9-path", type=Path, default=repo_root / W9_RELATIVE_PATH)
    parser.add_argument("--matrix", choices=("shift32-smoke", "shift128-diagnostic"), default="shift32-smoke")
    parser.add_argument("--single", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-domains", type=int, default=24)
    parser.add_argument("--heldout-domains", type=int, default=8)
    parser.add_argument("--symbol-count", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--train-log-every", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=repo_root / "runs/w9_distribution_shift")
    return parser.parse_args()


def configure_env(cli: argparse.Namespace) -> None:
    os.environ["OUTPUT_DIR"] = str(cli.repo_root)
    os.environ["RUN_ID"] = "debug_w9_distribution_shift"
    os.environ["RETENTION_EVAL_EVERY"] = "0"
    os.environ["VAL_LOSS_EVERY"] = "0"
    os.environ["TRAIN_LOG_EVERY"] = "0"
    os.environ["SEED"] = str(cli.seed)


def import_w9_module(path: Path):
    spec = importlib.util.spec_from_file_location("w9_shift_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import W9 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def autocast_for(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda"))


def make_model(mod, args, device: torch.device):
    model = mod.PsyArborLM(args).to(device)
    if device.type == "cuda":
        model = model.bfloat16()
        for module in model.modules():
            if isinstance(module, mod.CastedLinear):
                module.float()
        mod.restore_low_dim_params_to_fp32(model)
    return model


def param_bucket(name: str) -> str:
    if name.startswith("tok_emb") or name.startswith("lm_head"):
        return "embedding_output"
    if "norm" in name or "rope" in name or "rotary" in name or "pos" in name:
        return "norm_positional"
    if name.startswith(("coarse_router", "router", "plasticity_controller")):
        return "router_controller"
    if name.startswith(("branch_slot_emb", "edge_slot_emb", "branch_hyper", "edge_hyper", "adapter_bank")):
        return "adaptation_plasticity"
    if name.startswith("assemblies"):
        adaptation_tokens = (
            "branch_gate_net",
            "edge_gate_net",
            "stable_branch_scale",
            "stable_edge_gates",
            "stable_branch",
            "stable_edge_ops",
            "branch_slot_basis",
            "edge_slot_basis",
        )
        if any(token in name for token in adaptation_tokens):
            return "adaptation_plasticity"
        return "core_sequence"
    return "other"


def trainable_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def clone_named_params(named_params: list[tuple[str, nn.Parameter]]) -> dict[str, Tensor]:
    return {name: param.detach().clone() for name, param in named_params}


def bucket_stats(named_params: list[tuple[str, nn.Parameter]], before: dict[str, Tensor] | None = None) -> dict[str, dict[str, float]]:
    stats = {
        bucket: {
            "param_count": 0.0,
            "param_norm_sq": 0.0,
            "grad_norm_sq": 0.0,
            "update_norm_sq": 0.0,
            "grad_none_count": 0.0,
            "grad_zero_count": 0.0,
            "grad_nonfinite_count": 0.0,
        }
        for bucket in BUCKET_ORDER
    }
    for name, param in named_params:
        entry = stats[param_bucket(name)]
        entry["param_count"] += float(param.numel())
        p_float = param.detach().float()
        entry["param_norm_sq"] += float(p_float.pow(2).sum().item())
        grad = param.grad
        if grad is None:
            entry["grad_none_count"] += 1.0
        else:
            g_float = grad.detach().float()
            if not torch.isfinite(g_float).all():
                entry["grad_nonfinite_count"] += 1.0
            g_norm_sq = float(g_float.pow(2).sum().item())
            if g_norm_sq == 0.0:
                entry["grad_zero_count"] += 1.0
            entry["grad_norm_sq"] += g_norm_sq
        if before is not None and name in before:
            entry["update_norm_sq"] += float((p_float - before[name].float()).pow(2).sum().item())
    out: dict[str, dict[str, float]] = {}
    for bucket, entry in stats.items():
        param_norm = math.sqrt(entry.pop("param_norm_sq"))
        grad_norm = math.sqrt(entry.pop("grad_norm_sq"))
        update_norm = math.sqrt(entry.pop("update_norm_sq"))
        entry["param_norm"] = param_norm
        entry["grad_norm"] = grad_norm
        entry["update_norm"] = update_norm
        entry["update_to_param"] = update_norm / max(param_norm, 1e-12)
        out[bucket] = entry
    return out


def json_safe(payload: Any) -> Any:
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else None
    if isinstance(payload, dict):
        return {str(key): json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_safe(value) for value in payload]
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def snapshot_diff_count(before: dict[str, Tensor], model: nn.Module) -> int:
    changed = 0
    for name, buf in model.named_buffers():
        if name in before and not torch.equal(buf.detach().cpu(), before[name].detach().cpu()):
            changed += 1
    return changed


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


class ShiftEpisodeGenerator:
    def __init__(
        self,
        *,
        args,
        device: torch.device,
        seed: int,
        train_domains: int,
        heldout_domains: int,
        symbol_count: int,
    ):
        if symbol_count + 8 >= int(args.vocab_size):
            raise ValueError("symbol_count leaves too little room inside vocab")
        if train_domains + heldout_domains >= symbol_count:
            raise ValueError("domain shifts must fit inside symbol_count")
        self.args = args
        self.device = device
        self.seed = int(seed)
        self.symbol_count = int(symbol_count)
        self.token_base = 8
        self.train_shifts = list(range(1, int(train_domains) + 1))
        self.heldout_shifts = list(range(int(train_domains) + 1, int(train_domains) + int(heldout_domains) + 1))

    def _sequence(self, shift: int, episode_idx: int, block_idx: int) -> Tensor:
        gen = torch.Generator(device="cpu").manual_seed(self.seed + int(shift) * 100_003 + int(episode_idx) * 997 + int(block_idx) * 53)
        start = int(torch.randint(0, self.symbol_count, (1,), generator=gen).item())
        values = (start + int(shift) * torch.arange(self.args.train_seq_len + 1, dtype=torch.long)) % self.symbol_count
        return values + self.token_base

    def _to_batch(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        x = tokens[:-1].reshape(1, -1).to(device=self.device, dtype=torch.long)
        y = tokens[1:].reshape(1, -1).to(device=self.device, dtype=torch.long)
        return x, y

    def episode(self, shift: int, episode_idx: int) -> tuple[list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]], tuple[Tensor, Tensor]]:
        support = [
            self._to_batch(self._sequence(shift, episode_idx, block_idx))
            for block_idx in range(int(self.args.support_blocks))
        ]
        query = [
            self._to_batch(self._sequence(shift, episode_idx, 100 + block_idx))
            for block_idx in range(int(self.args.query_blocks))
        ]
        anchor_shift = self.train_shifts[(episode_idx * 7) % len(self.train_shifts)]
        anchor = self._to_batch(self._sequence(anchor_shift, episode_idx, 10_000))
        return support, query, anchor

    def train_episode(self, episode_idx: int) -> tuple[int, list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]], tuple[Tensor, Tensor]]:
        shift = self.train_shifts[episode_idx % len(self.train_shifts)]
        support, query, anchor = self.episode(shift, episode_idx)
        return shift, support, query, anchor

    def heldout_episode(self, episode_idx: int) -> tuple[int, list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]], tuple[Tensor, Tensor]]:
        shift = self.heldout_shifts[episode_idx % len(self.heldout_shifts)]
        support, query, anchor = self.episode(shift, 100_000 + episode_idx)
        return shift, support, query, anchor


def build_optimizer_for_mode(mod, model, args, mode: str, lr: float, device: torch.device):
    if mode == "pretrain":
        model.set_training_phase("pretrain")
    else:
        args.episodic_trainable = mode
        model.set_training_phase("episodic")
    named_params = trainable_named_params(model)
    optimizer = mod.build_optimizer([param for _, param in named_params], args, lr, device)
    return optimizer, named_params


def support_total(mod, trace, temp_pool, args, lm_only: bool) -> Tensor:
    if trace.lm_loss is None:
        raise RuntimeError("support trace has no LM loss")
    if lm_only:
        return trace.lm_loss.float()
    total = (
        trace.lm_loss
        + args.lambda_prox * temp_pool.prox_loss()
        + args.lambda_struct * trace.struct_loss
        + args.lambda_homeo * trace.homeo_loss
        + args.lambda_l0 * trace.l0_penalty
    )
    return total.float() if args.support_total_fp32 else total


def run_support_adaptation(
    *,
    mod,
    model,
    args,
    memory,
    temp_pool,
    support_batches: list[tuple[Tensor, Tensor]],
    device: torch.device,
    lm_only: bool,
    train: bool,
) -> tuple[list[Any], float, float, float]:
    support_traces: list[Any] = []
    before_values: list[float] = []
    after_values: list[float] = []
    proxy_norm = 0.0
    for support_x, support_y in support_batches:
        temp_pool.reset_support_gates(reset_growth=True, reset_probe=True)
        support_ctx = mod.EpisodeContext(
            temp_pool=temp_pool,
            episodic_memory=memory,
            phase="support",
            allow_growth=True,
            enable_temp=True,
        )
        with autocast_for(device):
            _, support_trace = model(support_x, support_y, episode_ctx=support_ctx, return_trace=True)
            support_before = support_total(mod, support_trace, temp_pool, args, lm_only)
        grad_proxies = temp_pool.fast_update(support_before)
        proxy_norm += math.sqrt(sum(float(v) ** 2 for v in grad_proxies.values()))
        temp_pool.reset_support_gates(reset_probe=True)
        probe_ctx = mod.EpisodeContext(
            temp_pool=temp_pool,
            episodic_memory=memory,
            phase="support",
            allow_growth=False,
            enable_temp=True,
        )
        with torch.no_grad():
            with autocast_for(device):
                _, probe_trace = model(support_x, support_y, episode_ctx=probe_ctx, return_trace=True)
                support_after = support_total(mod, probe_trace, temp_pool, args, lm_only)
        before_value = float(support_before.detach().item())
        after_value = float(support_after.detach().item())
        before_values.append(before_value)
        after_values.append(after_value)
        temp_pool.update_online_stats(probe_trace, before_value - after_value)
        memory.write(probe_trace)
        support_traces.append(probe_trace if train else support_trace)
    return support_traces, float(np.mean(before_values)), float(np.mean(after_values)), proxy_norm


def query_loss_and_traces(mod, model, args, memory, temp_pool, query_batches, device: torch.device):
    query_traces: list[Any] = []
    with autocast_for(device):
        for query_x, query_y in query_batches:
            query_ctx = mod.EpisodeContext(
                temp_pool=temp_pool,
                episodic_memory=memory,
                phase="query",
                allow_growth=False,
                enable_temp=True,
            )
            _, query_trace = model(query_x, query_y, episode_ctx=query_ctx, return_trace=True)
            query_traces.append(query_trace)
        query_lm_loss = torch.stack([trace.lm_loss for trace in query_traces if trace.lm_loss is not None]).mean()
        query_struct_loss = torch.stack([trace.struct_loss for trace in query_traces]).mean()
        query_homeo_loss = torch.stack([trace.homeo_loss for trace in query_traces]).mean()
        query_l0_penalty = torch.stack([trace.l0_penalty for trace in query_traces]).mean()
        query_router_spec_loss = torch.stack([trace.router_spec_loss for trace in query_traces]).mean()
    return query_lm_loss, query_traces, query_struct_loss, query_homeo_loss, query_l0_penalty, query_router_spec_loss


def supervisor_update(model, args, affinity_traces: list[Any], query_lm_loss: Tensor) -> dict[str, Any] | None:
    if not args.rebus_supervisor_enabled:
        return None
    fine_probs: list[Tensor] = []
    active_set: set[int] = set()
    for trace in affinity_traces:
        for probs in trace.fine_route_probs_pre_mask:
            fine_probs.append(probs.detach().float().to(device=model.assembly_health.device).reshape(-1))
        for selected in trace.selected_assemblies:
            for aid in selected:
                active_set.add(int(aid))
    mean_probs = (
        torch.stack(fine_probs, dim=0).mean(dim=0)
        if fine_probs
        else torch.full((args.num_assemblies,), 1.0 / float(args.num_assemblies), device=model.assembly_health.device)
    )
    active_mask = torch.zeros(args.num_assemblies, device=model.assembly_health.device, dtype=torch.float32)
    for aid in active_set:
        if 0 <= aid < args.num_assemblies:
            active_mask[aid] = 1.0
    return model.supervisor_step(mean_probs, float(query_lm_loss.detach().item()), active_mask)


def evaluate_shift(
    *,
    mod,
    model,
    args,
    generator: ShiftEpisodeGenerator,
    device: torch.device,
    eval_episodes: int,
    adapt: bool,
) -> dict[str, Any]:
    was_training = model.training
    snapshot = model.snapshot_runtime_buffers()
    model.train()
    support_before: list[float] = []
    support_after: list[float] = []
    query_after: list[float] = []
    query_no_adapt: list[float] = []
    shifts: list[int] = []
    try:
        for idx in range(eval_episodes):
            shift, support_batches, query_batches, _ = generator.heldout_episode(idx)
            shifts.append(int(shift))
            with torch.no_grad():
                with autocast_for(device):
                    no_adapt_losses = [model(x, y).detach().float() for x, y in query_batches]
                    query_no_adapt.append(float(torch.stack(no_adapt_losses).mean().item()))
            if adapt:
                memory = mod.EpisodicMemory(args.memory_size, args.model_dim, args.num_assemblies)
                temp_pool = mod.TempPool(args, device)
                _, before, after, _ = run_support_adaptation(
                    mod=mod,
                    model=model,
                    args=args,
                    memory=memory,
                    temp_pool=temp_pool,
                    support_batches=support_batches,
                    device=device,
                    lm_only=True,
                    train=False,
                )
                query_lm, _, _, _, _, _ = query_loss_and_traces(mod, model, args, memory, temp_pool, query_batches, device)
                support_before.append(before)
                support_after.append(after)
                query_after.append(float(query_lm.detach().float().item()))
            else:
                support_before.append(float("nan"))
                support_after.append(float("nan"))
                query_after.append(query_no_adapt[-1])
    finally:
        changed_before_restore = snapshot_diff_count(snapshot, model)
        model.restore_runtime_buffers(snapshot)
        changed_after_restore = snapshot_diff_count(snapshot, model)
        model.train(was_training)
    return {
        "heldout_shifts": shifts,
        "query_ce_no_adapt": float(np.mean(query_no_adapt)),
        "query_ce_after_support": float(np.mean(query_after)),
        "support_ce_before": safe_mean(support_before),
        "support_ce_after": safe_mean(support_after),
        "support_adaptation_gain": safe_mean(np.asarray(support_before) - np.asarray(support_after)),
        "eval_mutations_before_restore": int(changed_before_restore),
        "eval_mutations_after_restore": int(changed_after_restore),
    }


def top_update_bucket(stats: dict[str, dict[str, float]]) -> str:
    return max(stats.items(), key=lambda item: float(item[1].get("update_to_param", 0.0)))[0] if stats else "none"


def make_specs(matrix: str, single: str) -> list[RunSpec]:
    specs = {
        "shift32_pure_lm": RunSpec("shift32_pure_lm", iterations=32, pretrain_episodes=32, mode="pure"),
        "shift32_episodic_lm_only": RunSpec("shift32_episodic_lm_only", iterations=32, pretrain_episodes=8, mode="episodic", episodic_objective="lm_only"),
        "shift32_episodic_supervisor_off": RunSpec(
            "shift32_episodic_supervisor_off",
            iterations=32,
            pretrain_episodes=8,
            mode="episodic",
            episodic_objective="lm_only",
            arg_overrides={"rebus_supervisor_enabled": False},
        ),
        "shift128_pure_lm": RunSpec("shift128_pure_lm", iterations=128, pretrain_episodes=128, mode="pure"),
        "shift128_episodic_lm_only": RunSpec("shift128_episodic_lm_only", iterations=128, pretrain_episodes=32, mode="episodic", episodic_objective="lm_only"),
        "shift128_episodic_supervisor_off": RunSpec(
            "shift128_episodic_supervisor_off",
            iterations=128,
            pretrain_episodes=32,
            mode="episodic",
            episodic_objective="lm_only",
            arg_overrides={"rebus_supervisor_enabled": False},
        ),
        "shift128_lm_dominant_anchor0_lrmatch": RunSpec(
            "shift128_lm_dominant_anchor0_lrmatch",
            iterations=128,
            pretrain_episodes=32,
            mode="episodic",
            episodic_objective="lm_dominant",
            arg_overrides={"lambda_anchor": 0.0, "outer_lr": "$stable_lr"},
        ),
        "shift128_growth_disabled": RunSpec(
            "shift128_growth_disabled",
            iterations=128,
            pretrain_episodes=32,
            mode="episodic",
            episodic_objective="lm_only",
            arg_overrides={"rebus_growth_budget": 0},
        ),
    }
    if single:
        if single not in specs:
            raise ValueError(f"Unknown --single {single}; available: {sorted(specs)}")
        return [specs[single]]
    if matrix == "shift32-smoke":
        return [specs[name] for name in ["shift32_pure_lm", "shift32_episodic_lm_only", "shift32_episodic_supervisor_off"]]
    if matrix == "shift128-diagnostic":
        return [
            specs[name]
            for name in [
                "shift128_pure_lm",
                "shift128_episodic_lm_only",
                "shift128_episodic_supervisor_off",
                "shift128_lm_dominant_anchor0_lrmatch",
                "shift128_growth_disabled",
            ]
        ]
    raise ValueError(f"Unsupported matrix: {matrix}")


def resolve_override(args, value: Any) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return getattr(args, value[1:])
    return value


def apply_overrides(args, overrides: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown Hyperparameters override: {key}")
        resolved[key] = resolve_override(args, value)
        setattr(args, key, resolved[key])
    return resolved


def run_spec(*, mod, base_args, spec: RunSpec, generator: ShiftEpisodeGenerator, device: torch.device, output_dir: Path, train_log_every: int, eval_episodes: int) -> dict[str, Any]:
    set_seed(int(base_args.seed))
    args = copy.copy(base_args)
    args.iterations = spec.iterations
    args.pretrain_episodes = spec.pretrain_episodes
    args.retention_eval_every = 0
    args.val_loss_every = 0
    resolved_overrides = apply_overrides(args, spec.arg_overrides)
    model = make_model(mod, args, device)
    model.train()
    memory = mod.EpisodicMemory(args.memory_size, args.model_dim, args.num_assemblies)
    optimizer = None
    named_params: list[tuple[str, nn.Parameter]] = []
    phase_key = ""
    events_path = output_dir / f"{spec.run_id}.events.jsonl"
    start = time.perf_counter()
    last_objective: dict[str, Any] = {}
    last_update_stats: dict[str, Any] = {}
    last_supervisor: dict[str, Any] | None = None
    last_support_before = float("nan")
    last_support_after = float("nan")
    last_query_ce = float("nan")

    for episode_idx in range(spec.iterations):
        shift, support_batches, query_batches, anchor_batch = generator.train_episode(episode_idx)
        phase = "pretrain" if episode_idx < spec.pretrain_episodes else "episodic"
        if spec.mode == "pure":
            phase = "pretrain"
        desired_key = "pretrain:all" if phase == "pretrain" else f"episodic:{spec.episodic_trainable}"
        if desired_key != phase_key:
            if phase == "pretrain":
                optimizer, named_params = build_optimizer_for_mode(mod, model, args, "pretrain", args.stable_lr, device)
            else:
                optimizer, named_params = build_optimizer_for_mode(mod, model, args, spec.episodic_trainable, args.outer_lr, device)
            phase_key = desired_key
        assert optimizer is not None
        named_params = trainable_named_params(model)
        before_params = clone_named_params(named_params)
        optimizer.zero_grad(set_to_none=True)

        if phase == "pretrain":
            with autocast_for(device):
                losses = [model(x, y) for x, y in support_batches + query_batches]
                loss = torch.stack(losses).mean()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([param for _, param in named_params], args.grad_clip_norm)
            optimizer.step()
            last_objective = {"phase": phase, "total_loss": float(loss.detach().float().item()), "query_ce": float(loss.detach().float().item())}
            last_support_before = last_support_after = float("nan")
            last_query_ce = float(loss.detach().float().item())
        else:
            temp_pool = mod.TempPool(args, device)
            support_traces, support_before, support_after, proxy_norm = run_support_adaptation(
                mod=mod,
                model=model,
                args=args,
                memory=memory,
                temp_pool=temp_pool,
                support_batches=support_batches,
                device=device,
                lm_only=True,
                train=True,
            )
            query_lm, query_traces, struct_loss, homeo_loss, l0_penalty, router_spec_loss = query_loss_and_traces(
                mod, model, args, memory, temp_pool, query_batches, device
            )
            anchor_ctx = mod.EpisodeContext(temp_pool=temp_pool, episodic_memory=memory, phase="anchor", allow_growth=False, enable_temp=True)
            with autocast_for(device):
                anchor_loss = model(anchor_batch[0], anchor_batch[1], episode_ctx=anchor_ctx)
                episodic_regularizer = (
                    args.lambda_struct * struct_loss
                    + args.lambda_homeo * homeo_loss
                    + args.lambda_anchor * anchor_loss
                    + args.lambda_l0 * l0_penalty
                    + args.lambda_router_spec * router_spec_loss
                )
                if spec.episodic_objective == "lm_dominant":
                    loss = query_lm + 0.05 * episodic_regularizer
                else:
                    loss = query_lm
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([param for _, param in named_params], args.grad_clip_norm)
            optimizer.step()
            model.update_query_ema(query_lm.detach())
            affinity_traces = [*support_traces, *query_traces]
            for trace in query_traces:
                memory.write(trace)
            decisions = model.consolidator.evaluate(model, temp_pool, support_batches + query_batches, [anchor_batch], memory, device)
            consolidation_counts = model.consolidator.apply(model, temp_pool, decisions)
            model.update_affinity_ema(affinity_traces)
            last_supervisor = supervisor_update(model, args, affinity_traces, query_lm)
            cadence_due = ((episode_idx - spec.pretrain_episodes + 1) % args.community_refresh_episodes == 0)
            if cadence_due and not (last_supervisor and last_supervisor.get("transitioned_to_grow")):
                model.refresh_communities()
            last_support_before = support_before
            last_support_after = support_after
            last_query_ce = float(query_lm.detach().float().item())
            last_objective = {
                "phase": phase,
                "total_loss": float(loss.detach().float().item()),
                "query_ce": last_query_ce,
                "support_ce_before": support_before,
                "support_ce_after": support_after,
                "support_adaptation_gain": support_before - support_after,
                "support_fast_proxy_norm": proxy_norm,
                "episodic_regularizer": float(episodic_regularizer.detach().float().item()),
                "consolidation_counts": consolidation_counts,
            }

        last_update_stats = bucket_stats(named_params, before_params)
        record = {
            "run_id": spec.run_id,
            "episode": episode_idx + 1,
            "shift": int(shift),
            "phase": phase,
            "objective": last_objective,
            "top_update_bucket": top_update_bucket(last_update_stats),
            "supervisor": last_supervisor,
        }
        write_jsonl(events_path, record)
        if train_log_every > 0 and ((episode_idx + 1) <= 3 or (episode_idx + 1) % train_log_every == 0 or (episode_idx + 1) == spec.iterations):
            print(
                f"{spec.run_id} episode={episode_idx + 1}/{spec.iterations} phase={phase} "
                f"query_ce={last_query_ce:.4f} support_before={last_support_before:.4f} support_after={last_support_after:.4f}"
            )

    sync(device)
    train_seconds = time.perf_counter() - start
    validation_no_adapt = evaluate_shift(
        mod=mod,
        model=model,
        args=args,
        generator=generator,
        device=device,
        eval_episodes=eval_episodes,
        adapt=False,
    )
    validation_adapt = evaluate_shift(
        mod=mod,
        model=model,
        args=args,
        generator=generator,
        device=device,
        eval_episodes=eval_episodes,
        adapt=True,
    )
    summary = {
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "resolved_arg_overrides": resolved_overrides,
        "train_seconds": train_seconds,
        "last_objective": last_objective,
        "last_update_stats": last_update_stats,
        "validation_no_adapt": validation_no_adapt,
        "validation_adapt": validation_adapt,
        "final_supervisor": {
            "mode": ("CLOSED", "GROW", "RECOVER")[int(model.supervisor_mode.item())],
            "mode_counts": [int(value) for value in model.supervisor_mode_counts.detach().cpu().tolist()],
            "growth_events": int(model.growth_event_count_buf.item()),
            "health_min": float(model.assembly_health.min().item()),
            "health_mean": float(model.assembly_health.mean().item()),
        },
        "top_update_bucket": top_update_bucket(last_update_stats),
    }
    write_json(output_dir / f"{spec.run_id}.summary.json", summary)
    print(
        f"RUN_SUMMARY {spec.run_id} train_seconds={train_seconds:.1f} "
        f"heldout_no_adapt_ce={validation_no_adapt['query_ce_no_adapt']:.4f} "
        f"heldout_after_support_ce={validation_adapt['query_ce_after_support']:.4f} "
        f"support_gain={validation_adapt['support_adaptation_gain']:.4f} "
        f"eval_mutations={validation_adapt['eval_mutations_after_restore']}"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return summary


def print_summary_table(summaries: list[dict[str, Any]]) -> None:
    print("")
    print("| run_id | heldout_query_no_adapt | heldout_query_after_support | support_gain | train_query_ce | modes | growth | top_update_bucket | eval_mut | seconds |")
    print("| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |")
    for summary in summaries:
        no_adapt = summary["validation_no_adapt"]
        adapt = summary["validation_adapt"]
        objective = summary["last_objective"]
        supervisor = summary["final_supervisor"]
        print(
            f"| {summary['run_id']} "
            f"| {no_adapt['query_ce_no_adapt']:.4f} "
            f"| {adapt['query_ce_after_support']:.4f} "
            f"| {adapt['support_adaptation_gain']:.4f} "
            f"| {float(objective.get('query_ce') or float('nan')):.4f} "
            f"| {supervisor['mode_counts']} "
            f"| {supervisor['growth_events']} "
            f"| {summary['top_update_bucket']} "
            f"| {adapt['eval_mutations_after_restore']} "
            f"| {summary['train_seconds']:.1f} |"
        )


def main() -> int:
    cli = parse_args()
    cli.repo_root = cli.repo_root.resolve()
    cli.w9_path = cli.w9_path.resolve()
    configure_env(cli)
    mod = import_w9_module(cli.w9_path)
    args = mod.Hyperparameters()
    args.seed = int(cli.seed)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    generator = ShiftEpisodeGenerator(
        args=args,
        device=device,
        seed=cli.seed,
        train_domains=cli.train_domains,
        heldout_domains=cli.heldout_domains,
        symbol_count=cli.symbol_count,
    )
    output_dir = cli.output_root / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{cli.matrix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"repo_root={cli.repo_root}")
    print(f"w9_path={cli.w9_path}")
    print(f"output_dir={output_dir}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
    print(f"train_shifts={generator.train_shifts}")
    print(f"heldout_shifts={generator.heldout_shifts}")
    summaries: list[dict[str, Any]] = []
    for spec in make_specs(cli.matrix, cli.single):
        summary = run_spec(
            mod=mod,
            base_args=args,
            spec=spec,
            generator=generator,
            device=device,
            output_dir=output_dir,
            train_log_every=cli.train_log_every,
            eval_episodes=cli.eval_episodes,
        )
        summaries.append(summary)
    write_json(output_dir / "matrix_summary.json", summaries)
    print_summary_table(summaries)
    print(f"MATRIX_SUMMARY_PATH {output_dir / 'matrix_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
