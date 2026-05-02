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
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


W9_RELATIVE_PATH = (
    "records/track_non_record_16mb/"
    "2026-05-01_psyarbor_lab_rebus_calculations_w9/train_gpt.py"
)
DEFAULT_REBUS_LAB_PATH = Path("C:/Users/jjor3/Dev/rebus-lab")
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
    episodic_objective: str = "full_aux"
    episodic_trainable: str = "outer_only"
    pretrain_trainable: str = "all"
    aux_zero: bool = False
    support_lm_only: bool = False
    full_validation: bool = False
    val_slice_seqs: int = 512
    config_diff: dict[str, Any] = field(default_factory=dict)
    arg_overrides: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Instrument W9 training path objective components, optimizer groups, "
            "parameter update distribution, validation modes, and REBUS-lab-inspired invariants."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--w9-path", type=Path, default=repo_root / W9_RELATIVE_PATH)
    parser.add_argument("--rebus-lab-path", type=Path, default=DEFAULT_REBUS_LAB_PATH)
    parser.add_argument(
        "--train-glob",
        default=str(repo_root / "data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"),
    )
    parser.add_argument(
        "--val-glob",
        default=str(repo_root / "data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=str(repo_root / "data/tokenizers/fineweb_1024_bpe.model"),
    )
    parser.add_argument(
        "--matrix",
        choices=(
            "dry1",
            "primary64-fullval",
            "extended32-sliceval",
            "schedule128-fullval",
            "lmfirst-fullval",
            "claude32-sliceval",
            "claude-fullval",
        ),
        default="dry1",
    )
    parser.add_argument("--single", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-slice-seqs", type=int, default=512)
    parser.add_argument("--eval-probe-seqs", type=int, default=32)
    parser.add_argument("--train-log-every", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=repo_root / "runs/w9_training_path")
    parser.add_argument("--skip-rebus-lab-oracle", action="store_true")
    parser.add_argument(
        "--reported-eval-source",
        choices=("snapshot_restore", "label_free_probe"),
        default="snapshot_restore",
        help="Metric source used for RUN_SUMMARY/table reporting. snapshot_restore runs W9 eval under buffer snapshot/restore.",
    )
    parser.add_argument("--promotion-baseline-bpb", type=float, default=3.4099)
    parser.add_argument(
        "--save-community-snapshots",
        action="store_true",
        help="Write final W9 community/affinity buffers for offline oracle comparison.",
    )
    return parser.parse_args()


def configure_env(cli: argparse.Namespace) -> None:
    updates = {
        "TRAIN_FILES_GLOB": str(cli.train_glob),
        "VAL_FILES_GLOB": str(cli.val_glob),
        "TOKENIZER_PATH": str(cli.tokenizer_path),
        "OUTPUT_DIR": str(cli.repo_root),
        "RUN_ID": "debug_w9_training_path",
        "RETENTION_EVAL_EVERY": "0",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "SEED": str(cli.seed),
    }
    for key, value in updates.items():
        os.environ[key] = value


def import_w9_module(w9_path: Path):
    if not w9_path.exists():
        raise FileNotFoundError(f"Missing W9 train_gpt.py: {w9_path}")
    module_name = f"w9_training_path_{abs(hash(str(w9_path.resolve()))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, w9_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import W9 module from {w9_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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
        torch.cuda.synchronize(device)


def autocast_for(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=(device.type == "cuda"),
    )


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


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
        bucket = param_bucket(name)
        entry = stats[bucket]
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


def clone_named_params(named_params: list[tuple[str, nn.Parameter]]) -> dict[str, Tensor]:
    return {name: param.detach().clone() for name, param in named_params}


def prefix_summary(names: list[str]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for name in names:
        prefix = name.split(".", 1)[0]
        summary[prefix] = summary.get(prefix, 0) + 1
    return dict(sorted(summary.items()))


def optimizer_summary(optimizer: torch.optim.Optimizer, named_params: list[tuple[str, nn.Parameter]]) -> list[dict[str, Any]]:
    name_by_id = {id(param): name for name, param in named_params}
    groups: list[dict[str, Any]] = []
    for group_idx, group in enumerate(optimizer.param_groups):
        group_names = [name_by_id.get(id(param), "<unnamed>") for param in group["params"]]
        groups.append(
            {
                "group_idx": group_idx,
                "lr": float(group.get("lr", 0.0)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_tensors": len(group["params"]),
                "parameter_count": int(sum(param.numel() for param in group["params"])),
                "module_prefixes": prefix_summary(group_names),
                "bucket_counts": {
                    bucket: int(sum(1 for name in group_names if param_bucket(name) == bucket))
                    for bucket in BUCKET_ORDER
                },
                "sample_names": group_names[:25],
            }
        )
    return groups


def make_model(mod, args, device: torch.device):
    model = mod.PsyArborLM(args).to(device)
    if device.type == "cuda":
        model = model.bfloat16()
        for module in model.modules():
            if isinstance(module, mod.CastedLinear):
                module.float()
        mod.restore_low_dim_params_to_fp32(model)
    return model


def trainability_matches(name: str, mode: str) -> bool:
    bucket = param_bucket(name)
    if mode == "all":
        return True
    if mode == "base_and_outer":
        return bucket in {
            "embedding_output",
            "core_sequence",
            "norm_positional",
            "router_controller",
            "adaptation_plasticity",
        }
    if mode == "outer_only":
        return bucket in {"router_controller", "adaptation_plasticity"}
    if mode == "base_lm_only":
        return bucket in {"embedding_output", "core_sequence", "norm_positional"}
    if mode == "freeze_router_plasticity":
        return bucket not in {"router_controller", "adaptation_plasticity", "supervisor_state"}
    raise ValueError(f"Unsupported trainability mode: {mode}")


def set_trainability(model: nn.Module, mode: str) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = trainability_matches(name, mode)
    setattr(model, "training_phase", mode)


def trainable_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def build_optimizer(mod, args, named_params: list[tuple[str, nn.Parameter]], lr: float, device: torch.device):
    return mod.build_optimizer([param for _, param in named_params], args, lr, device)


def apply_aux_zero(args) -> dict[str, float]:
    keys = [
        "lambda_prox",
        "lambda_struct",
        "lambda_homeo",
        "lambda_anchor",
        "lambda_l0",
        "lambda_router_spec",
    ]
    old = {key: float(getattr(args, key)) for key in keys}
    for key in keys:
        setattr(args, key, 0.0)
    return old


def set_single_aux_zero(args, key: str) -> dict[str, float]:
    old = {key: float(getattr(args, key))}
    setattr(args, key, 0.0)
    return old


def resolve_arg_override(args, value: Any) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        attr = value[1:]
        if not hasattr(args, attr):
            raise ValueError(f"Unknown arg reference override: {value}")
        return getattr(args, attr)
    return value


def apply_arg_overrides(args, overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    original: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"RunSpec override references unknown Hyperparameters field: {key}")
        original[key] = getattr(args, key)
        resolved_value = resolve_arg_override(args, value)
        setattr(args, key, resolved_value)
        resolved[key] = resolved_value
    return original, resolved


def support_total(trace, temp_pool, args, lm_only: bool) -> Tensor:
    if trace.lm_loss is None:
        raise RuntimeError("support trace has no LM loss")
    if lm_only:
        return trace.lm_loss.float()
    return mod_compute_support_total(trace, temp_pool, args)


def mod_compute_support_total(trace, temp_pool, args) -> Tensor:
    total = (
        trace.lm_loss
        + args.lambda_prox * temp_pool.prox_loss()
        + args.lambda_struct * trace.struct_loss
        + args.lambda_homeo * trace.homeo_loss
        + args.lambda_l0 * trace.l0_penalty
    )
    return total.float() if args.support_total_fp32 else total


def objective_component_record(
    *,
    phase: str,
    total_loss: Tensor,
    lm_ce: Tensor,
    support_before: float | None = None,
    support_after: float | None = None,
    support_gain: float | None = None,
    anchor_loss: Tensor | None = None,
    struct_loss: Tensor | None = None,
    homeo_loss: Tensor | None = None,
    l0_penalty: Tensor | None = None,
    router_spec_loss: Tensor | None = None,
    episodic_regularizer: Tensor | None = None,
    lm_weight: float = 1.0,
    episodic_weight: float = 0.0,
    args: Any | None = None,
) -> dict[str, Any]:
    def scalar(value: Tensor | float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, Tensor):
            return float(value.detach().float().item())
        return float(value)

    components: dict[str, Any] = {
        "phase": phase,
        "total_loss": scalar(total_loss),
        "lm_ce": scalar(lm_ce),
        "support_total_before_mean": support_before,
        "support_total_after_mean": support_after,
        "support_gain_mean": support_gain,
        "anchor_loss": scalar(anchor_loss),
        "struct_loss": scalar(struct_loss),
        "homeo_loss": scalar(homeo_loss),
        "l0_penalty": scalar(l0_penalty),
        "router_spec_loss": scalar(router_spec_loss),
        "episodic_regularizer": scalar(episodic_regularizer),
        "lm_weight": float(lm_weight),
        "episodic_weight": float(episodic_weight),
    }
    if args is not None:
        components["coefficients"] = {
            "lambda_prox": float(args.lambda_prox),
            "lambda_struct": float(args.lambda_struct),
            "lambda_homeo": float(args.lambda_homeo),
            "lambda_anchor": float(args.lambda_anchor),
            "lambda_l0": float(args.lambda_l0),
            "lambda_router_spec": float(args.lambda_router_spec),
        }
        components["weighted_terms"] = {
            "lm": float(lm_weight) * float(scalar(lm_ce) or 0.0),
            "episodic_regularizer": float(episodic_weight) * float(scalar(episodic_regularizer) or 0.0),
            "anchor": float(args.lambda_anchor) * float(scalar(anchor_loss) or 0.0),
            "struct": float(args.lambda_struct) * float(scalar(struct_loss) or 0.0),
            "homeo": float(args.lambda_homeo) * float(scalar(homeo_loss) or 0.0),
            "l0": float(args.lambda_l0) * float(scalar(l0_penalty) or 0.0),
            "router_spec": float(args.lambda_router_spec) * float(scalar(router_spec_loss) or 0.0),
        }
    return components


def batch_bpb(
    mod,
    x: Tensor,
    y: Tensor,
    loss: Tensor,
    base_bytes_lut: Tensor,
    has_space_lut: Tensor,
    boundary_lut: Tensor,
) -> float:
    tokens, bytes_count = mod.batch_bpb_components(x, y, base_bytes_lut, has_space_lut, boundary_lut)
    return float(loss.detach().float().item()) / math.log(2.0) * (float(tokens) / max(float(bytes_count), 1.0))


def snapshot_buffers(model: nn.Module) -> dict[str, Tensor]:
    include_substrings = (
        "community_route_ema",
        "assembly_affinity_ema",
        "route_entropy",
        "calibration_",
        "h_grow_threshold",
        "h_recover_threshold",
        "sparse_active",
        "route_probs_buffer",
        "dispersion_entropy_last",
        "assembly_health",
        "per_assembly_error_ema",
        "supervisor_mode",
        "growth_event_count_buf",
        "activity_ema",
        "branch_load",
        "structural_stress",
        "recent_utility",
    )
    out: dict[str, Tensor] = {}
    for name, buffer in model.named_buffers():
        if any(token in name for token in include_substrings):
            out[name] = buffer.detach().clone()
    return out


def restore_buffers(model: nn.Module, snapshot: dict[str, Tensor]) -> None:
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, value in snapshot.items():
            if name in buffers and buffers[name].shape == value.shape:
                buffers[name].copy_(value.to(device=buffers[name].device, dtype=buffers[name].dtype))


def diff_snapshots(before: dict[str, Tensor], after: dict[str, Tensor]) -> dict[str, Any]:
    changed: dict[str, float] = {}
    for name, old in before.items():
        new = after.get(name)
        if new is None or new.shape != old.shape:
            changed[name] = float("inf")
            continue
        if old.dtype == torch.bool:
            delta = float((old.cpu() != new.cpu()).sum().item())
        else:
            delta = float((old.float().cpu() - new.float().cpu()).abs().max().item())
        if delta != 0.0:
            changed[name] = delta
    return {
        "changed_count": len(changed),
        "max_abs_delta": max(changed.values()) if changed else 0.0,
        "changed": dict(sorted(changed.items())[:50]),
    }


def truncate_val_tokens(mod, args, val_tokens: Tensor, max_seqs: int) -> Tensor:
    if max_seqs <= 0:
        return val_tokens
    usable_tokens = max_seqs * int(args.train_seq_len) + 1
    return val_tokens[: min(int(val_tokens.numel()), usable_tokens)].contiguous()


def eval_current(
    mod,
    args,
    model: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_space_lut: Tensor,
    boundary_lut: Tensor,
) -> tuple[float, float, dict[str, Any]]:
    before = snapshot_buffers(model)
    loss, bpb = mod.eval_val(args, model, 0, 1, device, val_tokens, base_bytes_lut, has_space_lut, boundary_lut)
    after = snapshot_buffers(model)
    diff = diff_snapshots(before, after)
    restore_buffers(model, before)
    return loss, bpb, diff


def eval_label_free_external(
    mod,
    args,
    model: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_space_lut: Tensor,
    boundary_lut: Tensor,
) -> tuple[float, float, dict[str, Any]]:
    local_batch_tokens = min(int(args.val_batch_size), max(int(args.train_seq_len), int(val_tokens.numel()) - 1))
    local_batch_seqs = max(1, local_batch_tokens // int(args.train_seq_len))
    total_seqs = (int(val_tokens.numel()) - 1) // int(args.train_seq_len)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    before = snapshot_buffers(model)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(0, total_seqs, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, total_seqs)
            raw_start = batch_seq_start * int(args.train_seq_len)
            raw_end = batch_seq_end * int(args.train_seq_len) + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, int(args.train_seq_len))
            y = local[1:].reshape(-1, int(args.train_seq_len))
            with autocast_for(device):
                logits = model(x, None)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
            batch_tokens = float(y.numel())
            loss_sum += loss.to(torch.float64) * batch_tokens
            token_count += batch_tokens
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += token_bytes.to(torch.float64).sum()
    after = snapshot_buffers(model)
    diff = diff_snapshots(before, after)
    restore_buffers(model, before)
    val_loss = loss_sum / token_count
    bpb = (float(val_loss.item()) / math.log(2.0)) * (float(token_count.item()) / max(float(byte_count.item()), 1.0))
    model.train()
    return float(val_loss.item()), float(bpb), diff


def validation_report(
    mod,
    args,
    model: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
    probe_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_space_lut: Tensor,
    boundary_lut: Tensor,
    full_validation: bool,
    reported_eval_source: str,
    promotion_baseline_bpb: float,
) -> dict[str, Any]:
    current_tokens = val_tokens if full_validation else probe_tokens
    current_loss, current_bpb, current_mutation = eval_current(
        mod, args, model, device, current_tokens, base_bytes_lut, has_space_lut, boundary_lut
    )
    label_free_loss, label_free_bpb, label_free_mutation = eval_label_free_external(
        mod, args, model, device, probe_tokens, base_bytes_lut, has_space_lut, boundary_lut
    )
    restored_loss, restored_bpb, restored_mutation = eval_current(
        mod, args, model, device, probe_tokens, base_bytes_lut, has_space_lut, boundary_lut
    )
    if reported_eval_source == "snapshot_restore":
        reported_loss = current_loss
        reported_bpb = current_bpb
        reported_scope = "full" if full_validation else "slice"
        reported_mutation = current_mutation
    elif reported_eval_source == "label_free_probe":
        reported_loss = label_free_loss
        reported_bpb = label_free_bpb
        reported_scope = "probe"
        reported_mutation = label_free_mutation
    else:
        raise ValueError(f"Unsupported reported_eval_source: {reported_eval_source}")
    return {
        "reported_eval_source": reported_eval_source,
        "reported_eval_scope": reported_scope,
        "reported_eval_loss": reported_loss,
        "reported_eval_bpb": reported_bpb,
        "reported_eval_buffer_mutation": reported_mutation,
        "promotion_baseline_bpb": float(promotion_baseline_bpb),
        "beats_promotion_baseline": bool(reported_bpb < float(promotion_baseline_bpb)),
        "current_eval_scope": "full" if full_validation else "slice",
        "current_eval_loss": current_loss,
        "current_eval_bpb": current_bpb,
        "current_eval_buffer_mutation": current_mutation,
        "label_free_probe_loss": label_free_loss,
        "label_free_probe_bpb": label_free_bpb,
        "label_free_probe_buffer_mutation": label_free_mutation,
        "snapshot_restored_probe_loss": restored_loss,
        "snapshot_restored_probe_bpb": restored_bpb,
        "snapshot_restored_probe_buffer_mutation": restored_mutation,
    }


def community_members_from_assignment(assignment: np.ndarray, community_count: int) -> list[list[int]]:
    return [
        [int(idx) for idx in np.flatnonzero(assignment == community_id).tolist()]
        for community_id in range(int(community_count))
    ]


def community_metrics(affinity: np.ndarray, assignment: np.ndarray, route_ema: np.ndarray | None = None) -> dict[str, Any]:
    dense = np.asarray(affinity, dtype=np.float64)
    dense = 0.5 * (dense + dense.T)
    np.fill_diagonal(dense, 0.0)
    assignment = np.asarray(assignment, dtype=np.int64)
    community_count = int(assignment.max()) + 1 if assignment.size else 0
    communities = community_members_from_assignment(assignment, community_count)
    n = int(dense.shape[0])
    offdiag_mask = ~np.eye(n, dtype=bool) if n > 0 else np.zeros((0, 0), dtype=bool)
    offdiag_values = dense[offdiag_mask] if n > 1 else np.array([], dtype=np.float64)
    baseline = float(offdiag_values.mean()) if offdiag_values.size else 0.0

    within_values: list[float] = []
    cross_values: list[float] = []
    within_sum = 0.0
    cross_sum = 0.0
    for left in range(n):
        for right in range(left + 1, n):
            value = float(dense[left, right])
            if int(assignment[left]) == int(assignment[right]):
                within_values.append(value)
                within_sum += value
            else:
                cross_values.append(value)
                cross_sum += value
    within_mean = float(np.mean(within_values)) if within_values else 0.0
    cross_mean = float(np.mean(cross_values)) if cross_values else 0.0
    total_pair_weight = within_sum + cross_sum
    metrics: dict[str, Any] = {
        "community_count": community_count,
        "community_sizes": [len(members) for members in communities],
        "size_balance": {
            "min": min((len(members) for members in communities), default=0),
            "max": max((len(members) for members in communities), default=0),
            "std": float(np.std([len(members) for members in communities])) if communities else 0.0,
        },
        "offdiag_affinity_mean": baseline,
        "within_affinity_mean": within_mean,
        "cross_affinity_mean": cross_mean,
        "density_lift": within_mean - baseline,
        "within_affinity_fraction": within_sum / max(total_pair_weight, 1e-12),
        "cross_affinity_fraction": cross_sum / max(total_pair_weight, 1e-12),
        "within_pair_count": len(within_values),
        "cross_pair_count": len(cross_values),
    }
    if route_ema is not None and route_ema.size:
        route = np.asarray(route_ema, dtype=np.float64)
        row_sums = np.maximum(route.sum(axis=1, keepdims=True), 1e-12)
        route = route / row_sums
        own_masses: list[float] = []
        best_other_masses: list[float] = []
        for community_id, members in enumerate(communities):
            if community_id >= route.shape[0] or not members:
                continue
            own = float(route[community_id, members].sum())
            other_masses = [
                float(route[community_id, other_members].sum())
                for other_id, other_members in enumerate(communities)
                if other_id != community_id and other_members
            ]
            own_masses.append(own)
            best_other_masses.append(max(other_masses) if other_masses else 0.0)
        metrics["route"] = {
            "own_mass_mean": float(np.mean(own_masses)) if own_masses else 0.0,
            "cross_community_route_rate": 1.0 - (float(np.mean(own_masses)) if own_masses else 0.0),
            "community_route_separation": float(np.mean(np.asarray(own_masses) - np.asarray(best_other_masses)))
            if own_masses
            else 0.0,
        }
    return metrics


def build_community_snapshot(
    *,
    args,
    model: nn.Module,
    spec: RunSpec,
    validation: dict[str, Any],
) -> dict[str, Any]:
    affinity = model.assembly_affinity_ema.detach().float().cpu().numpy().astype(np.float64, copy=False)
    assignment = model.community_assignment.detach().cpu().numpy().astype(np.int64, copy=False)
    route_ema = model.community_route_ema.detach().float().cpu().numpy().astype(np.float64, copy=False)
    metrics = community_metrics(affinity, assignment, route_ema)
    return {
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "validation": {
            "reported_eval_source": validation["reported_eval_source"],
            "reported_eval_scope": validation["reported_eval_scope"],
            "reported_eval_loss": validation["reported_eval_loss"],
            "reported_eval_bpb": validation["reported_eval_bpb"],
            "current_eval_scope": validation["current_eval_scope"],
            "current_eval_loss": validation["current_eval_loss"],
            "current_eval_bpb": validation["current_eval_bpb"],
            "label_free_probe_bpb": validation["label_free_probe_bpb"],
        },
        "args": {
            "num_assemblies": int(args.num_assemblies),
            "community_count": int(args.community_count),
            "community_target_size": int(args.community_target_size),
            "community_tau": float(args.community_tau),
            "community_greedy_rounds": int(args.community_greedy_rounds),
        },
        "community_assignment": assignment.tolist(),
        "community_members": community_members_from_assignment(assignment, int(args.community_count)),
        "assembly_affinity_ema": affinity.tolist(),
        "community_route_ema": route_ema.tolist(),
        "metrics": metrics,
    }


def probs_with_target_entropy(n: int, target_entropy: float, seed: int = 0) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.randn(n, generator=gen)
    lo, hi = 0.05, 50.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        probs = torch.softmax(scores / mid, dim=-1).clamp_min(1e-9)
        entropy = float(-(probs * probs.log()).sum().item())
        if entropy < target_entropy:
            lo = mid
        else:
            hi = mid
    return torch.softmax(scores / hi, dim=-1)


def run_w9_invariants(mod, args, device: torch.device) -> dict[str, Any]:
    results: dict[str, Any] = {}
    inv_args = copy.copy(args)
    inv_args.rebus_supervisor_enabled = True
    inv_args.rebus_use_dispersion_entropy = False
    inv_args.rebus_use_online_quantiles = False
    inv_args.rebus_use_huber_attribution = False
    inv_args.rebus_grow_direction = "high"
    inv_args.rebus_h_grow_quantile = 0.66
    inv_args.rebus_h_recover_quantile = 0.34
    inv_args.rebus_calibration_episodes = 3
    inv_args.rebus_health_decay = 0.5
    inv_args.rebus_health_refresh = 0.05
    inv_args.rebus_health_prune_threshold = 0.25
    inv_args.rebus_growth_budget = 1
    model = make_model(mod, inv_args, device)

    gated = torch.full((inv_args.num_assemblies,), -1e9, device=device)
    gated[: inv_args.community_target_size] = torch.linspace(0.0, 1.0, inv_args.community_target_size, device=device)
    sparse_mask, sparse_weights = model._sparse_route_weights(gated)
    active = torch.nonzero(sparse_mask > 0, as_tuple=False).reshape(-1).detach().cpu().tolist()
    results["sparse_route"] = {
        "ok": int((sparse_mask > 0).sum().item()) == min(inv_args.rebus_k_sparse, inv_args.community_target_size)
        and abs(float(sparse_weights.sum().item()) - 1.0) < 1e-5
        and all(int(idx) < inv_args.community_target_size for idx in active),
        "active_indices": [int(idx) for idx in active],
        "weight_sum": float(sparse_weights.sum().item()),
    }

    model.assembly_health.fill_(1.0)
    inactive = torch.zeros(inv_args.num_assemblies, device=device, dtype=torch.float32)
    expected_steps = int(math.ceil(math.log(inv_args.rebus_health_prune_threshold) / math.log(inv_args.rebus_health_decay)))
    for _ in range(expected_steps + 2):
        model._update_assembly_health(inactive)
    below = float(model.assembly_health.min().item()) < float(inv_args.rebus_health_prune_threshold)
    active_mask = torch.zeros_like(inactive)
    active_mask[0] = 1.0
    before_refresh = float(model.assembly_health[0].item())
    model._update_assembly_health(active_mask)
    results["health_decay_recovery"] = {
        "ok": below and float(model.assembly_health[0].item()) > before_refresh,
        "min_after_decay": float(model.assembly_health.min().item()),
        "assembly0_before_refresh": before_refresh,
        "assembly0_after_refresh": float(model.assembly_health[0].item()),
    }

    model.calibration_entropy_buf.copy_(torch.tensor([0.5, 1.5, 1.0], dtype=torch.float32, device=device))
    model.calibration_filled.fill_(True)
    model.calibration_cursor.fill_(int(model.calibration_entropy_buf.numel()))
    model._freeze_thresholds()
    h_grow = float(model.h_grow_threshold.item())
    h_recover = float(model.h_recover_threshold.item())
    low_probs = probs_with_target_entropy(inv_args.num_assemblies, max(h_recover - 0.05, 0.05)).to(device)
    high_probs = probs_with_target_entropy(inv_args.num_assemblies, h_grow + 0.1).to(device)
    model.train()
    full_active = torch.ones(inv_args.num_assemblies, device=device, dtype=torch.float32)
    modes = [
        model.supervisor_step(low_probs, 1.0, full_active)["mode"],
        model.supervisor_step(high_probs, 1.0, full_active)["mode"],
        model.supervisor_step(low_probs, 1.0, full_active)["mode"],
        model.supervisor_step(low_probs, 1.0, full_active)["mode"],
    ]
    results["mode_transitions"] = {"ok": modes == ["CLOSED", "GROW", "RECOVER", "CLOSED"], "modes": modes}

    model.eval()
    model.supervisor_mode.fill_(1)
    model.assembly_health.fill_(1.0)
    model.assembly_health[0] = inv_args.rebus_health_prune_threshold * 0.5
    eval_growth = model._maybe_clone_and_perturb()
    model.train()
    model.growth_event_count_buf.fill_(inv_args.rebus_growth_budget)
    budget_growth = model._maybe_clone_and_perturb()
    results["clone_blocking"] = {"ok": eval_growth is None and budget_growth is None}

    # W9 has no production reset method; validate the diagnostic snapshot/restore reset semantics.
    before = snapshot_buffers(model)
    model.supervisor_mode.fill_(2)
    model.assembly_health.mul_(0.1)
    restore_buffers(model, before)
    after = snapshot_buffers(model)
    results["snapshot_restore"] = {"ok": diff_snapshots(before, after)["changed_count"] == 0}
    return results


def run_rebus_lab_oracle(rebus_lab_path: Path) -> dict[str, Any]:
    src_path = rebus_lab_path / "python" / "src"
    if not src_path.exists():
        return {"available": False, "reason": f"missing {src_path}"}
    sys.path.insert(0, str(src_path))
    try:
        from rebus import identification as rid
        from rebus import supervisor as rs
        from rebus.synthetic import make_synthetic_supervisor_episode
    except Exception as exc:
        return {"available": False, "reason": f"import failed: {exc}"}

    out: dict[str, Any] = {"available": True}
    x_t = np.array([[1.0]], dtype=float)
    u_t = np.array([[0.0]], dtype=float)
    x_tp1 = np.array([[0.0]], dtype=float)
    A = np.stack([np.array([[float(i)]], dtype=float) for i in range(4)], axis=2)
    B = np.zeros((1, 1, 4), dtype=float)
    d = np.zeros((1, 4), dtype=float)
    weights = rs.sparse_routing_weights(A, B, d, x_t, u_t, x_tp1, k_sparse=2, tau=1.0)
    out["sparse_routing"] = {
        "ok": weights.shape == (4,) and int(np.count_nonzero(weights)) == 2 and abs(float(np.sum(weights)) - 1.0) < 1e-12,
        "weights": weights,
    }

    episode, truth = make_synthetic_supervisor_episode()
    bounds = rid.RebusBounds(
        a_lb=0.2,
        P=np.array([[0.65, 0.05], [0.05, 0.45]], dtype=float),
        lambda_lb=0.25,
        mu_lb=0.30,
        kappa_ub=0.35,
        c_alpha_ub=0.40,
        c_nu_ub=0.45,
        b0_ub=0.80,
        b1_ub=0.60,
        b2_ub=0.50,
        b3_ub=0.55,
    )
    offsets = (-0.12, -0.03, 0.04, 0.11)
    a_base = np.array([[0.88, 0.04], [0.03, 0.82]], dtype=float)
    b_base = np.array([[0.16, 0.05, 0.08], [0.05, 0.11, 0.06]], dtype=float)
    d_base = np.array([[0.01], [-0.01]], dtype=float)
    bootstrap = rid.BootstrapScenarios(
        A_scenarios=np.stack([a_base + off * np.array([[0.08, 0.01], [0.01, 0.07]], dtype=float) for off in offsets], axis=2),
        B_scenarios=np.stack([b_base + off * np.array([[0.03, 0.02, 0.01], [0.01, 0.03, 0.02]], dtype=float) for off in offsets], axis=2),
        d_scenarios=np.column_stack([(d_base + off * np.array([[0.005], [0.004]], dtype=float)).reshape(-1) for off in offsets]),
    )
    supervisor = rs.RebusOnlineSupervisor(
        bounds=bounds,
        bootstrap_scenarios=bootstrap,
        X_ref=episode["X_ref"],
        U_ref=episode["U_ref"],
        config=rs.EntropyTriggerConfig(k_sparse=2, transient_reset_floor=0.6),
        rng=np.random.default_rng(0),
    )
    for idx in range(truth.closed_reentry):
        supervisor.step(episode["X_t"][:, [idx]], episode["U_t"][:, [idx]], episode["X_tp1"][:, [idx]])
    supervisor.reset()
    out["reset_contract"] = {
        "ok": supervisor.mode == rs.SupervisorMode.CLOSED
        and np.all(supervisor.scenario_active_mask)
        and np.all(supervisor.scenario_health >= 0.6)
        and abs(float(np.sum(supervisor.routing_weights)) - 1.0) < 1e-12,
        "health_min": float(np.min(supervisor.scenario_health)),
        "growth_event_count": int(supervisor.growth_event_count),
    }
    return out


def make_specs(matrix: str, single: str, val_slice_seqs: int) -> list[RunSpec]:
    specs: dict[str, RunSpec] = {
        "dry1": RunSpec(
            run_id="dry1_pretrain",
            iterations=1,
            pretrain_episodes=1,
            full_validation=False,
            val_slice_seqs=min(val_slice_seqs, 32),
            config_diff={"purpose": "one-step diagnostics dry run"},
        ),
        "pure_pretrain64": RunSpec(
            run_id="pure_pretrain64",
            iterations=64,
            pretrain_episodes=64,
            full_validation=True,
            config_diff={"pretrain_episodes": 64, "iterations": 64},
        ),
        "pretrain64_auxzero": RunSpec(
            run_id="pretrain64_auxzero",
            iterations=64,
            pretrain_episodes=64,
            aux_zero=True,
            full_validation=True,
            config_diff={"pretrain_episodes": 64, "iterations": 64, "aux_coefficients": 0.0},
        ),
        "episodic64_lm_only_outer": RunSpec(
            run_id="episodic64_lm_only_outer",
            iterations=64,
            pretrain_episodes=16,
            episodic_objective="lm_only",
            support_lm_only=True,
            episodic_trainable="outer_only",
            full_validation=True,
            config_diff={"pretrain_episodes": 16, "episodic_steps": 48, "outer_loss": "query_lm_loss", "episodic_trainable": "outer_only"},
        ),
        "episodic64_full_aux_outer": RunSpec(
            run_id="episodic64_full_aux_outer",
            iterations=64,
            pretrain_episodes=16,
            episodic_objective="full_aux",
            episodic_trainable="outer_only",
            full_validation=True,
            config_diff={"pretrain_episodes": 16, "episodic_steps": 48, "outer_loss": "configured_full_aux", "episodic_trainable": "outer_only"},
        ),
        "episodic64_lm_only_base": RunSpec(
            run_id="episodic64_lm_only_base",
            iterations=64,
            pretrain_episodes=16,
            episodic_objective="lm_only",
            support_lm_only=True,
            episodic_trainable="base_lm_only",
            full_validation=True,
            config_diff={"pretrain_episodes": 16, "episodic_steps": 48, "outer_loss": "query_lm_loss", "episodic_trainable": "base_lm_only"},
        ),
        "episodic64_full_aux_base": RunSpec(
            run_id="episodic64_full_aux_base",
            iterations=64,
            pretrain_episodes=16,
            episodic_objective="full_aux",
            episodic_trainable="base_lm_only",
            full_validation=True,
            config_diff={"pretrain_episodes": 16, "episodic_steps": 48, "outer_loss": "configured_full_aux", "episodic_trainable": "base_lm_only"},
        ),
        "pure_pretrain128": RunSpec(
            run_id="pure_pretrain128",
            iterations=128,
            pretrain_episodes=128,
            full_validation=True,
            config_diff={"pretrain_episodes": 128, "iterations": 128},
        ),
        "episodic128_lm_only_base": RunSpec(
            run_id="episodic128_lm_only_base",
            iterations=128,
            pretrain_episodes=16,
            episodic_objective="lm_only",
            support_lm_only=True,
            episodic_trainable="base_lm_only",
            full_validation=True,
            config_diff={"pretrain_episodes": 16, "episodic_steps": 112, "outer_loss": "query_lm_loss", "episodic_trainable": "base_lm_only"},
        ),
        "episodic128_lm_dominant_base_outer": RunSpec(
            run_id="episodic128_lm_dominant_base_outer",
            iterations=128,
            pretrain_episodes=16,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=True,
            config_diff={
                "pretrain_episodes": 16,
                "episodic_steps": 112,
                "outer_loss": "query_lm_loss_plus_0.05_aux",
                "episodic_trainable": "base_and_outer",
            },
        ),
        "episodic128_full_aux_base_outer": RunSpec(
            run_id="episodic128_full_aux_base_outer",
            iterations=128,
            pretrain_episodes=16,
            episodic_objective="full_aux",
            support_lm_only=False,
            episodic_trainable="base_and_outer",
            full_validation=True,
            config_diff={"pretrain_episodes": 16, "episodic_steps": 112, "outer_loss": "configured_full_aux", "episodic_trainable": "base_and_outer"},
        ),
        "pure64_then_episodic64_lm_dominant_base_outer": RunSpec(
            run_id="pure64_then_episodic64_lm_dominant_base_outer",
            iterations=128,
            pretrain_episodes=64,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=True,
            config_diff={
                "pretrain_episodes": 64,
                "episodic_steps": 64,
                "outer_loss": "query_lm_loss_plus_0.05_aux",
                "episodic_trainable": "base_and_outer",
            },
        ),
        "pure_pretrain256": RunSpec(
            run_id="pure_pretrain256",
            iterations=256,
            pretrain_episodes=256,
            full_validation=True,
            config_diff={"pretrain_episodes": 256, "iterations": 256},
        ),
        "pure_pretrain384": RunSpec(
            run_id="pure_pretrain384",
            iterations=384,
            pretrain_episodes=384,
            full_validation=True,
            config_diff={"pretrain_episodes": 384, "iterations": 384},
        ),
        "pure_pretrain512": RunSpec(
            run_id="pure_pretrain512",
            iterations=512,
            pretrain_episodes=512,
            full_validation=True,
            config_diff={"pretrain_episodes": 512, "iterations": 512},
        ),
        "pure_pretrain256_then_lm_dominant_episodic128": RunSpec(
            run_id="pure_pretrain256_then_lm_dominant_episodic128",
            iterations=384,
            pretrain_episodes=256,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=True,
            config_diff={
                "pretrain_episodes": 256,
                "episodic_steps": 128,
                "outer_loss": "query_lm_loss_plus_0.05_aux",
                "episodic_trainable": "base_and_outer",
            },
        ),
        "pure_pretrain256_then_lm_only_ep128_lrmatch": RunSpec(
            run_id="pure_pretrain256_then_lm_only_ep128_lrmatch",
            iterations=384,
            pretrain_episodes=256,
            episodic_objective="lm_only",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=True,
            config_diff={
                "pretrain_episodes": 256,
                "episodic_steps": 128,
                "outer_loss": "query_lm_loss",
                "episodic_trainable": "base_and_outer",
                "outer_lr": "$stable_lr",
            },
            arg_overrides={"outer_lr": "$stable_lr"},
        ),
        "pure_pretrain256_then_lm_dominant_ep128_anchor0_lrmatch": RunSpec(
            run_id="pure_pretrain256_then_lm_dominant_ep128_anchor0_lrmatch",
            iterations=384,
            pretrain_episodes=256,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=True,
            config_diff={
                "pretrain_episodes": 256,
                "episodic_steps": 128,
                "outer_loss": "query_lm_loss_plus_0.05_aux",
                "episodic_trainable": "base_and_outer",
                "lambda_anchor": 0.0,
                "outer_lr": "$stable_lr",
            },
            arg_overrides={"lambda_anchor": 0.0, "outer_lr": "$stable_lr"},
        ),
        "pure_pretrain256_then_full_aux_episodic128": RunSpec(
            run_id="pure_pretrain256_then_full_aux_episodic128",
            iterations=384,
            pretrain_episodes=256,
            episodic_objective="full_aux",
            support_lm_only=False,
            episodic_trainable="base_and_outer",
            full_validation=True,
            config_diff={
                "pretrain_episodes": 256,
                "episodic_steps": 128,
                "outer_loss": "configured_full_aux",
                "episodic_trainable": "base_and_outer",
            },
        ),
    }
    extended_names = []
    for run_id, trainable in [
        ("ext32_pretrain_all", "all"),
        ("ext32_pretrain_auxzero", "all"),
        ("ext32_pretrain_router_plasticity_frozen", "freeze_router_plasticity"),
        ("ext32_pretrain_base_lm_only", "base_lm_only"),
    ]:
        specs[run_id] = RunSpec(
            run_id=run_id,
            iterations=32,
            pretrain_episodes=32,
            pretrain_trainable=trainable,
            aux_zero=("auxzero" in run_id),
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "extended32", "pretrain_trainable": trainable},
        )
        extended_names.append(run_id)

    claude32_names = []
    claude32_specs = [
        RunSpec(
            run_id="claude32_pure_pretrain",
            iterations=32,
            pretrain_episodes=32,
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "claude32", "pretrain_episodes": 32, "iterations": 32},
        ),
        RunSpec(
            run_id="claude32_lm_dominant_default",
            iterations=32,
            pretrain_episodes=8,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "claude32", "pretrain_episodes": 8, "episodic_steps": 24, "outer_loss": "query_lm_loss_plus_0.05_aux"},
        ),
        RunSpec(
            run_id="claude32_lm_only_lrmatch",
            iterations=32,
            pretrain_episodes=8,
            episodic_objective="lm_only",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "claude32", "outer_loss": "query_lm_loss", "outer_lr": "$stable_lr"},
            arg_overrides={"outer_lr": "$stable_lr"},
        ),
        RunSpec(
            run_id="claude32_anchor0",
            iterations=32,
            pretrain_episodes=8,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "claude32", "lambda_anchor": 0.0},
            arg_overrides={"lambda_anchor": 0.0},
        ),
        RunSpec(
            run_id="claude32_lrmatch",
            iterations=32,
            pretrain_episodes=8,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "claude32", "outer_lr": "$stable_lr"},
            arg_overrides={"outer_lr": "$stable_lr"},
        ),
        RunSpec(
            run_id="claude32_anchor0_lrmatch",
            iterations=32,
            pretrain_episodes=8,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "claude32", "lambda_anchor": 0.0, "outer_lr": "$stable_lr"},
            arg_overrides={"lambda_anchor": 0.0, "outer_lr": "$stable_lr"},
        ),
        RunSpec(
            run_id="claude32_huber_wide",
            iterations=32,
            pretrain_episodes=8,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "claude32", "rebus_huber_m": 8.0},
            arg_overrides={"rebus_huber_m": 8.0},
        ),
        RunSpec(
            run_id="claude32_huber_off",
            iterations=32,
            pretrain_episodes=8,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "claude32", "rebus_use_huber_attribution": False},
            arg_overrides={"rebus_use_huber_attribution": False},
        ),
        RunSpec(
            run_id="claude32_supervisor_off",
            iterations=32,
            pretrain_episodes=8,
            episodic_objective="lm_dominant",
            support_lm_only=True,
            episodic_trainable="base_and_outer",
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "claude32", "rebus_supervisor_enabled": False},
            arg_overrides={"rebus_supervisor_enabled": False},
        ),
    ]
    for spec in claude32_specs:
        specs[spec.run_id] = spec
        claude32_names.append(spec.run_id)
    for run_id, objective, trainable in [
        ("ext32_episodic_lm_only_outer", "lm_only", "outer_only"),
        ("ext32_episodic_full_aux_outer", "full_aux", "outer_only"),
        ("ext32_episodic_base_frozen_outer", "full_aux", "outer_only"),
        ("ext32_episodic_outer_frozen_base", "full_aux", "base_lm_only"),
    ]:
        specs[run_id] = RunSpec(
            run_id=run_id,
            iterations=32,
            pretrain_episodes=8,
            episodic_objective=objective,
            support_lm_only=(objective == "lm_only"),
            episodic_trainable=trainable,
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "extended32", "pretrain_episodes": 8, "episodic_objective": objective, "episodic_trainable": trainable},
        )
        extended_names.append(run_id)
    for lambda_name in ["lambda_struct", "lambda_homeo", "lambda_anchor", "lambda_l0", "lambda_router_spec"]:
        run_id = f"ext32_zero_{lambda_name}"
        specs[run_id] = RunSpec(
            run_id=run_id,
            iterations=32,
            pretrain_episodes=8,
            episodic_objective="full_aux",
            episodic_trainable="outer_only",
            full_validation=False,
            val_slice_seqs=val_slice_seqs,
            config_diff={"matrix": "extended32", lambda_name: 0.0},
        )
        extended_names.append(run_id)

    if single:
        if single not in specs:
            raise ValueError(f"Unknown --single run_id {single}; available: {sorted(specs)}")
        return [specs[single]]
    if matrix == "dry1":
        return [specs["dry1"]]
    if matrix == "primary64-fullval":
        return [specs[name] for name in ["pure_pretrain64", "pretrain64_auxzero", "episodic64_lm_only_outer", "episodic64_full_aux_outer"]]
    if matrix == "extended32-sliceval":
        return [specs[name] for name in extended_names]
    if matrix == "schedule128-fullval":
        return [
            specs[name]
            for name in [
                "pure_pretrain128",
                "episodic128_lm_only_base",
                "episodic128_lm_dominant_base_outer",
                "episodic128_full_aux_base_outer",
                "pure64_then_episodic64_lm_dominant_base_outer",
            ]
        ]
    if matrix == "lmfirst-fullval":
        return [
            specs[name]
            for name in [
                "pure_pretrain256",
                "pure_pretrain512",
                "pure_pretrain256_then_lm_dominant_episodic128",
                "pure_pretrain256_then_full_aux_episodic128",
            ]
        ]
    if matrix == "claude32-sliceval":
        return [specs[name] for name in claude32_names]
    if matrix == "claude-fullval":
        return [
            specs[name]
            for name in [
                "pure_pretrain256",
                "pure_pretrain384",
                "pure_pretrain512",
                "pure_pretrain256_then_lm_only_ep128_lrmatch",
                "pure_pretrain256_then_lm_dominant_episodic128",
                "pure_pretrain256_then_lm_dominant_ep128_anchor0_lrmatch",
            ]
        ]
    raise ValueError(f"Unsupported matrix: {matrix}")


def run_training_spec(
    *,
    mod,
    base_args,
    spec: RunSpec,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_space_lut: Tensor,
    boundary_lut: Tensor,
    output_dir: Path,
    train_log_every: int,
    save_community_snapshots: bool,
    reported_eval_source: str,
    promotion_baseline_bpb: float,
) -> dict[str, Any]:
    set_seed(int(base_args.seed))
    args = copy.copy(base_args)
    args.iterations = spec.iterations
    args.pretrain_episodes = spec.pretrain_episodes
    args.retention_eval_every = 0
    args.val_loss_every = 0
    aux_old: dict[str, float] = {}
    if spec.aux_zero:
        aux_old.update(apply_aux_zero(args))
    for key, value in spec.config_diff.items():
        if key in spec.arg_overrides:
            continue
        if key.startswith("lambda_") and hasattr(args, key):
            aux_old.update(set_single_aux_zero(args, key))
    original_arg_values, resolved_arg_overrides = apply_arg_overrides(args, spec.arg_overrides)

    model = make_model(mod, args, device)
    model.train()
    loader = mod.EpisodeLoader(args, args.train_files, rank=0, world_size=1, device=device)
    memory = mod.EpisodicMemory(args.memory_size, args.model_dim, args.num_assemblies)
    events_path = output_dir / f"{spec.run_id}.events.jsonl"
    summary_path = output_dir / f"{spec.run_id}.summary.json"

    phase_optimizer: torch.optim.Optimizer | None = None
    current_phase_key = ""
    last_components: dict[str, Any] = {}
    last_bucket_stats: dict[str, Any] = {}
    optimizer_summaries: dict[str, Any] = {}
    start = time.perf_counter()

    for episode_idx in range(spec.iterations):
        support_batches, query_batches, anchor_batch = loader.next_episode()
        phase = "pretrain" if episode_idx < spec.pretrain_episodes else "episodic"
        if phase == "pretrain":
            trainability = spec.pretrain_trainable
            lr = args.stable_lr
        else:
            trainability = spec.episodic_trainable
            lr = args.outer_lr
        phase_key = f"{phase}:{trainability}"
        if phase_key != current_phase_key:
            set_trainability(model, trainability)
            named_params = trainable_named_params(model)
            phase_optimizer = build_optimizer(mod, args, named_params, lr, device)
            optimizer_summaries[phase_key] = optimizer_summary(phase_optimizer, named_params)
            current_phase_key = phase_key
        assert phase_optimizer is not None
        named_params = trainable_named_params(model)
        phase_optimizer.zero_grad(set_to_none=True)
        before = clone_named_params(named_params)

        if phase == "pretrain":
            losses: list[Tensor] = []
            with autocast_for(device):
                for x, y in support_batches + query_batches:
                    losses.append(model(x, y))
                loss = torch.stack(losses).mean()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([param for _, param in named_params], args.grad_clip_norm)
            pre_step_stats = bucket_stats(named_params)
            phase_optimizer.step()
            post_step_stats = bucket_stats(named_params, before)
            components = objective_component_record(
                phase=phase,
                total_loss=loss,
                lm_ce=loss,
                args=args,
            )
            active_branches = active_edges = 0
            supervisor_info = None
        else:
            temp_pool = mod.TempPool(args, device)
            support_gains: list[float] = []
            support_before_values: list[float] = []
            support_after_values: list[float] = []
            affinity_traces: list[Any] = []
            support_proxy_norm = 0.0
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
                    support_before = support_total(support_trace, temp_pool, args, spec.support_lm_only)
                grad_proxies = temp_pool.fast_update(support_before)
                support_proxy_norm += math.sqrt(sum(float(v) ** 2 for v in grad_proxies.values()))
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
                        support_after = support_total(probe_trace, temp_pool, args, spec.support_lm_only)
                before_value = float(support_before.detach().item())
                after_value = float(support_after.detach().item())
                support_before_values.append(before_value)
                support_after_values.append(after_value)
                support_gains.append(before_value - after_value)
                temp_pool.update_online_stats(probe_trace, before_value - after_value)
                memory.write(probe_trace)
                affinity_traces.append(probe_trace)

            episodic_step = episode_idx - spec.pretrain_episodes
            lm_weight, episodic_weight = mod.episodic_objective_weights(args, episodic_step)
            query_traces: list[Any] = []
            anchor_ctx = mod.EpisodeContext(
                temp_pool=temp_pool,
                episodic_memory=memory,
                phase="anchor",
                allow_growth=False,
                enable_temp=True,
            )
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
                anchor_loss = model(anchor_batch[0], anchor_batch[1], episode_ctx=anchor_ctx)
                episodic_regularizer = (
                    args.lambda_struct * query_struct_loss
                    + args.lambda_homeo * query_homeo_loss
                    + args.lambda_anchor * anchor_loss
                    + args.lambda_l0 * query_l0_penalty
                    + args.lambda_router_spec * query_router_spec_loss
                )
                if spec.episodic_objective == "lm_only":
                    loss = query_lm_loss
                    effective_lm_weight = 1.0
                    effective_episodic_weight = 0.0
                elif spec.episodic_objective == "lm_dominant":
                    loss = query_lm_loss + 0.05 * episodic_regularizer
                    effective_lm_weight = 1.0
                    effective_episodic_weight = 0.05
                else:
                    loss = lm_weight * query_lm_loss + episodic_weight * episodic_regularizer
                    effective_lm_weight = lm_weight
                    effective_episodic_weight = episodic_weight
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([param for _, param in named_params], args.grad_clip_norm)
            pre_step_stats = bucket_stats(named_params)
            phase_optimizer.step()
            post_step_stats = bucket_stats(named_params, before)
            model.update_query_ema(query_lm_loss.detach())
            for query_trace in query_traces:
                memory.write(query_trace)
                affinity_traces.append(query_trace)
            decisions = model.consolidator.evaluate(model, temp_pool, support_batches + query_batches, [anchor_batch], memory, device)
            consolidation_counts = model.consolidator.apply(model, temp_pool, decisions)
            model.update_affinity_ema(affinity_traces)
            active_branches, active_edges = temp_pool.active_counts()
            supervisor_info = None
            if args.rebus_supervisor_enabled:
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
                supervisor_info = model.supervisor_step(mean_probs, float(query_lm_loss.detach().item()), active_mask)
            cadence_due = ((episodic_step + 1) % args.community_refresh_episodes == 0)
            if cadence_due and not (supervisor_info and supervisor_info.get("transitioned_to_grow")):
                model.refresh_communities()
            components = objective_component_record(
                phase=phase,
                total_loss=loss,
                lm_ce=query_lm_loss,
                support_before=float(np.mean(support_before_values)) if support_before_values else None,
                support_after=float(np.mean(support_after_values)) if support_after_values else None,
                support_gain=float(np.mean(support_gains)) if support_gains else None,
                anchor_loss=anchor_loss,
                struct_loss=query_struct_loss,
                homeo_loss=query_homeo_loss,
                l0_penalty=query_l0_penalty,
                router_spec_loss=query_router_spec_loss,
                episodic_regularizer=episodic_regularizer,
                lm_weight=effective_lm_weight,
                episodic_weight=effective_episodic_weight,
                args=args,
            )
            components["support_fast_proxy_norm"] = float(support_proxy_norm)
            components["consolidation_counts"] = consolidation_counts

        last_components = components
        last_bucket_stats = post_step_stats
        record = {
            "type": "step",
            "run_id": spec.run_id,
            "episode": episode_idx + 1,
            "phase": phase,
            "trainability": trainability,
            "active_branches": int(active_branches),
            "active_edges": int(active_edges),
            "objective": components,
            "grad_update_pre_step": pre_step_stats,
            "grad_update_post_step": post_step_stats,
            "supervisor": supervisor_info,
        }
        write_jsonl(events_path, record)
        if train_log_every > 0 and ((episode_idx + 1) <= 3 or (episode_idx + 1) % train_log_every == 0 or (episode_idx + 1) == spec.iterations):
            print(
                f"{spec.run_id} episode={episode_idx + 1}/{spec.iterations} "
                f"phase={phase} total={components['total_loss']:.4f} lm_ce={components['lm_ce']:.4f} "
                f"active=({active_branches},{active_edges})"
            )

    sync(device)
    train_seconds = time.perf_counter() - start
    probe_tokens = truncate_val_tokens(mod, args, val_tokens, spec.val_slice_seqs)
    validation = validation_report(
        mod,
        args,
        model,
        device,
        val_tokens,
        probe_tokens,
        base_bytes_lut,
        has_space_lut,
        boundary_lut,
        spec.full_validation,
        reported_eval_source,
        promotion_baseline_bpb,
    )
    community_snapshot = None
    community_snapshot_path = None
    if save_community_snapshots:
        community_snapshot = build_community_snapshot(args=args, model=model, spec=spec, validation=validation)
        community_snapshot_path = output_dir / f"{spec.run_id}.community_snapshot.json"
        write_json(community_snapshot_path, community_snapshot)
    summary = {
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "train_seconds": train_seconds,
        "last_objective": last_components,
        "last_update_stats": last_bucket_stats,
        "optimizer_groups": optimizer_summaries,
        "validation": validation,
        "community_snapshot_path": str(community_snapshot_path) if community_snapshot_path is not None else None,
        "community_metrics": community_snapshot["metrics"] if community_snapshot is not None else None,
        "aux_original_values": aux_old,
        "original_arg_values": original_arg_values,
        "resolved_arg_overrides": resolved_arg_overrides,
        "final_supervisor": {
            "mode": ("CLOSED", "GROW", "RECOVER")[int(model.supervisor_mode.item())],
            "mode_counts": [int(value) for value in model.supervisor_mode_counts.detach().cpu().tolist()],
            "health_min": float(model.assembly_health.min().item()),
            "health_mean": float(model.assembly_health.mean().item()),
            "growth_events": int(model.growth_event_count_buf.item()),
        },
    }
    write_json(summary_path, summary)
    print(
        f"RUN_SUMMARY {spec.run_id} train_seconds={train_seconds:.1f} "
        f"reported_source={validation['reported_eval_source']} "
        f"reported_scope={validation['reported_eval_scope']} "
        f"reported_loss={validation['reported_eval_loss']:.6f} reported_bpb={validation['reported_eval_bpb']:.6f} "
        f"beats_baseline={validation['beats_promotion_baseline']} "
        f"label_free_probe_bpb={validation['label_free_probe_bpb']:.6f} "
        f"eval_mutations={validation['reported_eval_buffer_mutation']['changed_count']}"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return summary


def print_summary_table(summaries: list[dict[str, Any]]) -> None:
    print("")
    print("| run_id | train_lm_ce | train_total | reported_scope | reported_bpb | full_eval_bpb | label_free_probe_bpb | mode_counts | growth | top_update_bucket | eval_mutations | seconds | config_diff |")
    print("| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |")
    for summary in summaries:
        objective = summary["last_objective"]
        validation = summary["validation"]
        bucket_stats = summary["last_update_stats"]
        top_bucket = max(bucket_stats.items(), key=lambda item: float(item[1].get("update_to_param", 0.0)))[0] if bucket_stats else "none"
        supervisor = summary.get("final_supervisor", {})
        config = dict(summary.get("spec", {}).get("config_diff", {}))
        if summary.get("resolved_arg_overrides"):
            config["resolved_arg_overrides"] = summary["resolved_arg_overrides"]
        config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
        print(
            f"| {summary['run_id']} "
            f"| {float(objective.get('lm_ce') or float('nan')):.4f} "
            f"| {float(objective.get('total_loss') or float('nan')):.4f} "
            f"| {validation['reported_eval_scope']} "
            f"| {validation['reported_eval_bpb']:.4f} "
            f"| {validation['current_eval_bpb']:.4f} "
            f"| {validation['label_free_probe_bpb']:.4f} "
            f"| {supervisor.get('mode_counts', [])} "
            f"| {supervisor.get('growth_events', 0)} "
            f"| {top_bucket} "
            f"| {validation['reported_eval_buffer_mutation']['changed_count']} "
            f"| {summary['train_seconds']:.1f} "
            f"| `{config_text}` |"
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
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = cli.output_root.resolve() / f"{timestamp}_{cli.matrix if not cli.single else cli.single}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_root={cli.repo_root}")
    print(f"w9_path={cli.w9_path}")
    print(f"output_dir={output_dir}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device_name={torch.cuda.get_device_name(device)}")
    print(f"train_files={args.train_files}")
    print(f"val_files={args.val_files}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != int(args.vocab_size):
        raise AssertionError(f"tokenizer vocab {sp.vocab_size()} != args vocab {args.vocab_size}")
    base_bytes_lut, has_space_lut, boundary_lut = mod.build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = mod.load_validation_tokens(args.val_files, args.train_seq_len)

    invariants = {"w9": run_w9_invariants(mod, args, device)}
    if not cli.skip_rebus_lab_oracle:
        invariants["rebus_lab"] = run_rebus_lab_oracle(cli.rebus_lab_path.resolve())
    write_json(output_dir / "invariants.json", invariants)
    print("INVARIANTS " + json.dumps(invariants, sort_keys=True, default=json_default))

    specs = make_specs(cli.matrix, cli.single, cli.val_slice_seqs)
    write_json(output_dir / "matrix_specs.json", [asdict(spec) for spec in specs])
    summaries = []
    for spec in specs:
        summaries.append(
            run_training_spec(
                mod=mod,
                base_args=args,
                spec=spec,
                device=device,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_space_lut=has_space_lut,
                boundary_lut=boundary_lut,
                output_dir=output_dir,
                train_log_every=cli.train_log_every,
                save_community_snapshots=bool(cli.save_community_snapshots),
                reported_eval_source=cli.reported_eval_source,
                promotion_baseline_bpb=float(cli.promotion_baseline_bpb),
            )
        )
    write_json(output_dir / "matrix_summary.json", summaries)
    print_summary_table(summaries)
    print(f"MATRIX_SUMMARY_PATH {output_dir / 'matrix_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
