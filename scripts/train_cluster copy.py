from __future__ import annotations

import argparse
import inspect
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _ensure_src_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


ROOT = _ensure_src_on_path()

from pu_osls_tabpfn.eval_pu_osls import EvalConfig, evaluate_pu_osls, print_results
from pu_osls_tabpfn.model import CustomNanoTabPFNModel
from pu_osls_tabpfn.prior_data import (
    PriorGeneratorConfig,
    TabICLPriorConfig,
    TestLabelShiftConfig,
    generate_batch,
)
from pu_osls_tabpfn.train import CurriculumConfig, _build_step_cfg, _curriculum_update_idx, _moving_average


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU cluster training with checkpoint resume.")
    parser.add_argument("--num-steps", type=int, default=30000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=99)

    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)

    parser.add_argument("--max-classes", type=int, default=5)
    parser.add_argument("--min-features", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=10)
    parser.add_argument("--min-rows", type=int, default=800)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--min-train-fraction", type=float, default=0.5)
    parser.add_argument("--max-train-fraction", type=float, default=0.6)
    parser.add_argument("--remove-poisson-lambda", type=float, default=1.2)
    parser.add_argument("--min-train-rows-after-removal", type=int, default=30)

    parser.add_argument("--tabicl-prior-type", type=str, default="mlp_scm")
    parser.add_argument("--tabicl-n-jobs", type=int, default=1)
    parser.add_argument("--tabicl-batch-size-per-gp", type=int, default=4)
    parser.add_argument("--tabicl-batch-size-per-subgp", type=int, default=2)

    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--curriculum-update-every-steps", type=int, default=100)
    parser.add_argument("--curriculum-max-updates", type=int, default=20)
    parser.add_argument("--curriculum-start-max-classes", type=int, default=3)
    parser.add_argument("--curriculum-start-max-features", type=int, default=4)
    parser.add_argument("--curriculum-start-min-rows", type=int, default=700)
    parser.add_argument("--curriculum-start-max-rows", type=int, default=800)
    parser.add_argument("--curriculum-start-remove-poisson-lambda", type=float, default=0.3)

    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-tasks", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)

    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/cluster_checkpoints"))
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--resume-from", type=Path, default=None)

    return parser.parse_args()


def setup_distributed() -> Dict[str, int | bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "distributed": distributed,
        "device": device,
    }


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def save_checkpoint(
    *,
    checkpoint_path: Path,
    model: CustomNanoTabPFNModel | DDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: List[float],
    args: argparse.Namespace,
    cfg: PriorGeneratorConfig,
    curriculum_cfg: CurriculumConfig | None,
) -> None:
    base_model = model.module if isinstance(model, DDP) else model
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "args": vars(args),
        "cfg": asdict(cfg),
        "curriculum_cfg": asdict(curriculum_cfg) if curriculum_cfg is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, checkpoint_path)


def maybe_load_checkpoint(
    *,
    resume_from: Path | None,
    model: CustomNanoTabPFNModel | DDP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, List[float]]:
    if resume_from is None:
        return 0, []
    if not resume_from.exists():
        return 0, []

    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(resume_from, **load_kwargs)
    base_model = model.module if isinstance(model, DDP) else model
    base_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        cuda_states = checkpoint["cuda_rng_state_all"]
        if isinstance(cuda_states, list) and len(cuda_states) > 0:
            # Reuse rank0 RNG state for all ranks on resume.
            torch.cuda.set_rng_state(cuda_states[0])

    start_step = int(checkpoint["step"])
    losses = list(checkpoint.get("losses", []))
    return start_step, losses


def _summary_stats(values: torch.Tensor) -> str:
    values_f = values.to(torch.float)
    return f"min={int(values.min().item())} mean={values_f.mean().item():.1f} max={int(values.max().item())}"


def main() -> None:
    args = parse_args()
    dist_info = setup_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    distributed = bool(dist_info["distributed"])
    device = dist_info["device"]  # type: ignore[assignment]
    main_proc = is_main_process(rank)

    if args.global_batch_size % world_size != 0:
        raise ValueError(
            f"--global-batch-size ({args.global_batch_size}) must be divisible by world size ({world_size})."
        )
    local_batch_size = args.global_batch_size // world_size

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    cfg = PriorGeneratorConfig(
        max_classes=args.max_classes,
        min_features=args.min_features,
        max_features=args.max_features,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_train_fraction=args.min_train_fraction,
        max_train_fraction=args.max_train_fraction,
        remove_poisson_lambda=args.remove_poisson_lambda,
        seed=args.seed,
        min_train_rows_after_removal=args.min_train_rows_after_removal,
        prior_backend="tabicl",
        tabicl=TabICLPriorConfig(
            prior_type=args.tabicl_prior_type,
            n_jobs=args.tabicl_n_jobs,
            batch_size_per_gp=args.tabicl_batch_size_per_gp,
            batch_size_per_subgp=args.tabicl_batch_size_per_subgp,
        ),
        test_label_shift=TestLabelShiftConfig(enabled=False, strategy="none", strength=0.0),
    )

    curriculum_cfg = None
    if args.use_curriculum:
        curriculum_cfg = CurriculumConfig(
            enabled=True,
            update_every_steps=args.curriculum_update_every_steps,
            max_updates=args.curriculum_max_updates,
            start_max_classes=args.curriculum_start_max_classes,
            start_max_features=args.curriculum_start_max_features,
            start_min_rows=args.curriculum_start_min_rows,
            start_max_rows=args.curriculum_start_max_rows,
            start_remove_poisson_lambda=args.curriculum_start_remove_poisson_lambda,
            tabicl_sampled_hp_start={
                "num_layers": {"max_mean": 2.0},
                "hidden_dim": {"max_mean": 24.0, "min_mean": 4.0},
                "num_causes": {"max_mean": 4.0},
            },
        )

    unseen_label = cfg.max_classes
    model = CustomNanoTabPFNModel(
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        num_layers=args.num_layers,
        num_outputs=cfg.max_classes + 1,
        unseen_label=unseen_label,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    rng = np.random.default_rng(args.seed + rank)
    losses: List[float] = []
    skipped_nonfinite = 0

    start_step = 0
    if args.resume_from is not None:
        start_step, losses = maybe_load_checkpoint(
            resume_from=args.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        if main_proc and start_step > 0:
            print(f"Resumed from {args.resume_from} at step {start_step}.")
        if main_proc and start_step == 0:
            print(f"No checkpoint found at {args.resume_from}. Starting from scratch.")

    should_stop = False

    def _handle_signal(signum, _frame):  # type: ignore[no-untyped-def]
        nonlocal should_stop
        should_stop = True
        if main_proc:
            print(f"\nReceived signal {signum}; checkpointing and stopping gracefully.")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if main_proc:
        print(
            f"Starting training on world_size={world_size} | local_batch={local_batch_size} | "
            f"global_batch={args.global_batch_size} | device={device}"
        )

    run_start = time.time()
    last_step = start_step

    for step in range(start_step, args.num_steps):
        last_step = step + 1
        step_cfg = _build_step_cfg(cfg, step=step, curriculum_cfg=curriculum_cfg)
        batch = generate_batch(step_cfg, batch_size=local_batch_size, device=device, rng=rng)
        x = batch["x"]
        y_full = batch["y"]
        split = batch["train_test_split_index"]
        removed_count = batch["removed_class_count"]
        num_classes = batch["num_classes"]
        row_mask = batch["row_mask"]
        num_features = batch.get("num_features")
        if num_features is None:
            num_features = torch.full(
                (x.shape[0],),
                fill_value=x.shape[2],
                dtype=torch.long,
                device=x.device,
            )

        max_split = int(split.max().item())
        y_train_padded = torch.full(
            (x.shape[0], max_split),
            fill_value=unseen_label,
            device=x.device,
            dtype=y_full.dtype,
        )
        for b in range(x.shape[0]):
            split_b = int(split[b].item())
            y_train_padded[b, :split_b] = y_full[b, :split_b]

        logits = model((x, y_train_padded), split)
        if not torch.isfinite(logits).all():
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        batch_size, _rows = y_full.shape
        max_test = logits.shape[1]
        targets = torch.full((batch_size, max_test), fill_value=-100, device=x.device, dtype=torch.long)
        seen_counts = (num_classes - removed_count).to(torch.long)

        for b in range(batch_size):
            split_b = int(split[b].item())
            seen_count = int(seen_counts[b].item())
            row_count = int(row_mask[b].sum().item())
            y_test = y_full[b, split_b:row_count].clone()
            y_test = torch.where(y_test >= seen_count, torch.tensor(unseen_label, device=x.device), y_test)
            y_test = torch.where(y_test == unseen_label, torch.tensor(seen_count, device=x.device), y_test)
            targets[b, : y_test.shape[0]] = y_test.to(torch.long)

        max_seen = int(seen_counts.max().item())
        logits_sliced = torch.full(
            (batch_size, max_test, max_seen + 1),
            fill_value=torch.finfo(logits.dtype).min,
            device=logits.device,
            dtype=logits.dtype,
        )
        for b in range(batch_size):
            seen_count = int(seen_counts[b].item())
            if seen_count > 0:
                logits_sliced[b, :, :seen_count] = logits[b, :, :seen_count]
            logits_sliced[b, :, seen_count] = logits[b, :, -1]

        targets_flat = targets.reshape(-1)
        logits_flat = logits_sliced.reshape(-1, logits_sliced.shape[-1])
        loss = criterion(logits_flat, targets_flat)
        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_value = loss.detach()
        if distributed:
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            loss_value = loss_value / world_size
        losses.append(float(loss_value.cpu()))

        if main_proc and ((step + 1) % 10 == 0 or step == start_step):
            ma10 = _moving_average(losses, 10)
            if curriculum_cfg is not None and curriculum_cfg.enabled:
                curr_update_idx = _curriculum_update_idx(step, curriculum_cfg)
                max_updates = max(0, curriculum_cfg.max_updates)
                display_idx = min(curr_update_idx + 1, max_updates) if max_updates > 0 else 0
                curriculum_step = f"{display_idx}/{max_updates}"
            else:
                curriculum_step = "disabled"
            print(
                f"step {step+1:5d}/{args.num_steps} | "
                f"loss={losses[-1]:.4f} | "
                f"ma_loss={ma10:.4f} | "
                f"curriculum_step={curriculum_step}"
            )

        if main_proc and args.save_every_steps > 0 and (step + 1) % args.save_every_steps == 0:
            ckpt_name = f"checkpoint_step_{step+1:06d}.pt"
            save_checkpoint(
                checkpoint_path=args.checkpoint_dir / ckpt_name,
                model=model,
                optimizer=optimizer,
                step=step + 1,
                losses=losses,
                args=args,
                cfg=cfg,
                curriculum_cfg=curriculum_cfg,
            )
            save_checkpoint(
                checkpoint_path=args.checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step + 1,
                losses=losses,
                args=args,
                cfg=cfg,
                curriculum_cfg=curriculum_cfg,
            )
            print(f"Saved checkpoint at step {step+1} -> {args.checkpoint_dir}")

        if args.eval_interval > 0 and (step + 1) % args.eval_interval == 0 and main_proc:
            eval_cfg = EvalConfig(
                n_tasks=args.eval_tasks,
                batch_size=args.eval_batch_size,
                seed=999,
                outlier_score="msp",
            )
            eval_model = model.module if isinstance(model, DDP) else model
            res = evaluate_pu_osls(
                eval_model,
                cfg_prior=cfg,
                unseen_label=unseen_label,
                device=device,
                eval_cfg=eval_cfg,
            )
            print_results(res)

        if distributed:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            if main_proc:
                save_checkpoint(
                    checkpoint_path=args.checkpoint_dir / "interrupt_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                    losses=losses,
                    args=args,
                    cfg=cfg,
                    curriculum_cfg=curriculum_cfg,
                )
                print(f"Saved interrupt checkpoint at step {step+1}.")
            break

    if main_proc:
        save_checkpoint(
            checkpoint_path=args.checkpoint_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            step=last_step,
            losses=losses,
            args=args,
            cfg=cfg,
            curriculum_cfg=curriculum_cfg,
        )
        elapsed = time.time() - run_start
        final_loss = losses[-1] if losses else float("nan")
        print(f"Training done in {elapsed/60:.1f} min | final loss={final_loss:.4f}")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
