from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .eval_pu_osls import EvalConfig, evaluate_pu_osls, print_results
from .model import CustomNanoTabPFNModel
from .prior_data import (
    PriorGeneratorConfig,
    TabICLPriorConfig,
    TestLabelShiftConfig,
    generate_batch,
)


@dataclass
class CurriculumConfig:
    enabled: bool = False
    update_every_steps: int = 500
    max_updates: int = 20
    start_max_classes: Optional[int] = 2
    start_max_features: Optional[int] = None
    start_min_rows: Optional[int] = 100
    start_max_rows: Optional[int] = 300
    start_remove_poisson_lambda: Optional[float] = 0.5
    tabicl_sampled_hp_start: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _moving_average(values: List[float], window: int) -> float:
    if len(values) == 0:
        return float("nan")
    width = min(window, len(values))
    return float(sum(values[-width:]) / width)


def _summary_stats(values: torch.Tensor) -> str:
    values_f = values.to(torch.float)
    return f"min={int(values.min().item())} mean={values_f.mean().item():.1f} max={int(values.max().item())}"


def _curriculum_update_idx(step: int, curriculum_cfg: CurriculumConfig) -> int:
    update_every_steps = max(1, curriculum_cfg.update_every_steps)
    max_updates = max(0, curriculum_cfg.max_updates)
    return int(min(step // update_every_steps, max_updates))


def _curriculum_progress(step: int, curriculum_cfg: CurriculumConfig) -> float:
    max_updates = max(0, curriculum_cfg.max_updates)
    update_idx = _curriculum_update_idx(step, curriculum_cfg)
    if max_updates == 0:
        return 1.0
    return float(update_idx / max_updates)


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _build_step_cfg(
    base_cfg: PriorGeneratorConfig,
    *,
    step: int,
    curriculum_cfg: CurriculumConfig | None,
) -> PriorGeneratorConfig:
    if curriculum_cfg is None or not curriculum_cfg.enabled:
        return base_cfg

    step_cfg = copy.deepcopy(base_cfg)
    progress = _curriculum_progress(step, curriculum_cfg)

    start_max_features = curriculum_cfg.start_max_features
    if start_max_features is None:
        start_max_features = base_cfg.min_features
    int_fields = [
        ("max_classes", curriculum_cfg.start_max_classes, 2),
        ("max_features", start_max_features, 1),
        ("min_rows", curriculum_cfg.start_min_rows, 2),
        ("max_rows", curriculum_cfg.start_max_rows, 2),
    ]
    for field_name, start_val, lower_bound in int_fields:
        if start_val is None:
            continue
        end_val = int(getattr(base_cfg, field_name))
        interpolated = int(round(_lerp(float(start_val), float(end_val), progress)))
        setattr(step_cfg, field_name, max(lower_bound, interpolated))

    if curriculum_cfg.start_remove_poisson_lambda is not None:
        step_cfg.remove_poisson_lambda = _lerp(
            float(curriculum_cfg.start_remove_poisson_lambda),
            float(base_cfg.remove_poisson_lambda),
            progress,
        )

    step_cfg.min_features = base_cfg.min_features
    step_cfg.max_features = max(step_cfg.max_features, step_cfg.min_features)
    step_cfg.min_rows = min(step_cfg.min_rows, step_cfg.max_rows)

    for hp_name, hp_overrides in curriculum_cfg.tabicl_sampled_hp_start.items():
        target_hp = step_cfg.tabicl.scm_sampled_hp.get(hp_name)
        final_hp = base_cfg.tabicl.scm_sampled_hp.get(hp_name)
        if not isinstance(target_hp, dict) or not isinstance(final_hp, dict):
            continue
        for param_name, start_value in hp_overrides.items():
            end_value = final_hp.get(param_name)
            if not isinstance(start_value, (int, float)):
                continue
            if not isinstance(end_value, (int, float)) or isinstance(end_value, bool):
                continue
            new_value = _lerp(float(start_value), float(end_value), progress)
            if isinstance(end_value, int):
                new_value = int(round(new_value))
            target_hp[param_name] = new_value

    return step_cfg


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    model: CustomNanoTabPFNModel,
    cfg: PriorGeneratorConfig,
    *,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_steps: int,
    unseen_label: int,
    eval_cfg: EvalConfig | None = None,
    eval_interval: int = 100,
    curriculum_cfg: CurriculumConfig | None = None,
) -> Tuple[CustomNanoTabPFNModel, List[float]]:
    """
    True on-the-fly training:
      Each step:
        - generate one batch of synthetic datasets (B tasks)
        - run model + loss + update
        - discard batch
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ignore_index handles padded test rows only (unseen label is a valid class)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    rng = np.random.default_rng(cfg.seed)
    losses: List[float] = []
    skipped_nonfinite = 0

    for step in range(num_steps):
        step_cfg = _build_step_cfg(cfg, step=step, curriculum_cfg=curriculum_cfg)
        batch = generate_batch(step_cfg, batch_size=batch_size, device=device, rng=rng)
        x = batch["x"]                    # (B,R,F)
        y_full = batch["y"]               # (B,R)
        split = batch["train_test_split_index"]  # (B,)
        removed_count = batch["removed_class_count"]  # (B,)
        num_classes = batch["num_classes"]  # (B,)
        num_features = batch.get("num_features")
        if num_features is None:
            num_features = torch.full(
                (x.shape[0],),
                fill_value=x.shape[2],
                dtype=torch.long,
                device=x.device,
            )
        row_mask = batch["row_mask"]  # (B,R)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        # Build label input for model: only provide y_train; encoder will pad internally.
        # Since splits vary per task, slice per task.
        y_train_list = []
        for b in range(x.shape[0]):
            split_b = int(split[b].item())
            y_train_list.append(y_full[b:b+1, :split_b])
        # Pad y_train_list to max split so we can batch; model handles per-task split anyway.
        max_split = int(split.max().item())
        y_train_padded = torch.full(
            (x.shape[0], max_split),
            fill_value=unseen_label,
            device=x.device,
            dtype=y_full.dtype,
        )
        for b, ytb in enumerate(y_train_list):
            y_train_padded[b, : ytb.shape[1]] = ytb

        logits = model((x, y_train_padded), split)  # (B, max_test, num_outputs)
        if not torch.isfinite(logits).all():
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            print(f"warning: non-finite logits at step {step+1}, skipping optimizer step")
            continue

        # Build targets for test rows; map removed classes to unseen_label,
        # then remap unseen_label to the last index in each task's sliced logits.
        B, R = y_full.shape
        max_test = logits.shape[1]
        targets = torch.full((B, max_test), fill_value=-100, device=x.device, dtype=torch.long)
        seen_counts = (num_classes - removed_count).to(torch.long)
        for b in range(B):
            split_b = int(split[b].item())
            seen_count = int(seen_counts[b].item())
            # Actual test rows for this task (exclude padding rows)
            row_count = int(row_mask[b].sum().item())
            y_test = y_full[b, split_b:row_count].clone()
            # Map labels >= seen_count to unseen_label
            y_test = torch.where(y_test >= seen_count, torch.tensor(unseen_label, device=x.device), y_test)
            # Remap unseen_label to the last index in the sliced logits for this task
            y_test = torch.where(y_test == unseen_label, torch.tensor(seen_count, device=x.device), y_test)
            targets[b, : y_test.shape[0]] = y_test.to(torch.long)

        # Slice logits to seen classes + unseen for each task, pad to common class dim
        max_seen = int(seen_counts.max().item())
        logits_sliced = torch.full(
            (B, max_test, max_seen + 1),
            fill_value=torch.finfo(logits.dtype).min,
            device=logits.device,
            dtype=logits.dtype,
        )
        for b in range(B):
            seen_count = int(seen_counts[b].item())
            if seen_count > 0:
                logits_sliced[b, :, :seen_count] = logits[b, :, :seen_count]
            # unseen logit is always the last index of full head
            logits_sliced[b, :, seen_count] = logits[b, :, -1]

        targets_flat = targets.reshape(-1)                 # (B*max_test,)
        logits_flat = logits_sliced.reshape(-1, logits_sliced.shape[-1])  # (B*max_test, C')

        loss = criterion(logits_flat, targets_flat)
        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            print(f"warning: non-finite loss at step {step+1}, skipping optimizer step")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(float(loss.detach().cpu()))

        # Progress logging
        if (step + 1) % 10 == 0 or step == 0:
            ma10 = _moving_average(losses, 10)
            if curriculum_cfg is not None and curriculum_cfg.enabled:
                curr_update_idx = _curriculum_update_idx(step, curriculum_cfg)
                max_updates = max(0, curriculum_cfg.max_updates)
                display_idx = min(curr_update_idx + 1, max_updates) if max_updates > 0 else 0
                curriculum_step = f"{display_idx}/{max_updates}"
            else:
                curriculum_step = "disabled"

            print(
                f"step {step+1:5d}/{num_steps} | "
                f"loss={losses[-1]:.4f} | "
                f"ma_loss={ma10:.4f} | "
                f"curriculum_step={curriculum_step}"
            )

        if eval_interval > 0 and (step + 1) % eval_interval == 0:
            print(f"\nEval at step {step+1}...")
            res = evaluate_pu_osls(
                model,
                cfg_prior=cfg,
                unseen_label=unseen_label,
                device=device,
                eval_cfg=eval_cfg or EvalConfig(),
            )
            print_results(res)

    return model, losses


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PU/OSLS TabPFN on synthetic prior tasks.")
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-backend", type=str, choices=["tabicl", "legacy"], default="tabicl")
    parser.add_argument("--tabicl-prior-type", type=str, default="mlp_scm")
    parser.add_argument("--tabicl-n-jobs", type=int, default=1)
    parser.add_argument("--tabicl-batch-size-per-gp", type=int, default=4)
    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--curriculum-update-every-steps", type=int, default=500)
    parser.add_argument("--curriculum-max-updates", type=int, default=20)
    parser.add_argument("--curriculum-start-max-classes", type=int, default=2)
    parser.add_argument("--curriculum-start-max-features", type=int, default=None)
    parser.add_argument("--curriculum-start-min-rows", type=int, default=100)
    parser.add_argument("--curriculum-start-max-rows", type=int, default=300)
    parser.add_argument("--curriculum-start-remove-poisson-lambda", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    cfg = PriorGeneratorConfig(
        max_classes=10,
        min_features=3,
        max_features=8,
        min_rows=500,
        max_rows=1000,
        min_train_fraction=0.4,
        max_train_fraction=0.8,
        remove_poisson_lambda=1.0,
        seed=args.seed,
        label_noise=0.1,
        prior_backend=args.prior_backend,
        tabicl=TabICLPriorConfig(
            prior_type=args.tabicl_prior_type,
            n_jobs=args.tabicl_n_jobs,
            batch_size_per_gp=args.tabicl_batch_size_per_gp,
        ),
        test_label_shift=TestLabelShiftConfig(enabled=False, strategy="none", strength=0.0),
    )

    device = get_device()
    print(f"Training on device: {device}")

    unseen_label = cfg.max_classes
    num_outputs = cfg.max_classes + 1  # include unseen_label as a reserved id

    model = CustomNanoTabPFNModel(
        embedding_size=32,
        num_attention_heads=4,
        mlp_hidden_size=64,
        num_layers=2,
        num_outputs=num_outputs,
        unseen_label=unseen_label,
    )

    start = time.time()
    eval_cfg = EvalConfig(
        n_tasks=args.eval_tasks,
        batch_size=args.batch_size,
        seed=999,
        outlier_score="msp",
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
                "hidden_dim": {"max_mean": 24.0},
                "num_causes": {"max_mean": 4.0},
            },
        )

    model, losses = train(
        model,
        cfg,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        num_steps=args.num_steps,
        unseen_label=unseen_label,
        eval_cfg=eval_cfg,
        eval_interval=args.eval_interval,
        curriculum_cfg=curriculum_cfg,
    )
    print(f"Done. time={time.time()-start:.2f}s | final loss={losses[-1]:.4f}")

    print("\nFinal eval...")
    res = evaluate_pu_osls(
        model,
        cfg_prior=cfg,
        unseen_label=unseen_label,
        device=device,
        eval_cfg=eval_cfg,
    )
    print_results(res)

if __name__ == "__main__":
    main()
