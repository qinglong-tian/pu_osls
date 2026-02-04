from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch

from .prior_data_legacy import LegacyPriorGeneratorConfig, generate_batch_legacy
from .tabicl_prior.dataset import PriorDataset
from .tabicl_prior.prior_config import (
    DEFAULT_FIXED_HP as TABICL_DEFAULT_FIXED_HP,
    DEFAULT_SAMPLED_HP as TABICL_DEFAULT_SAMPLED_HP,
)


@dataclass
class TestLabelShiftConfig:
    enabled: bool = False
    strategy: str = "none"
    strength: float = 0.0


@dataclass
class TabICLPriorConfig:
    batch_size_per_gp: int = 4
    batch_size_per_subgp: Optional[int] = None
    prior_type: str = "mlp_scm"
    log_seq_len: bool = False
    seq_len_per_gp: bool = False
    replay_small: bool = False
    n_jobs: int = 1
    num_threads_per_generate: int = 1
    scm_fixed_hp: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(TABICL_DEFAULT_FIXED_HP))
    scm_sampled_hp: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(TABICL_DEFAULT_SAMPLED_HP))


@dataclass
class PriorGeneratorConfig:
    max_classes: int = 10
    min_features: int = 3
    max_features: int = 8
    min_rows: int = 50
    max_rows: int = 100
    min_train_fraction: float = 0.4
    max_train_fraction: float = 0.8
    remove_poisson_lambda: float = 1.25
    x_mean: float = 0.0
    x_std: float = 1.0
    label_noise: float = 0.1
    seed: int = 0
    min_train_rows_after_removal: int = 2
    prior_backend: Literal["tabicl", "legacy"] = "tabicl"
    tabicl: TabICLPriorConfig = field(default_factory=TabICLPriorConfig)
    test_label_shift: TestLabelShiftConfig = field(default_factory=TestLabelShiftConfig)


def _sample_num_removed(rng: np.random.Generator, K: int, lam: float) -> int:
    max_removed = min(K - 1, K // 2)
    ks = np.arange(0, max_removed + 1)
    lam = max(lam, 1e-12)
    log_probs = ks * math.log(lam) - np.array([math.lgamma(k + 1) for k in ks])
    probs = np.exp(log_probs - log_probs.max())
    probs /= probs.sum()
    return int(rng.choice(ks, p=probs))


def _seed_tabicl_generators(rng: np.random.Generator) -> None:
    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_dense_if_nested(x: torch.Tensor) -> torch.Tensor:
    if getattr(x, "is_nested", False):
        return x.to_padded_tensor(0.0)
    return x


def _remap_to_contiguous(labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    unique_vals = torch.unique(labels)
    sorted_vals = torch.sort(unique_vals).values
    remapped = torch.searchsorted(sorted_vals, labels)
    return remapped.to(torch.long), int(sorted_vals.numel())


def _apply_test_label_shift(
    y_test: torch.Tensor,
    *,
    num_classes: int,
    cfg: TestLabelShiftConfig,
    rng: np.random.Generator,
) -> torch.Tensor:
    if not cfg.enabled or cfg.strategy == "none":
        return y_test
    raise NotImplementedError(
        f"test label shift strategy '{cfg.strategy}' is reserved for future extension."
    )


def _build_tabicl_full_batch(
    cfg: PriorGeneratorConfig,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _seed_tabicl_generators(rng)
    prior = PriorDataset(
        batch_size=batch_size,
        batch_size_per_gp=cfg.tabicl.batch_size_per_gp,
        batch_size_per_subgp=cfg.tabicl.batch_size_per_subgp,
        min_features=cfg.min_features,
        max_features=cfg.max_features,
        max_classes=cfg.max_classes,
        min_seq_len=cfg.min_rows,
        max_seq_len=cfg.max_rows,
        log_seq_len=cfg.tabicl.log_seq_len,
        seq_len_per_gp=cfg.tabicl.seq_len_per_gp,
        min_train_size=cfg.min_train_fraction,
        max_train_size=cfg.max_train_fraction,
        replay_small=cfg.tabicl.replay_small,
        prior_type=cfg.tabicl.prior_type,
        scm_fixed_hp=cfg.tabicl.scm_fixed_hp,
        scm_sampled_hp=cfg.tabicl.scm_sampled_hp,
        n_jobs=cfg.tabicl.n_jobs,
        num_threads_per_generate=cfg.tabicl.num_threads_per_generate,
        device="cpu",
    )

    x_full, y_full, d_full, seq_lens, train_sizes = prior.get_batch(batch_size=batch_size)
    x_full = _to_dense_if_nested(x_full).to(torch.float32)
    y_full = _to_dense_if_nested(y_full).to(torch.long)
    d_full = d_full.to(torch.long)
    return x_full, y_full, d_full, seq_lens.to(torch.long), train_sizes.to(torch.long)


def _generate_batch_tabicl(
    cfg: PriorGeneratorConfig,
    *,
    batch_size: int,
    device: Optional[torch.device],
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    x_full, y_full, d_full, seq_lens, split_full = _build_tabicl_full_batch(cfg, batch_size=batch_size, rng=rng)
    unseen_label = cfg.max_classes

    x_list = []
    y_list = []
    split_list = []
    num_classes_list = []
    active_features_list = []

    removed_class_mask = np.zeros((batch_size, cfg.max_classes), dtype=bool)
    seen_class_mask = np.zeros((batch_size, cfg.max_classes), dtype=bool)

    for b in range(batch_size):
        row_count = int(seq_lens[b].item())
        split = int(split_full[b].item())
        split = max(1, min(split, row_count - 1))

        active_features = int(d_full[b].item())
        active_features = max(1, min(active_features, x_full.shape[-1]))
        x_row = x_full[b, :row_count, :active_features].clone()
        x_row = torch.nan_to_num(x_row, nan=0.0, posinf=1e4, neginf=-1e4)
        y_row = y_full[b, :row_count].clone()
        y_row, num_classes = _remap_to_contiguous(y_row)
        if num_classes > cfg.max_classes:
            raise ValueError(
                f"TabICL generated {num_classes} classes but max_classes={cfg.max_classes}."
            )

        y_train = y_row[:split]
        y_test = y_row[split:]
        y_test = _apply_test_label_shift(
            y_test,
            num_classes=num_classes,
            cfg=cfg.test_label_shift,
            rng=rng,
        )

        x_train = x_row[:split]
        x_test = x_row[split:]

        num_removed = _sample_num_removed(rng, num_classes, cfg.remove_poisson_lambda)
        min_train_rows = max(1, cfg.min_train_rows_after_removal)
        removed: set[int] = set(range(num_classes - num_removed, num_classes)) if num_removed > 0 else set()
        while True:
            if removed:
                remove_mask = torch.isin(y_train, torch.tensor(sorted(removed), device=y_train.device))
                keep_mask_train = ~remove_mask
                x_train_kept = x_train[keep_mask_train]
                y_train_kept = y_train[keep_mask_train]
            else:
                x_train_kept = x_train
                y_train_kept = y_train

            if x_train_kept.shape[0] >= min_train_rows or len(removed) == 0:
                break
            k = len(removed) - 1
            removed = set(range(num_classes - k, num_classes)) if k > 0 else set()

        if removed:
            removed_class_mask[b, list(removed)] = True

        if y_train_kept.numel() > 0:
            present = torch.unique(y_train_kept).cpu().numpy()
            seen_class_mask[b, present] = True

        x_new = torch.cat([x_train_kept, x_test], dim=0)
        y_new = torch.cat([y_train_kept, y_test], dim=0)
        new_split = int(x_train_kept.shape[0])

        x_list.append(x_new)
        y_list.append(y_new)
        split_list.append(new_split)
        num_classes_list.append(num_classes)
        active_features_list.append(active_features)

    max_rows = max(x.shape[0] for x in x_list)
    max_features = max(x.shape[1] for x in x_list)

    x_batch = torch.zeros((batch_size, max_rows, max_features), dtype=torch.float32)
    y_batch = torch.full((batch_size, max_rows), fill_value=unseen_label, dtype=torch.long)
    row_mask = torch.zeros((batch_size, max_rows), dtype=torch.bool)

    for b in range(batch_size):
        row_len = x_list[b].shape[0]
        feat_len = x_list[b].shape[1]
        x_batch[b, :row_len, :feat_len] = x_list[b]
        y_batch[b, :row_len] = y_list[b]
        row_mask[b, :row_len] = True

    split_t = torch.tensor(split_list, dtype=torch.long)
    num_classes_t = torch.tensor(num_classes_list, dtype=torch.long)
    active_features_t = torch.tensor(active_features_list, dtype=torch.long)
    removed_class_mask_t = torch.from_numpy(removed_class_mask)
    seen_class_mask_t = torch.from_numpy(seen_class_mask)
    removed_class_count_t = removed_class_mask_t.sum(dim=1)
    removed_class_indices = [torch.where(removed_class_mask_t[b])[0] for b in range(batch_size)]

    if device is not None:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        row_mask = row_mask.to(device)
        split_t = split_t.to(device)
        num_classes_t = num_classes_t.to(device)
        active_features_t = active_features_t.to(device)
        removed_class_mask_t = removed_class_mask_t.to(device)
        seen_class_mask_t = seen_class_mask_t.to(device)
        removed_class_count_t = removed_class_count_t.to(device)
        removed_class_indices = [idx.to(device) for idx in removed_class_indices]

    return {
        "x": x_batch,
        "y": y_batch,
        "row_mask": row_mask,
        "train_test_split_index": split_t,
        "num_classes": num_classes_t,
        "num_features": active_features_t,
        "unseen_label": unseen_label,
        "removed_class_mask": removed_class_mask_t,
        "seen_class_mask": seen_class_mask_t,
        "removed_class_count": removed_class_count_t,
        "removed_class_indices": removed_class_indices,
    }


def generate_batch(
    cfg: PriorGeneratorConfig,
    batch_size: int,
    device: Optional[torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, torch.Tensor]:
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    if cfg.prior_backend == "legacy":
        legacy_cfg = LegacyPriorGeneratorConfig(
            max_classes=cfg.max_classes,
            min_features=cfg.min_features,
            max_features=cfg.max_features,
            min_rows=cfg.min_rows,
            max_rows=cfg.max_rows,
            min_train_fraction=cfg.min_train_fraction,
            max_train_fraction=cfg.max_train_fraction,
            remove_poisson_lambda=cfg.remove_poisson_lambda,
            x_mean=cfg.x_mean,
            x_std=cfg.x_std,
            label_noise=cfg.label_noise,
            seed=cfg.seed,
        )
        return generate_batch_legacy(legacy_cfg, batch_size=batch_size, device=device, rng=rng)

    if cfg.prior_backend == "tabicl":
        return _generate_batch_tabicl(cfg, batch_size=batch_size, device=device, rng=rng)

    raise ValueError(f"Unknown prior_backend='{cfg.prior_backend}'. Use 'tabicl' or 'legacy'.")
