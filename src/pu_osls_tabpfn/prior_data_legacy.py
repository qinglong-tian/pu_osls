from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class LegacyPriorGeneratorConfig:
    # Dataset shape ranges
    max_classes: int = 10  # inlier classes are 0..K-1; unseen_label = max_classes (reserved for padding)
    min_features: int = 3
    max_features: int = 8
    min_rows: int = 50
    max_rows: int = 100

    # Train/test split fraction (computed on the ORIGINAL R rows, before removals)
    min_train_fraction: float = 0.4
    max_train_fraction: float = 0.8

    # Class-removal distribution: truncated Poisson (renormalized)
    # P(remove k) ∝ Poisson(lambda) for k=0..max_removed, where max_removed <= K//2
    remove_poisson_lambda: float = 1.25

    # Feature distribution controls (simple Gaussian)
    x_mean: float = 0.0
    x_std: float = 1.0

    # Label generation: simple linear scoring + argmax over class logits
    # (kept simple; you can swap in TabICL-style SCM later)
    label_noise: float = 0.1

    # RNG seed
    seed: int = 0


def _sample_num_classes(rng: np.random.Generator, max_classes: int) -> int:
    # Ensure at least 2 classes (otherwise CE is awkward).
    return int(rng.integers(low=2, high=max_classes + 1))


def _sample_rows_features(rng: np.random.Generator, cfg: LegacyPriorGeneratorConfig) -> Tuple[int, int]:
    R = int(rng.integers(cfg.min_rows, cfg.max_rows + 1))
    F = int(rng.integers(cfg.min_features, cfg.max_features + 1))
    return R, F


def _sample_split_index(rng: np.random.Generator, R: int, cfg: LegacyPriorGeneratorConfig) -> int:
    frac = rng.uniform(cfg.min_train_fraction, cfg.max_train_fraction)
    idx = int(np.clip(round(frac * R), 1, R - 1))
    return idx


def _sample_num_removed(rng: np.random.Generator, K: int, lam: float) -> int:
    # remove in {0..max_removed}, where max_removed is at most half of classes
    max_removed = min(K - 1, K // 2)
    ks = np.arange(0, max_removed + 1)
    lam = max(lam, 1e-12)
    # Truncated Poisson distribution over ks (renormalized)
    log_probs = ks * math.log(lam) - np.array([math.lgamma(k + 1) for k in ks])
    probs = np.exp(log_probs - log_probs.max())
    probs /= probs.sum()
    return int(rng.choice(ks, p=probs))


def _make_labels_from_linear_model(
    rng: np.random.Generator, X: np.ndarray, K: int, noise: float
) -> np.ndarray:
    """
    Generate multiclass labels via:
      logits = X @ W + eps
      y = argmax logits
    """
    R, F = X.shape
    W = rng.normal(size=(F, K)).astype(np.float32)
    logits = X.astype(np.float32) @ W
    if noise > 0:
        logits = logits + rng.normal(scale=noise, size=logits.shape).astype(np.float32)
    y = logits.argmax(axis=1).astype(np.int64)
    return y


def generate_batch_legacy(
    cfg: LegacyPriorGeneratorConfig,
    batch_size: int,
    device: Optional[torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate ONE batch of synthetic tasks.

    Behavior:
      - Generate (X,y) with R rows, F features, K classes (K varies per task).
      - Choose a subset of classes 'removed' (subset of {0..K-1}) for each task.
      - In the TRAIN portion ONLY (first split rows of the original table),
        drop any rows whose label is in 'removed'.
      - TEST portion (rows >= original split) is kept unchanged with true labels.
      - Because row counts vary per task after dropping rows, we pad within batch
        to Rmax and return row_mask to identify real rows.

    Returns:
      {
        "x": (B,Rmax,F) float32
        "y": (B,Rmax)   int64  (true class ids for real rows; unseen_label for padding only)
        "row_mask": (B,Rmax) bool  (True for real rows)
        "train_test_split_index": (B,) int64  (split AFTER dropping train rows)
        "num_classes": (B,) int64   (K for each task)
        "unseen_label": int         (padding label id)
        "removed_class_mask": (B, max_classes) bool  (True where class id was removed-by-design)
        "seen_class_mask": (B, max_classes) bool     (True where class id appears in kept train rows)
        "removed_class_count": (B,) int64  (# removed classes per task)
        "removed_class_indices": list[Tensor]  (per-task indices of removed classes)
      }
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    # Shared (R,F) across the whole batch (simple, no feature padding)
    R, F = _sample_rows_features(rng, cfg)
    unseen_label = cfg.max_classes  # reserved id for padding rows only

    X_list = []
    y_list = []
    split_list = []
    K_list = []

    removed_class_mask = np.zeros((batch_size, cfg.max_classes), dtype=bool)
    seen_class_mask = np.zeros((batch_size, cfg.max_classes), dtype=bool)

    for b in range(batch_size):
        K = _sample_num_classes(rng, cfg.max_classes)
        split = _sample_split_index(rng, R, cfg)

        X = rng.normal(loc=cfg.x_mean, scale=cfg.x_std, size=(R, F)).astype(np.float32)
        y = _make_labels_from_linear_model(rng, X, K, cfg.label_noise)

        # Choose classes to remove from TRAIN portion only.
        # To keep training labels contiguous, remove the highest class ids.
        num_removed = _sample_num_removed(rng, K, cfg.remove_poisson_lambda)
        removed: set[int] = set()
        if num_removed > 0:
            removed = set(range(K - num_removed, K))

        # Record removed-by-design
        if removed:
            removed_class_mask[b, list(removed)] = True

        # Split original into train/test
        X_train = X[:split]
        y_train = y[:split]
        X_test = X[split:]
        y_test = y[split:]

        # Drop removed-class rows from TRAIN
        if removed:
            keep_mask_train = ~np.isin(y_train, list(removed))
            X_train_kept = X_train[keep_mask_train]
            y_train_kept = y_train[keep_mask_train]
        else:
            X_train_kept = X_train
            y_train_kept = y_train

        # Record which classes actually appear in kept train
        if y_train_kept.size > 0:
            present = np.unique(y_train_kept)
            seen_class_mask[b, present] = True

        # Reassemble; new split is the kept-train length
        X_new = np.concatenate([X_train_kept, X_test], axis=0)
        y_new = np.concatenate([y_train_kept, y_test], axis=0)
        new_split = int(X_train_kept.shape[0])

        X_list.append(X_new)
        y_list.append(y_new)
        split_list.append(new_split)
        K_list.append(K)

    # Pad to max rows within this batch so we can stack tensors
    Rmax = max(x.shape[0] for x in X_list)

    X_batch = np.zeros((batch_size, Rmax, F), dtype=np.float32)
    y_batch = np.full((batch_size, Rmax), fill_value=unseen_label, dtype=np.int64)  # padding label
    row_mask = np.zeros((batch_size, Rmax), dtype=bool)

    for b in range(batch_size):
        rr = X_list[b].shape[0]
        X_batch[b, :rr] = X_list[b]
        y_batch[b, :rr] = y_list[b]
        row_mask[b, :rr] = True

    x_t = torch.from_numpy(X_batch)  # (B,Rmax,F)
    y_t = torch.from_numpy(y_batch)  # (B,Rmax)
    row_mask_t = torch.from_numpy(row_mask)  # (B,Rmax)
    split_t = torch.tensor(split_list, dtype=torch.long)  # (B,)
    K_t = torch.tensor(K_list, dtype=torch.long)  # (B,)

    removed_class_mask_t = torch.from_numpy(removed_class_mask)  # (B,max_classes)
    seen_class_mask_t = torch.from_numpy(seen_class_mask)  # (B,max_classes)
    removed_class_count_t = removed_class_mask_t.sum(dim=1)  # (B,)
    removed_class_indices = [
        torch.where(removed_class_mask_t[b])[0] for b in range(batch_size)
    ]

    if device is not None:
        x_t = x_t.to(device)
        y_t = y_t.to(device)
        row_mask_t = row_mask_t.to(device)
        split_t = split_t.to(device)
        K_t = K_t.to(device)
        removed_class_mask_t = removed_class_mask_t.to(device)
        seen_class_mask_t = seen_class_mask_t.to(device)
        removed_class_count_t = removed_class_count_t.to(device)
        removed_class_indices = [idx.to(device) for idx in removed_class_indices]

    return {
        "x": x_t,
        "y": y_t,
        "row_mask": row_mask_t,
        "train_test_split_index": split_t,  # (B,)
        "num_classes": K_t,                 # (B,)
        "unseen_label": unseen_label,       # int
        "removed_class_mask": removed_class_mask_t,  # (B,max_classes)
        "seen_class_mask": seen_class_mask_t,        # (B,max_classes)
        "removed_class_count": removed_class_count_t,  # (B,)
        "removed_class_indices": removed_class_indices,  # list[Tensor], each (num_removed,)
    }


# Backward-compatible aliases.
PriorGeneratorConfig = LegacyPriorGeneratorConfig
generate_batch = generate_batch_legacy
