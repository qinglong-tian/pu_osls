# eval_pu_osls.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
)


@dataclass
class EvalConfig:
    n_tasks: int = 200
    batch_size: int = 8
    seed: int = 123
    outlier_score: str = "msp"  # {"msp", "entropy", "p_unseen"}
    fpr_targets: Tuple[float, ...] = (0.01, 0.05, 0.10)


def _entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    if y_true.sum() == 0 or (y_true == 0).sum() == 0:
        return np.nan
    order = np.argsort(scores)[::-1]
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    fpr_curve = fp / max(N, 1)
    tpr_curve = tp / max(P, 1)
    ok = np.where(fpr_curve <= fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(tpr_curve[ok].max())


def _build_padded_y_train(
    y_full: torch.Tensor, split: torch.Tensor, unseen_label: int
) -> torch.Tensor:
    B = y_full.shape[0]
    y_train_list = []
    for b in range(B):
        split_b = int(split[b].item())
        y_train_list.append(y_full[b:b + 1, :split_b])
    max_split = int(split.max().item())
    y_train_padded = torch.full(
        (B, max_split),
        fill_value=unseen_label,
        device=y_full.device,
        dtype=y_full.dtype,
    )
    for b, ytb in enumerate(y_train_list):
        y_train_padded[b, : ytb.shape[1]] = ytb
    return y_train_padded


def _slice_logits_for_seen(
    logits: torch.Tensor,
    seen_counts: torch.Tensor,
) -> torch.Tensor:
    B, max_test, _ = logits.shape
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
        logits_sliced[b, :, seen_count] = logits[b, :, -1]
    return logits_sliced


@torch.no_grad()
def evaluate_pu_osls(
    model,
    *,
    cfg_prior,
    unseen_label: int,
    device: torch.device,
    eval_cfg: Optional[EvalConfig] = None,
) -> Dict[str, float]:
    from .prior_data import generate_batch

    if eval_cfg is None:
        eval_cfg = EvalConfig()

    was_training = model.training
    model = model.to(device)
    model.eval()

    rng = np.random.default_rng(eval_cfg.seed)

    all_outlier_true: List[int] = []
    all_outlier_score: List[float] = []
    all_seen_true: List[int] = []
    all_seen_pred: List[int] = []
    all_test_true: List[int] = []
    all_test_pred: List[int] = []

    n_done = 0
    while n_done < eval_cfg.n_tasks:
        B = min(eval_cfg.batch_size, eval_cfg.n_tasks - n_done)
        batch = generate_batch(cfg_prior, batch_size=B, device=device, rng=rng)
        x = batch["x"]
        y_full = batch["y"]
        split = batch["train_test_split_index"]
        row_mask = batch["row_mask"]
        seen_class_mask = batch["seen_class_mask"]
        removed_class_mask = batch["removed_class_mask"]
        num_classes = batch["num_classes"]

        seen_counts = seen_class_mask.sum(dim=1).to(torch.long)

        y_train_padded = _build_padded_y_train(y_full, split, unseen_label)
        logits = model((x, y_train_padded), split)  # (B, max_test, num_outputs)
        logits_sliced = _slice_logits_for_seen(logits, seen_counts)

        probs = F.softmax(logits_sliced, dim=-1)
        pred = probs.argmax(dim=-1)

        for b in range(B):
            split_b = int(split[b].item())
            row_count = int(row_mask[b].sum().item())
            y_test_true = y_full[b, split_b:row_count]
            if y_test_true.numel() == 0:
                continue

            seen_mask_b = seen_class_mask[b]
            removed_mask_b = removed_class_mask[b]
            seen_count = int(seen_counts[b].item())

            outlier_true = (~seen_mask_b[y_test_true]).to(torch.long)
            probs_b = probs[b, : y_test_true.shape[0], : seen_count + 1]

            if eval_cfg.outlier_score == "msp":
                score = 1.0 - probs_b.max(dim=-1).values
            elif eval_cfg.outlier_score == "entropy":
                score = _entropy(probs_b)
            elif eval_cfg.outlier_score == "p_unseen":
                score = probs_b[:, seen_count]
            else:
                raise ValueError(f"Unknown outlier_score={eval_cfg.outlier_score}")

            all_outlier_true.extend(outlier_true.detach().cpu().numpy().astype(int).tolist())
            all_outlier_score.extend(score.detach().cpu().numpy().astype(float).tolist())

            pred_b = pred[b, : y_test_true.shape[0]]
            mask_inlier = (outlier_true == 0)
            if mask_inlier.any():
                all_seen_true.extend(y_test_true[mask_inlier].detach().cpu().numpy().astype(int).tolist())
                all_seen_pred.extend(pred_b[mask_inlier].detach().cpu().numpy().astype(int).tolist())

            y_test_mapped = torch.where(
                removed_mask_b[y_test_true],
                torch.tensor(unseen_label, device=device),
                y_test_true,
            )
            pred_mapped = torch.where(
                pred_b == seen_count,
                torch.tensor(unseen_label, device=device),
                pred_b,
            )
            all_test_true.extend(y_test_mapped.detach().cpu().numpy().astype(int).tolist())
            all_test_pred.extend(pred_mapped.detach().cpu().numpy().astype(int).tolist())

        n_done += B

    y_out = np.asarray(all_outlier_true, dtype=np.int64)
    s_out = np.asarray(all_outlier_score, dtype=np.float64)
    outlier_auc = np.nan
    outlier_ap = np.nan
    if y_out.sum() > 0 and (y_out == 0).sum() > 0:
        outlier_auc = float(roc_auc_score(y_out, s_out))
        outlier_ap = float(average_precision_score(y_out, s_out))

    seen_acc = np.nan
    seen_bal_acc = np.nan
    if len(all_seen_true) > 0:
        y_seen = np.asarray(all_seen_true, dtype=np.int64)
        p_seen = np.asarray(all_seen_pred, dtype=np.int64)
        seen_acc = float(accuracy_score(y_seen, p_seen))
        seen_bal_acc = float(balanced_accuracy_score(y_seen, p_seen))

    y_all = np.asarray(all_test_true, dtype=np.int64)
    p_all = np.asarray(all_test_pred, dtype=np.int64)
    overall_acc = float(accuracy_score(y_all, p_all)) if len(y_all) > 0 else np.nan
    overall_bal_acc = float(balanced_accuracy_score(y_all, p_all)) if len(y_all) > 0 else np.nan

    tpr_at = {}
    for fpr in eval_cfg.fpr_targets:
        tpr_at[f"tpr@fpr={fpr:.2f}"] = _tpr_at_fpr(y_out, s_out, fpr)

    if was_training:
        model.train()

    results: Dict[str, float] = {
        "n_tasks": float(eval_cfg.n_tasks),
        "outlier_auc": outlier_auc,
        "outlier_ap": outlier_ap,
        "seen_acc": seen_acc,
        "seen_bal_acc": seen_bal_acc,
        "overall_acc": overall_acc,
        "overall_bal_acc": overall_bal_acc,
        "outlier_rate_test": float(y_out.mean()) if len(y_out) > 0 else np.nan,
    }
    results.update({k: float(v) for k, v in tpr_at.items()})
    return results


def print_results(res: Dict[str, float]) -> None:
    keys = [
        "n_tasks",
        "outlier_rate_test",
        "outlier_auc",
        "outlier_ap",
        "tpr@fpr=0.01",
        "tpr@fpr=0.05",
        "tpr@fpr=0.10",
        "seen_acc",
        "seen_bal_acc",
        "overall_acc",
        "overall_bal_acc",
    ]
    for k in keys:
        if k in res:
            v = res[k]
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                print(f"{k:18s}: {v}")
            elif isinstance(v, float):
                print(f"{k:18s}: {v:0.4f}")
            else:
                print(f"{k:18s}: {v}")
