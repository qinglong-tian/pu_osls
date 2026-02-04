import numpy as np
import torch

from pu_osls_tabpfn.prior_data import PriorGeneratorConfig, generate_batch


def _assert_common_batch_shape(batch, batch_size: int) -> None:
    x = batch["x"]
    y = batch["y"]
    row_mask = batch["row_mask"]
    split = batch["train_test_split_index"]

    assert x.shape[0] == batch_size
    assert y.shape[:2] == x.shape[:2]
    assert row_mask.shape == y.shape
    assert split.shape[0] == batch_size
    assert row_mask.dtype == torch.bool


def test_generate_batch_shapes_and_masks_tabicl_backend():
    cfg = PriorGeneratorConfig(
        seed=7,
        min_rows=20,
        max_rows=25,
        min_features=3,
        max_features=6,
        prior_backend="tabicl",
    )
    batch = generate_batch(cfg, batch_size=4, rng=np.random.default_rng(cfg.seed))
    _assert_common_batch_shape(batch, batch_size=4)


def test_generate_batch_shapes_and_masks_legacy_backend():
    cfg = PriorGeneratorConfig(
        seed=7,
        min_rows=20,
        max_rows=25,
        min_features=3,
        max_features=6,
        prior_backend="legacy",
    )
    batch = generate_batch(cfg, batch_size=4, rng=np.random.default_rng(cfg.seed))
    _assert_common_batch_shape(batch, batch_size=4)
