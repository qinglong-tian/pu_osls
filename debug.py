import numpy as np

from pu_osls_tabpfn.prior_data import PriorGeneratorConfig, _generate_batch_tabicl


def main() -> None:
    cfg = PriorGeneratorConfig(
        seed=58,
        min_rows=20,
        max_rows=30,
        min_features=3,
        max_features=6,
        max_classes=5,
        prior_backend="tabicl",
    )
    rng = np.random.default_rng(cfg.seed)

    batch = _generate_batch_tabicl(
        cfg,
        batch_size=4,
        device=None,
        rng=rng,
    )

    x_batch = batch["x"]
    y_batch = batch["y"]
    row_mask = batch["row_mask"]
    split_t = batch["train_test_split_index"]
    num_classes_t = batch["num_classes"]
    num_features_t = batch["num_features"]
    unseen_label = int(batch["unseen_label"])

    b = 0
    row_count = int(row_mask[b].sum().item())
    split = int(split_t[b].item())
    active_features = int(num_features_t[b].item())

    assert 1 <= split < row_count, f"Invalid split: split={split}, rows={row_count}"
    assert row_count <= x_batch.shape[1], "row_count exceeds padded rows"
    assert active_features <= x_batch.shape[2], "active feature count exceeds padded features"

    for i in range(x_batch.shape[0]):
        row_count_i = int(row_mask[i].sum().item())
        split_i = int(split_t[i].item())
        feat_i = int(num_features_t[i].item())
        assert 1 <= split_i < row_count_i, f"Invalid split for item {i}: split={split_i}, rows={row_count_i}"
        assert feat_i <= x_batch.shape[2], f"Feature count out of bounds for item {i}: {feat_i}"
        assert (y_batch[i, row_count_i:] == unseen_label).all(), f"Unexpected padded label values for item {i}"

    x_active = x_batch[b, :row_count, :active_features].cpu().numpy()
    y_active = y_batch[b, :row_count].cpu().numpy()
    x_train = x_active[:split]
    x_test = x_active[split:]
    y_train = y_active[:split]
    y_test = y_active[split:]

    np.set_printoptions(linewidth=160, precision=4, suppress=True)
    print("=== Metadata ===")
    print("x_batch.shape:", tuple(x_batch.shape))
    print("y_batch.shape:", tuple(y_batch.shape))
    print("row_mask.shape:", tuple(row_mask.shape))
    print("splits:", split_t.tolist())
    print("num_classes per item:", num_classes_t.tolist())
    print("active_features per item:", num_features_t.tolist())
    print("unseen_label:", unseen_label)
    print("row_count (item 0):", row_count)
    print("split (item 0):", split)
    print("active_features (item 0):", active_features)
    print("num_classes (item 0, from y):", int(np.unique(y_active).size))

    print("\n=== Full Active Dataset (X) ===")
    print(x_active)
    print("\n=== Full Active Labels (y) ===")
    print(y_active)

    print("\n=== Train Split ===")
    print("X_train:")
    print(x_train)
    print("y_train:")
    print(y_train)
    print("train class counts:", {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))})

    print("\n=== Test Split ===")
    print("X_test:")
    print(x_test)
    print("y_test:")
    print(y_test)
    print("test class counts:", {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))})

    print("\nSanity checks passed for _generate_batch_tabicl.")


if __name__ == "__main__":
    main()
