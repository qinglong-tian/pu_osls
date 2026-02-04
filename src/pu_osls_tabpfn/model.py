"""
model.py
========

This module contains an adaptation of the lightweight TabPFN reimplementation
from the `nanoTabPFN` project.  The original implementation embeds
tabular features and target labels and applies a transformer encoder to
perform in‑context classification.  The architecture is designed to
approximate Bayesian inference on small tabular datasets.

The key modification in this file is a custom target encoder that
imputes missing labels for the test portion of each dataset using a
strategy suitable for positive–unlabeled (PU) learning and open‑set
label shift (OSLS).  In PU/OSLS scenarios the training labels may be
biased towards a subset of classes, so taking the mean of the training
labels (as done in the original implementation) is inappropriate.  Our
``CustomTargetEncoder`` instead uses a dedicated learnable "unknown"
embedding for the test portion, which avoids collapsing to a single
class value when the training labels are all positive.

The rest of the architecture (feature encoder, transformer encoder and
decoder) remains unchanged from the original implementation.  For
completeness we reproduce these components here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm


class FeatureEncoder(nn.Module):
    """Encodes scalar feature values into embeddings.

    The encoder normalises each feature based on the mean and standard
    deviation of the training data, clips extreme values to the range
    [‑100, 100] and then applies a linear layer to map the scalar to an
    ``embedding_size``‑dimensional vector.
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        # x has shape (batch_size, num_rows, num_features)
        # Add a singleton dimension for each scalar so that the linear layer can be applied
        x = x.unsqueeze(-1)  # (B,R,C-1,1)
        # Compute per‑dataset mean and std from the training portion
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdim=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdim=True, unbiased=False) + 1e-20
        x = (x - mean) / std
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        x = torch.clamp(x, min=-100.0, max=100.0)
        return self.linear_layer(x)


class CustomTargetEncoder(nn.Module):
    def __init__(self, embedding_size: int, unseen_label: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)
        # Learnable embedding used to impute unknown test labels (PU/OSLS safe).
        self.unknown_label_embedding = nn.Parameter(torch.zeros(embedding_size))
        self.unseen_label = unseen_label

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Input:  y_train shape (B, R_train) or (B, R_train, 1)
        Output: y_emb   shape (B, R, 1, E)
        """
        # Ensure (B, R_train, 1)
        if y_train.dim() == 2:
            y_train = y_train.unsqueeze(-1)
        elif y_train.dim() == 3 and y_train.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"y_train must be (B,R_train) or (B,R_train,1), got {tuple(y_train.shape)}")

        B, R_train, _ = y_train.shape
        if num_rows < R_train:
            raise ValueError("num_rows must be >= number of training rows")

        # Embed training labels
        y_train_float = y_train.float()
        y_train_emb = self.linear_layer(y_train_float)  # (B,R_train,E)

        # Pad to full length using a learnable "unknown" embedding: (B, R, E)
        pad_len = num_rows - R_train
        if pad_len > 0:
            padding = self.unknown_label_embedding.view(1, 1, -1).repeat(B, pad_len, 1)
            y_full = torch.cat([y_train_emb, padding], dim=1)  # (B,R,E)
        else:
            y_full = y_train_emb

        # Restore the singleton target-column dimension -> (B,R,1,E)
        y_emb = y_full.unsqueeze(2)
        return y_emb

class TransformerEncoderLayer(nn.Module):
    """A transformer encoder block with separate attention over features and data points.

    This implementation follows the structure described in the TabPFN paper,
    combining self‑attention between features and self‑attention between
    data points, followed by a two‑layer feedforward network with GELU
    activation and residual connections.
    """

    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)
        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape
        # Attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # Attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        # Training data attends to itself
        src_left = self.self_attention_between_datapoints(
            src[:, :train_test_split_index], src[:, :train_test_split_index], src[:, :train_test_split_index]
        )[0]
        # Test data attends to training data
        src_right = self.self_attention_between_datapoints(
            src[:, train_test_split_index:], src[:, :train_test_split_index], src[:, :train_test_split_index]
        )[0]
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # Feedforward
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src


class Decoder(nn.Module):
    """Maps transformer embeddings to class logits."""
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class CustomNanoTabPFNModel(nn.Module):
    """TabPFN model with custom target encoder for PU/OSLS scenarios."""
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int,
                 num_layers: int, num_outputs: int, unseen_label: int):
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = CustomTargetEncoder(embedding_size, unseen_label)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size))
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
        self.unseen_label = unseen_label

    def forward(
        self,
        src: tuple[torch.Tensor, torch.Tensor],
        train_test_split_index: int | torch.Tensor,
    ) -> torch.Tensor:
        x_src, y_src = src
        # x_src shape: (B, R, C) and y_src shape: (B, R) or (B, R, 1) or (B, R_train)
        if y_src.dim() < x_src.dim():
            y_src = y_src.unsqueeze(-1)

        if isinstance(train_test_split_index, torch.Tensor):
            # Per-task split indices in batch; process each task independently and pad logits.
            splits = train_test_split_index.to(torch.long)
            B, num_rows, _ = x_src.shape
            max_test = int(num_rows - splits.min().item())
            logits_batch = []
            for b in range(B):
                split_b = int(splits[b].item())
                x_b = x_src[b:b+1]
                y_b = y_src[b:b+1]
                # Use only the training part for target encoding
                y_train_b = y_b[:, :split_b]
                # Encode features: (1, R, C-1, E)
                x_enc = self.feature_encoder(x_b, split_b)
                # Encode targets: (1, R, 1, E)
                y_enc = self.target_encoder(y_train_b, num_rows)
                # Concatenate along feature dimension: (1, R, C, E)
                src_b = torch.cat([x_enc, y_enc], dim=2)
                for block in self.transformer_blocks:
                    src_b = block(src_b, train_test_split_index=split_b)
                output_embeddings = src_b[:, split_b:, -1, :]  # (1, test_len, E)
                logits_b = self.decoder(output_embeddings).squeeze(0)  # (test_len, C)
                if logits_b.shape[0] < max_test:
                    pad = torch.zeros(
                        (max_test - logits_b.shape[0], logits_b.shape[1]),
                        device=logits_b.device,
                        dtype=logits_b.dtype,
                    )
                    logits_b = torch.cat([logits_b, pad], dim=0)
                logits_batch.append(logits_b)
            return torch.stack(logits_batch, dim=0)  # (B, max_test, C)

        # Single split index (shared across batch)
        split = int(train_test_split_index)
        # Encode features: returns (B, R, C-1, E)
        x_src = self.feature_encoder(x_src, split)
        num_rows = x_src.shape[1]
        # Encode targets: returns (B, R, 1, E)
        y_src = self.target_encoder(y_src[:, :split], num_rows)
        # Concatenate along feature dimension: (B, R, C, E)
        src = torch.cat([x_src, y_src], dim=2)
        # Apply transformer blocks
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=split)
        # Select embeddings corresponding to the target column and test rows
        output_embeddings = src[:, split:, -1, :]  # (B, num_test_rows, E)
        # Compute logits
        logits = self.decoder(output_embeddings)
        return logits


class CustomNanoTabPFNClassifier:
    """A scikit‑learn–like wrapper around ``CustomNanoTabPFNModel``."""
    def __init__(self, model: CustomNanoTabPFNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.num_classes: int | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Store training data for later predictions
        self.X_train = X_train
        self.y_train = y_train
        # Determine number of classes from contiguous labels 0..K-1
        seen_labels = np.unique(y_train.astype(int))
        if seen_labels.size == 0 or seen_labels[0] != 0 or not np.array_equal(
            seen_labels, np.arange(seen_labels[-1] + 1)
        ):
            raise ValueError("Training labels must be contiguous and start at 0.")
        self.num_classes = int(seen_labels[-1]) + 1

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        # Stack training and test features
        x = np.concatenate((self.X_train, X_test), axis=0)
        y = self.y_train
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y_tensor = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            logits = self.model((x_tensor, y_tensor), train_test_split_index=len(self.X_train)).squeeze(0)
            # Keep logits for seen classes plus unseen label (last index of full head)
            unseen_logit = logits[:, -1:]
            logits = torch.cat([logits[:, :self.num_classes], unseen_logit], dim=1)
            probabilities = F.softmax(logits, dim=1)
            return probabilities.cpu().numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X_test)
        return proba.argmax(axis=1)
