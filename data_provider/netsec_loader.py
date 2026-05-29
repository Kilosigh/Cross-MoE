"""
Network security anomaly detection dataloaders with text modality support.

Supports:
  - UNSWNB15: UNSW-NB15 dataset with BERT-encoded text descriptions
  - EdgeIIoT: Edge-IIoTset dataset (to be added)
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class NetSecSegLoader(Dataset):
    """
    Generic network security time-series dataloader with text and classification support.

    Data directory structure:
      data/{dataset_name}/
        train.csv       - training data (features + label + attack_type + text_idx)
        test.csv        - test data
        text_emb.pt     - pre-computed BERT text embeddings

    Returns per sample:
      seq_x: [win_size, n_features]  - time series window
      label: int (0=normal, 1=attack) - binary classification label
      text_emb: [1, bert_dim]         - pre-computed BERT embedding for this sample's text
    """

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.args = args

        # Load data
        data_path = os.path.join(root_path, f'{flag}.csv')
        if not os.path.exists(data_path):
            # Try alternative: train/val split from train.csv
            if flag == 'val':
                data_path = os.path.join(root_path, 'train.csv')
                self._is_val_split = True
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")
        else:
            self._is_val_split = False

        df = pd.read_csv(data_path)

        # Separate features, labels, and text indices
        feature_cols = [c for c in df.columns if c.startswith('f')]
        if not feature_cols:
            # Auto-detect numerical columns
            feature_cols = []
            for c in df.columns:
                if c not in ['label', 'attack_type', 'text_idx']:
                    try:
                        pd.to_numeric(df[c].iloc[0])
                        feature_cols.append(c)
                    except:
                        pass

        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.int64)
        self.text_indices = df['text_idx'].values.astype(np.int64) if 'text_idx' in df.columns else None

        # Load pre-computed text embeddings
        text_emb_path = os.path.join(root_path, 'text_emb.pt')
        if os.path.exists(text_emb_path):
            self.text_embeddings = torch.load(text_emb_path, weights_only=True)
        else:
            self.text_embeddings = None

        # Scale features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # Handle NaN
        self.features = np.nan_to_num(self.features, nan=0.0)

        # Val split: take last 20% of training data
        if self._is_val_split:
            n_val = int(len(self.features) * 0.2)
            self.features = self.features[-n_val:]
            self.labels = self.labels[-n_val:]
            if self.text_indices is not None:
                self.text_indices = self.text_indices[-n_val:]

        print(f"{flag}: {len(self.features)} samples, "
              f"attack ratio: {self.labels.mean():.3f}")

    def __len__(self):
        return (len(self.features) - self.win_size) // self.step + 1

    def __getitem__(self, index):
        idx = index * self.step

        # Time series window
        seq_x = torch.from_numpy(self.features[idx:idx + self.win_size])

        # Label: use the label of the last time step in the window
        # (or majority vote for the window)
        window_labels = self.labels[idx:idx + self.win_size]
        label = int(np.median(window_labels))  # use median to handle edge cases

        # Text embedding
        if self.text_embeddings is not None and self.text_indices is not None:
            text_idx = self.text_indices[idx + self.win_size - 1]  # text for last timestep
            text_emb = self.text_embeddings[text_idx].squeeze(0)   # [bert_dim]
        else:
            text_emb = torch.zeros(768)  # BERT-base dim fallback

        return seq_x, label, text_emb


class UNSWNB15SegLoader(NetSecSegLoader):
    """UNSW-NB15 specific loader."""
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)
        super().__init__(args, root_path, win_size, step, flag)


class EdgeIIoTSegLoader(NetSecSegLoader):
    """Edge-IIoTset specific loader (placeholder)."""
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)
        super().__init__(args, root_path, win_size, step, flag)
