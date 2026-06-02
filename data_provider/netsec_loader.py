"""
UNSW-NB15 anomaly detection dataloader — follows Time-Series-Library format.

- Unimodal:  returns (batch_x [win_size, n_ts_features], batch_y)
- Multimodal: returns (batch_x [win_size, n_ts_features + n_text_features], batch_y)
  where text features are PCA-projected BERT embeddings replicated across time.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


N_TEXT_DIM = 8  # PCA target dimension for text features


class UNSWNB15Loader(Dataset):
    """
    args.use_text:  0 = unimodal (TS only)
                    1 = multimodal (TS + PCA text channels)
    """
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.use_text = getattr(args, 'use_text', 0)

        # --- load data ---
        train_df = pd.read_csv(os.path.join(root_path, "train.csv"))
        test_df  = pd.read_csv(os.path.join(root_path, "test.csv"))
        label_df = pd.read_csv(os.path.join(root_path, "test_label.csv"))

        # TS feature columns
        feat_cols = [c for c in train_df.columns if c.startswith('f')]

        self.scaler = StandardScaler()
        train_ts = np.nan_to_num(train_df[feat_cols].values.astype(np.float32))
        self.scaler.fit(train_ts)
        self.train_ts = self.scaler.transform(train_ts)

        test_ts = np.nan_to_num(test_df[feat_cols].values.astype(np.float32))
        self.test_ts = self.scaler.transform(test_ts)
        self.test_labels = label_df.values[:, 1:].astype(int)  # skip index col

        # --- text features (optional) ---
        if self.use_text:
            text_emb = torch.load(os.path.join(root_path, "text_emb.pt"), weights_only=False)
            # text_emb: [N_types, 768] → PCA → [N_types, N_TEXT_DIM]
            text_np = text_emb.numpy()

            # Fit PCA on training attack-type distribution
            self.text_pca = PCA(n_components=N_TEXT_DIM, random_state=42)
            text_pca = self.text_pca.fit_transform(text_np)  # [N_types, 8]

            # Map each sample to its text embedding
            train_text_idx = train_df['text_idx'].values.astype(int)
            test_text_idx  = test_df['text_idx'].values.astype(int)
            train_text_np = text_pca[train_text_idx]  # [N_train, 8]
            test_text_np  = text_pca[test_text_idx]    # [N_test, 8]

            # Concatenate TS + text features
            self.train_data = np.concatenate([self.train_ts, train_text_np], axis=1)
            self.test_data  = np.concatenate([self.test_ts,  test_text_np],  axis=1)
        else:
            self.train_data = self.train_ts
            self.test_data  = self.test_ts

        # Val split: last 20% of train
        n_val = int(len(self.train_data) * 0.2)
        self.val_data = self.train_data[-n_val:]
        self.train_data = self.train_data[:-n_val]

        print(f"{flag}: train {self.train_data.shape}, val {self.val_data.shape}, "
              f"test {self.test_data.shape}, text={'Y' if self.use_text else 'N'}")

    def _select(self):
        if self.flag == "train":
            return self.train_data, np.zeros((len(self.train_data), 1))
        elif self.flag == "val":
            return self.val_data, np.zeros((len(self.val_data), 1))
        else:
            return self.test_data, self.test_labels

    def __len__(self):
        data, _ = self._select()
        return (len(data) - self.win_size) // self.step + 1

    def __getitem__(self, index):
        data, labels = self._select()
        idx = index * self.step
        x = data[idx:idx + self.win_size]
        y = labels[idx:idx + self.win_size]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
