"""
Multimodal network security anomaly detection experiment.

Two modes:
  classify:    TS encoder + BERT text → MLP → binary (Normal/Attack)  [SOTA]
  reconstruct: TS encoder-decoder reconstruction error → threshold     [baseline]

Usage:
  python run.py --task_name anomaly_detection --data UNSWNB15 \
      --detect_mode classify --use_text 1 --model PatchTST ...
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import os
import time
import warnings
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                              roc_auc_score, confusion_matrix)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from exp.exp_model_dict import model_dict

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
#  Lightweight multimodal anomaly classifier
# ---------------------------------------------------------------------------

class AnomalyClassifier(nn.Module):
    """
    Standalone 1D-CNN encoder + BERT text → concat → MLP → binary class.

    Does NOT depend on base TS model internals → works identically on Linux.
    """
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.bert_dim = args.llm_dim
        self.use_text = args.use_text
        in_channels = args.enc_in

        # Standalone 1D-CNN TS encoder (avoids base-model interface differences)
        self.ts_encoder = nn.Sequential(
            nn.Conv1d(in_channels, self.d_model // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.d_model // 2),
            nn.GELU(),
            nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),     # → [B, d_model, 1]
            nn.Flatten(start_dim=1),     # → [B, d_model]
        )

        # Text projector
        hidden = max(self.bert_dim // 4, 64)
        self.text_proj = nn.Sequential(
            nn.Linear(self.bert_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, self.d_model),
        )

        # Classifier
        in_dim = self.d_model * 2 if self.use_text else self.d_model
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.d_model // 2, 2),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_enc, text_emb):
        """
        x_enc:    [B, seq_len, n_features]
        text_emb: [B, bert_dim]
        returns:  [B, 2] logits
        """
        # 1) TS encoding: [B, L, C] → [B, C, L] → CNN → [B, d_model]
        x = x_enc.permute(0, 2, 1)       # [B, C, L]
        ts_feat = self.ts_encoder(x)     # [B, d_model]

        # 2) Text projection
        txt_feat = self.text_proj(text_emb)  # [B, d_model]

        # 3) Fusion
        if self.use_text:
            fused = torch.cat([ts_feat, txt_feat], dim=-1)
        else:
            fused = ts_feat

        return self.classifier(fused)


# ---------------------------------------------------------------------------
#  Experiment
# ---------------------------------------------------------------------------

class Exp_Anomaly_Detection_MoE(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    # ---- build -----------------------------------------------------------
    def _build_model(self):
        if self.args.detect_mode == 'classify':
            model = AnomalyClassifier(self.args)
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            return model.float()
        else:
            # reconstruction mode: use base TS model only
            model = model_dict[self.args.model].Model(self.args).float()
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.args.detect_mode == 'classify':
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    # ---- classification training -----------------------------------------
    def _classify_step(self, loader, optimizer=None):
        """Single epoch train/val step for classification."""
        total_loss, correct, total = 0.0, 0, 0
        is_train = optimizer is not None

        for seq_x, label, text_emb in loader:
            seq_x = seq_x.float().to(self.device)
            label = label.long().to(self.device)
            text_emb = text_emb.float().to(self.device)

            logits = self.model(seq_x, text_emb)
            loss = self._criterion(logits, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def train(self, setting):
        if self.args.detect_mode != 'classify':
            return self._train_reconstruct(setting)

        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        self._criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        stopper = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss, train_acc = self._classify_step(train_loader, optimizer)

            self.model.eval()
            with torch.no_grad():
                val_loss, val_acc = self._classify_step(vali_loader)
                test_loss, test_acc = self._classify_step(test_loader)

            print(f"Epoch {epoch+1:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val acc {val_acc:.4f} | test acc {test_acc:.4f}")

            stopper(val_loss, self.model, path)
            if stopper.early_stop:
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        best = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best, map_location='cpu', weights_only=False))
        return self.model

    # ---- classification test ---------------------------------------------
    def test(self, setting, test=0):
        if self.args.detect_mode != 'classify':
            return self._test_reconstruct(setting, test)

        _, test_loader = self._get_data('test')
        ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, weights_only=False))

        all_preds, all_labels, all_probs = [], [], []
        self.model.eval()
        with torch.no_grad():
            for seq_x, label, text_emb in test_loader:
                seq_x = seq_x.float().to(self.device)
                text_emb = text_emb.float().to(self.device)
                logits = self.model(seq_x, text_emb)
                probs = F.softmax(logits, dim=-1)[:, 1]

                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(label.tolist())
                all_probs.extend(probs.cpu().tolist())

        preds, labels, probs = map(np.array, [all_preds, all_labels, all_probs])
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        auc = roc_auc_score(labels, probs)
        cm = confusion_matrix(labels, preds)

        print(f"\n{'='*55}")
        print(f"  Classification  |  {setting}")
        print(f"{'='*55}")
        print(f"  Acc {acc:.4f}  Prec {prec:.4f}  Rec {rec:.4f}  F1 {f1:.4f}  AUC {auc:.4f}")
        print(f"  TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
        print(f"  FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")
        print(f"{'='*55}\n")

        with open('result_anomaly_detection_MoE.txt', 'a') as f:
            f.write(f"{setting} | mode=classify | model={self.args.model}\n")
            f.write(f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}\n\n")
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}

    # ---- reconstruction (fallback) ---------------------------------------
    def _train_reconstruct(self, setting):
        """Standard reconstruction-based anomaly detection (from existing code)."""
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        stopper = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = self._select_optimizer()
        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            for seq_x, _, _ in train_loader:
                seq_x = seq_x.float().to(self.device)
                optimizer.zero_grad()
                outputs = self.model(seq_x, None, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, seq_x)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            self.model.eval()
            val_loss = self._reconstruct_loss(vali_loader)
            test_loss = self._reconstruct_loss(test_loader)
            print(f"Epoch {epoch+1:3d} | train {train_loss:.4f} val {val_loss:.4f} test {test_loss:.4f}")

            stopper(val_loss, self.model, path)
            if stopper.early_stop:
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth'), weights_only=False))
        return self.model

    def _reconstruct_loss(self, loader):
        loss = []
        criterion = nn.MSELoss()
        for seq_x, _, _ in loader:
            seq_x = seq_x.float().to(self.device)
            outputs = self.model(seq_x, None, None, None)
            f_dim = -1 if self.args.features == 'MS' else 0
            loss.append(criterion(outputs[:, :, f_dim:], seq_x).item())
        return np.average(loss)

    def _test_reconstruct(self, setting, test=0):
        """Standard reconstruction-based test (from existing code)."""
        test_data, test_loader = self._get_data('test')
        train_data, train_loader = self._get_data('train')

        ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, weights_only=False))

        self.model.eval()
        ac = nn.MSELoss(reduction='none')

        # Train energy
        train_energy = []
        for seq_x, _, _ in train_loader:
            seq_x = seq_x.float().to(self.device)
            outputs = self.model(seq_x, None, None, None)
            train_energy.append(torch.mean(ac(seq_x, outputs), dim=-1).cpu().numpy())
        train_energy = np.concatenate(train_energy).reshape(-1)

        # Test energy + labels
        test_energy, test_labels = [], []
        for seq_x, label, _ in test_loader:
            seq_x = seq_x.float().to(self.device)
            outputs = self.model(seq_x, None, None, None)
            test_energy.append(torch.mean(ac(seq_x, outputs), dim=-1).cpu().numpy())
            test_labels.append(label.numpy())
        test_energy = np.concatenate(test_energy).reshape(-1)
        test_labels = np.concatenate(test_labels).reshape(-1).astype(int)

        threshold = np.percentile(np.concatenate([train_energy, test_energy]),
                                   100 - self.args.anomaly_ratio)
        preds = (test_energy > threshold).astype(int)
        test_labels, preds = adjustment(test_labels, preds)

        acc = accuracy_score(test_labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(test_labels, preds, average='binary', zero_division=0)
        print(f"\nReconstruct | Acc {acc:.4f} F1 {f1:.4f} | threshold {threshold:.6f}")

        with open('result_anomaly_detection_MoE.txt', 'a') as f:
            f.write(f"{setting} | mode=reconstruct | model={self.args.model}\n")
            f.write(f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}\n\n")
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
