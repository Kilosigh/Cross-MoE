"""
Multimodal anomaly detection with MoE-Attn (mix_type=2) for standard benchmarks.

Each dataset gets a static text description (dataset background), encoded
once by BERT, and fused via Cross_MoE's MoE-Attn mixer during reconstruction.
"""
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os, warnings
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from exp.exp_model_dict import model_dict

warnings.filterwarnings('ignore')


# Static dataset descriptions (one sentence per dataset for MoE-Attn fusion)
DATASET_TEXTS = {
    'PSM':  "Pooled Server Metrics dataset for IT operations anomaly detection in server resource monitoring.",
    'MSL':  "Mars Science Laboratory rover sensor dataset for spacecraft telemetry anomaly detection.",
    'SMAP': "Soil Moisture Active Passive satellite dataset for environmental time-series anomaly detection.",
    'SMD':  "Server Machine Dataset for IT infrastructure and system metrics anomaly detection.",
    'SWaT': "Secure Water Treatment testbed dataset for industrial control system anomaly detection.",
    'UNSWNB15': "UNSW-NB15 network intrusion detection dataset with modern attack types and flow-based features.",
}


def get_static_text_embedding(dataset_name, bert_dim=768):
    """Get or compute BERT [CLS] embedding for a dataset description (cached)."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", ".text_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_text_emb.pt")

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    text = DATASET_TEXTS.get(dataset_name, f"Multivariate time series dataset for {dataset_name}.")
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[:, 0, :].cpu()  # [1, 768]
    torch.save(emb, cache_path)
    return emb


class Exp_Anomaly_Detection_MoE(Exp_Basic):
    """Reconstruction-based anomaly detection with optional static text via Cross_MoE."""

    def __init__(self, args):
        super().__init__(args)
        self.text_emb = None
        if args.use_text:
            self.text_emb = get_static_text_embedding(args.data, args.llm_dim)

    def _build_model(self):
        base_model = model_dict[self.args.model].Model
        self.model_dict = {**model_dict, 'Cross_MoE': __import__('models.Cross_MoE', fromlist=['Model']).Model}
        if self.args.use_text:
            model = self.model_dict['Cross_MoE'](self.args).float()
        else:
            model = base_model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    # ---- training --------------------------------------------------------
    def _step(self, batch_x, batch_text):
        if self.args.use_text:
            ret = self.model((batch_x, None, None, None, batch_text))
            return ret["outputs"], ret.get("aux_loss", 0.0)
        return self.model(batch_x, None, None, None), 0.0

    def train(self, setting):
        train_loader = self._get_data('train')[1]
        vali_loader = self._get_data('val')[1]
        test_loader = self._get_data('test')[1]

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer = self._select_optimizer()
        criterion = nn.MSELoss()
        stopper = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            for batch_x, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                if self.text_emb is not None:
                    batch_text = self.text_emb.expand(batch_x.shape[0], -1).to(self.device)
                else:
                    batch_text = None
                optimizer.zero_grad()
                outputs, aux_loss = self._step(batch_x, batch_text)
                f_dim = -1 if self.args.features == 'MS' else 0
                loss = criterion(outputs[:, :, f_dim:], batch_x)  # aux_loss excluded
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self._eval_loss(vali_loader)
            test_loss = self._eval_loss(test_loader)
            print(f"Epoch {epoch+1:3d} | train {train_loss:.4f} val {vali_loss:.4f} test {test_loss:.4f}")

            stopper(vali_loss, self.model, path)
            if stopper.early_stop:
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        ckpt = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(ckpt, weights_only=False))
        return self.model

    def _eval_loss(self, loader):
        losses = []
        self.model.eval()
        crit = nn.MSELoss()
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.float().to(self.device)
                if self.text_emb is not None:
                    batch_text = self.text_emb.expand(batch_x.shape[0], -1).to(self.device)
                else:
                    batch_text = None
                outputs, _ = self._step(batch_x, batch_text)
                f_dim = -1 if self.args.features == 'MS' else 0
                losses.append(crit(outputs[:, :, f_dim:], batch_x).item())
        self.model.train()
        return np.average(losses)

    # ---- testing ---------------------------------------------------------
    def test(self, setting, test=0):
        test_loader = self._get_data('test')[1]
        train_loader = self._get_data('train')[1]
        ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, weights_only=False))

        self.model.eval()
        ac = nn.MSELoss(reduction='none')

        def _compute_energy(loader):
            energy = []
            for batch_x, _ in loader:
                batch_x = batch_x.float().to(self.device)
                if self.text_emb is not None:
                    batch_text = self.text_emb.expand(batch_x.shape[0], -1).to(self.device)
                else:
                    batch_text = None
                outputs, _ = self._step(batch_x, batch_text)
                energy.append(torch.mean(ac(batch_x, outputs), dim=-1).detach().cpu().numpy())
            return np.concatenate(energy).reshape(-1)

        train_energy = _compute_energy(train_loader)
        test_energy = _compute_energy(test_loader)

        test_labels = []
        for _, batch_y in test_loader:
            test_labels.append(batch_y.numpy())
        test_labels = np.concatenate(test_labels).reshape(-1).astype(int)

        thr = np.percentile(np.concatenate([train_energy, test_energy]),
                            100 - self.args.anomaly_ratio)
        preds = (test_energy > thr).astype(int)
        test_labels, preds = adjustment(test_labels, preds)

        acc = accuracy_score(test_labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(test_labels, preds, average='binary', zero_division=0)

        tag = "multi" if self.args.use_text else "uni"
        print(f"\n  {tag} | {self.args.model} | data={self.args.data} | thr={thr:.6f}")
        print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

        with open('result_anomaly_detection_MoE.txt', 'a') as f:
            f.write(f"{setting} | data={self.args.data} | {tag} | model={self.args.model}\n")
            f.write(f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}\n\n")
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
