"""
Prepare UNSW-NB15 dataset for multimodal anomaly detection.

Input:  data/UNSW-NB15/raw/*.csv  (manually downloaded)
Output: data/UNSW-NB15/{train,val,test}.csv + text_emb.pt

Usage:  python scripts/prepare_unsw_nb15.py
"""
import os, sys, numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "UNSW-NB15")

ATTACK_DESCRIPTIONS = {
    "Normal": "Normal benign network traffic flow with no malicious activity detected.",
    "Fuzzers": "Fuzzing attack: sending malformed or random data to network services to discover vulnerabilities and cause crashes.",
    "Analysis": "Analysis attack: penetrating web applications through ports, emails, and web scripts to gather information.",
    "Backdoor": "Backdoor attack: bypassing normal authentication to gain remote access to a system while remaining undetected.",
    "Backdoors": "Backdoor attack: bypassing normal authentication to gain remote access to a system while remaining undetected.",
    "DoS": "Denial of Service attack: flooding the target with traffic to exhaust resources and disrupt legitimate services.",
    "Exploits": "Exploit attack: leveraging known software vulnerabilities to gain unauthorized access or control over a system.",
    "Generic": "Generic attack: using cryptographic hash collisions to break block cipher security without specific configuration knowledge.",
    "Reconnaissance": "Reconnaissance attack: gathering information about target networks through probing and scanning to identify vulnerabilities.",
    "Shellcode": "Shellcode attack: injecting and executing malicious code in shell environments to compromise target systems.",
    "Worms": "Worm attack: self-replicating malware that spreads across networks by exploiting vulnerabilities in target computers.",
}


def load_data():
    """Load training and testing CSV files."""
    raw = os.path.join(DATA_DIR, "raw")
    train_path = os.path.join(raw, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(raw, "UNSW_NB15_testing-set.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(f"Train: {df_train.shape}, Test: {df_test.shape}")
    return df_train, df_test


def preprocess(df_train, df_test):
    """Extract features, scale, generate text descriptions, split."""
    # --- 1. Identify columns ---
    # Numeric columns to use as features (exclude id, attack_cat, label, and categoricals)
    categorical_cols = ["proto", "service", "state"]
    exclude = ["id", "attack_cat", "label"] + categorical_cols

    feature_cols = []
    for c in df_train.columns:
        if c in exclude:
            continue
        try:
            pd.to_numeric(df_train[c].iloc[0])
            feature_cols.append(c)
        except (ValueError, TypeError):
            pass

    print(f"Selected {len(feature_cols)} numerical features")

    # --- 2. Scale numerical features ---
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- 3. Labels ---
    y_train = df_train["label"].values.astype(int)
    y_test = df_test["label"].values.astype(int)

    # Normalize attack category names
    attack_train = df_train["attack_cat"].fillna("Normal").str.strip()
    attack_test = df_test["attack_cat"].fillna("Normal").str.strip()

    # --- 4. Merge and split ---
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    attacks = np.concatenate([attack_train.values, attack_test.values])

    # Split: 70% train, 10% val, 20% test (preserving class distribution)
    indices = np.arange(len(X))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.125, random_state=42, stratify=y[tr_idx])

    # --- 5. Generate BERT text embeddings ---
    texts = [ATTACK_DESCRIPTIONS.get(a, f"Network traffic: {a}") for a in attacks]
    text_embeddings = encode_texts(texts)

    # --- 6. Save ---
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    for name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        df = pd.DataFrame(X[idx], columns=feature_names)
        df["label"] = y[idx]
        df["attack_type"] = attacks[idx]
        df["text_idx"] = idx  # maps back to text_emb.pt row
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  {name}: {len(df)} samples, attack ratio: {df['label'].mean():.3f}")

    print(f"\nFeatures: {X.shape[1]}, Total: {len(X)} samples")
    print(f"Data saved to {DATA_DIR}/")
    return X.shape[1]


def encode_texts(texts, model_name="bert-base-uncased", batch_size=64):
    """BERT-encode unique texts; cache to text_emb.pt."""
    cache_path = os.path.join(DATA_DIR, "text_emb.pt")
    if os.path.exists(cache_path):
        print(f"Using cached: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    # Encode each unique text once
    unique = list(set(texts))
    print(f"Encoding {len(unique)} unique texts with {model_name} ...")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    text2emb = {}
    for text in unique:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state[:, 0, :].cpu()  # [CLS]
        text2emb[text] = emb

    # Map back to original order
    embeddings = torch.cat([text2emb[t] for t in texts], dim=0)
    torch.save(embeddings, cache_path)
    print(f"Saved {embeddings.shape} to {cache_path}")
    return embeddings


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df_train, df_test = load_data()
    preprocess(df_train, df_test)
    print("\nDone. Ready for: python run.py --data UNSWNB15 ...")
