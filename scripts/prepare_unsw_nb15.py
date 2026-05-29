"""
Download and preprocess UNSW-NB15 dataset for multimodal anomaly detection.

UNSW-NB15: 49 features, 9 attack types, ~2.5M records
Download: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Preprocessed output:
  data/UNSW-NB15/train.csv    - training set (time-series windows + text)
  data/UNSW-NB15/test.csv     - test set
  data/UNSW-NB15/text_emb.pt  - pre-computed BERT embeddings for text descriptions
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import requests
import gzip
import shutil
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "UNSW-NB15")
UNSW_URL = "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download"

# Attack type → text description mapping
ATTACK_DESCRIPTIONS = {
    "Normal": "Normal benign network traffic flow with no malicious activity detected.",
    "Fuzzers": "Fuzzing attack: sending malformed or random data to network services to discover vulnerabilities and cause crashes.",
    "Analysis": "Analysis attack: penetrating web applications through ports (port scan), emails (spam), and web scripts.",
    "Backdoors": "Backdoor attack: bypassing normal authentication to gain remote access to a system while remaining undetected.",
    "DoS": "Denial of Service attack: flooding the target with traffic to exhaust resources and disrupt legitimate services.",
    "Exploits": "Exploit attack: leveraging known software vulnerabilities to gain unauthorized access or control over a system.",
    "Generic": "Generic attack: using cryptographic hash collisions to break block cipher security without requiring specific configuration knowledge.",
    "Reconnaissance": "Reconnaissance attack: gathering information about target networks through probing and scanning to identify vulnerabilities.",
    "Shellcode": "Shellcode attack: injecting and executing malicious code in shell environments to compromise target systems.",
    "Worms": "Worm attack: self-replicating malware that spreads across networks by exploiting vulnerabilities in target computers.",
}

# Feature names for UNSW-NB15
FEATURE_NAMES = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
    'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
    'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label'
]


def download_unsw():
    """Download UNSW-NB15 dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "UNSW-NB15.zip")
    extract_dir = os.path.join(DATA_DIR, "raw")

    if os.path.exists(extract_dir):
        print(f"Raw data already exists at {extract_dir}")
        return extract_dir

    print(f"Downloading UNSW-NB15 from {UNSW_URL}...")
    print("If this fails, manually download from:")
    print("  https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print(f"  and extract CSV files to {extract_dir}")

    try:
        # Try direct download
        import urllib.request
        urllib.request.urlretrieve(UNSW_URL, zip_path)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually.")
        sys.exit(1)

    # Extract
    import zipfile
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")
    return extract_dir


def load_raw_data():
    """Load and merge UNSW-NB15 CSV files."""
    raw_dir = os.path.join(DATA_DIR, "raw")
    if not os.path.exists(raw_dir):
        raw_dir = download_unsw()

    # Find all CSV files
    csv_files = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        # Try alternate paths - the dataset might be organized differently
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(root, f))

    if not csv_files:
        print(f"ERROR: No CSV files found in {DATA_DIR}")
        print("Please download UNSW-NB15 manually and place CSV files in:")
        print(f"  {DATA_DIR}/raw/")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files")

    # Load and merge
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
            print(f"  Loaded {f}: {df.shape}")
        except Exception as e:
            print(f"  Skipped {f}: {e}")

    if not dfs:
        print("ERROR: Could not load any CSV files")
        sys.exit(1)

    data = pd.concat(dfs, ignore_index=True)
    print(f"Merged data shape: {data.shape}")
    return data


def preprocess(data):
    """Preprocess UNSW-NB15 data."""
    print("\nPreprocessing...")

    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # Identify key columns
    print(f"Columns: {list(data.columns)}")

    # Standard column names in UNSW-NB15
    # Look for attack category column
    attack_col = None
    for col in ['attack_cat', 'Attack category', 'attack_category']:
        if col in data.columns:
            attack_col = col
            break

    # Look for label column (0=normal, 1=attack)
    label_col = None
    for col in ['label', 'Label']:
        if col in data.columns:
            label_col = col
            break

    if attack_col is None and label_col is None:
        print("WARNING: Could not find attack_cat or label columns.")
        print(f"Available columns: {list(data.columns)}")
        # Try to infer from last columns
        attack_col = data.columns[-2]  # usually second to last
        label_col = data.columns[-1]    # usually last
        print(f"Assuming '{attack_col}' is attack category and '{label_col}' is label")

    # Clean attack category
    if attack_col:
        data[attack_col] = data[attack_col].fillna('Normal').str.strip()
        # Map to standard names
        attack_map = {
            'Normal': 'Normal',
            'Fuzzers': 'Fuzzers',
            'Analysis': 'Analysis',
            'Backdoor': 'Backdoors',
            'Backdoors': 'Backdoors',
            'DoS': 'DoS',
            'Exploits': 'Exploits',
            'Generic': 'Generic',
            'Reconnaissance': 'Reconnaissance',
            'Shellcode': 'Shellcode',
            'Worms': 'Worms',
        }
        data['attack_type'] = data[attack_col].map(attack_map).fillna('Normal')

    # Create binary label
    if label_col:
        data['binary_label'] = (data[label_col].astype(str).str.strip() != '0').astype(int)
    else:
        data['binary_label'] = (data['attack_type'] != 'Normal').astype(int)

    # Select numerical features (exclude non-numerical)
    exclude_cols = [attack_col, label_col, 'attack_type', 'binary_label', 'attack_cat', 'label']
    if attack_col:
        exclude_cols.append(attack_col)
    if label_col:
        exclude_cols.append(label_col)

    numeric_cols = []
    for col in data.columns:
        if col not in exclude_cols:
            try:
                pd.to_numeric(data[col], errors='raise')
                numeric_cols.append(col)
            except:
                pass

    print(f"Selected {len(numeric_cols)} numerical features")

    # Extract numerical features
    X = data[numeric_cols].values.astype(np.float32)

    # Handle NaN and Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Generate text descriptions
    attack_types = data['attack_type'].values
    texts = [ATTACK_DESCRIPTIONS.get(at, f"Network traffic: {at}") for at in attack_types]

    # Create text embeddings using BERT
    print("\nGenerating BERT text embeddings...")
    text_embeddings = encode_texts_bert(texts)

    # Split: 70% train, 10% val, 20% test
    n = len(X)
    indices = np.arange(n)

    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=data['binary_label'])
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42, stratify=data['binary_label'].iloc[train_idx])

    # Create final DataFrames
    feature_cols = [f'f{i}' for i in range(X.shape[1])]

    for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        df = pd.DataFrame(X[idx], columns=feature_cols)
        df['label'] = data['binary_label'].values[idx]
        df['attack_type'] = attack_types[idx]
        df['text_idx'] = idx  # store original index for text lookup

        save_path = os.path.join(DATA_DIR, f'{split_name}.csv')
        df.to_csv(save_path, index=False)
        print(f"Saved {split_name}: {df.shape} to {save_path}")

    return X.shape[1], len(train_idx), len(val_idx), len(test_idx)


def encode_texts_bert(texts, batch_size=64, model_name='bert-base-uncased'):
    """Encode texts using BERT, with caching."""
    cache_path = os.path.join(DATA_DIR, 'text_emb.pt')

    if os.path.exists(cache_path):
        print(f"Loading cached text embeddings from {cache_path}")
        return torch.load(cache_path)

    print(f"Encoding {len(texts)} texts using {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    all_embeddings = []
    unique_texts = list(set(texts))
    text_to_emb = {}

    # Encode each unique text once
    for text in unique_texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            emb = outputs.last_hidden_state[:, 0, :].cpu()

        text_to_emb[text] = emb

    # Map back to original order
    for text in texts:
        all_embeddings.append(text_to_emb[text])

    embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings, cache_path)
    print(f"Saved text embeddings ({embeddings.shape}) to {cache_path}")
    return embeddings


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)

    data = load_raw_data()
    n_features, n_train, n_val, n_test = preprocess(data)

    print(f"\nPreprocessing complete!")
    print(f"  Features: {n_features}")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"  Data saved to: {DATA_DIR}")
    print(f"\nTo use this dataset, run:")
    print(f"  python run.py --task_name anomaly_detection --data UNSWNB15 "
          f"--root_path ./data/UNSW-NB15/ ...")
