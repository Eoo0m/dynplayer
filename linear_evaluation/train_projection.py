import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# Seed Utils
# ---------------------
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(SEED)

# ---------------------
# File paths
# ---------------------
csv_path = "linear_evaluation/spotify_genre_info_sampled.csv"

embedding_sets = {
    "u00": {
        "keys": "contrastive_learning/u00_keys.npy",
        "emb": "contrastive_learning/u00_embeddings.npy",
    },
    "u05": {
        "keys": "contrastive_learning/u05_keys.npy",
        "emb": "contrastive_learning/u05_embeddings.npy",
    },
    "u10": {
        "keys": "contrastive_learning/u10_keys.npy",
        "emb": "contrastive_learning/u10_embeddings.npy",
    },
}

# ---------------------
# Dataset Load (labels are shared)
# ---------------------
df = pd.read_csv(csv_path)
df = df.dropna(subset=["genres"])
df["label"] = df["genres"].apply(lambda x: x.split(",")[0].strip())
df = df[df["label"].notnull() & (df["label"] != "")]

# Filter genres that appear at least 100 times
genre_counts = df["label"].value_counts()
valid_genres = genre_counts[genre_counts >= 100].index.tolist()
print(f"Total genres before filtering: {len(genre_counts)}")
print(f"Genres with >= 100 samples: {len(valid_genres)}")

# Keep only valid genres
df = df[df["label"].isin(valid_genres)].copy()
print(f"Total samples after filtering: {len(df)}")

# LabelEncoder (공통)
labels = df["label"].tolist()
le = LabelEncoder()
le.fit(labels)
classes = le.classes_
num_classes = len(classes)
print(f"총 {num_classes}개 장르 (filtered)")


# ---------------------
# Linear Evaluation Function
# ---------------------
def linear_evaluate(tag, keys_path, emb_path):

    print(f"\n==============================")
    print(f"Running Linear Eval on: {tag}")
    print(f"==============================")

    keys = np.load(keys_path)
    embeddings = np.load(emb_path)

    # track_id → index
    tid2idx = {tid: i for i, tid in enumerate(keys)}
    df2 = df[df["track_id"].isin(tid2idx)].copy()
    df2["emb_idx"] = df2["track_id"].map(tid2idx)

    # X, y 생성
    X = np.stack([embeddings[i] for i in df2["emb_idx"]])
    y = le.transform(df2["label"].tolist())

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Split
    n_total = len(X)
    n_train = int(0.8 * n_total)
    train_ds, test_ds = random_split(
        TensorDataset(X, y),
        [n_train, n_total - n_train],
        generator=torch.Generator().manual_seed(SEED),
    )

    def seed_worker(worker_id):
        np.random.seed(SEED + worker_id)
        random.seed(SEED + worker_id)

    g = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    # Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(X.shape[1], num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(50):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} Loss={total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)

            preds = logits.argmax(dim=1).cpu().numpy()

            y_true.append(yb.numpy())
            y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc1 = accuracy_score(y_true, y_pred)

    print(f"🎯 {tag} Top-1: {acc1:.4f}")

    return acc1


# ---------------------
# Run all three evaluations
# ---------------------
results = {}

for tag, paths in embedding_sets.items():
    acc1 = linear_evaluate(tag, paths["keys"], paths["emb"])
    results[tag] = {"Top1": acc1}

# 결과 요약
print("\n==================== SUMMARY ====================")
df_summary = pd.DataFrame(results).T
print(df_summary)

df_summary.to_csv("linear_eval_summary.csv", index=True)
print("Saved → linear_eval_summary.csv")
