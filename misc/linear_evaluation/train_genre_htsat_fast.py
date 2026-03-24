"""
Phase 2: Fast genre classifier training using pre-extracted HTS-AT embeddings.

Loads embeddings from disk (no audio loading, no encoder forward pass).
Only trains the linear classifier - very fast!

Prerequisites:
    python linear_evaluation/extract_htsat_embeddings.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class EmbeddingDataset(Dataset):
    """Dataset for pre-extracted embeddings."""

    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class GenreClassifier(nn.Module):
    """Simple linear classifier for genre classification."""

    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def load_embeddings(embedding_dir):
    """Load pre-extracted embeddings and metadata."""
    embedding_dir = Path(embedding_dir)

    embeddings = np.load(embedding_dir / "htsat_embeddings.npy")
    track_ids = np.load(embedding_dir / "track_ids.npy", allow_pickle=True)
    genre_idxs = np.load(embedding_dir / "genre_idxs.npy")

    with open(embedding_dir / "genre_to_idx.pkl", 'rb') as f:
        genre_to_idx = pickle.load(f)
    with open(embedding_dir / "idx_to_genre.pkl", 'rb') as f:
        idx_to_genre = pickle.load(f)

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Genres: {len(genre_to_idx)}")

    return embeddings, track_ids, genre_idxs, genre_to_idx, idx_to_genre


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, return_top5=False):
    model.eval()
    total_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0

    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        logits = model(embeddings)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

        # Top-5 accuracy
        _, top5_preds = logits.topk(5, dim=1)
        correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += len(labels)

    if return_top5:
        return total_loss / total, correct / total, correct_top5 / total
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    batch_size = 256
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-4
    hidden_dim = 512
    dropout = 0.3
    patience = 10

    # Load embeddings
    print("\n=== Loading Pre-extracted Embeddings ===")
    embedding_dir = Path(__file__).parent.parent / "data" / "audio_embeddings"
    embeddings, track_ids, genre_idxs, genre_to_idx, idx_to_genre = load_embeddings(embedding_dir)

    input_dim = embeddings.shape[1]
    num_classes = len(genre_to_idx)

    print(f"\nInput dim: {input_dim}")
    print(f"Num classes: {num_classes}")

    # Filter out classes with too few samples for stratified split
    print("\n=== Filtering Classes ===")
    from collections import Counter
    class_counts = Counter(genre_idxs)
    min_samples = 3  # Need at least 3 for train/val/test split
    valid_classes = {cls for cls, count in class_counts.items() if count >= min_samples}

    # Filter data
    mask = np.array([g in valid_classes for g in genre_idxs])
    embeddings_filtered = embeddings[mask]
    genre_idxs_filtered = genre_idxs[mask]

    print(f"Original: {len(genre_idxs)} samples, {len(class_counts)} classes")
    print(f"Filtered: {len(genre_idxs_filtered)} samples, {len(valid_classes)} classes")
    excluded = len(class_counts) - len(valid_classes)
    if excluded > 0:
        print(f"Excluded {excluded} classes with < {min_samples} samples")

    # Update num_classes based on actual classes present
    num_classes = len(genre_to_idx)  # Keep original for model, use full label space

    # Split data
    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_filtered, genre_idxs_filtered, test_size=0.2, random_state=42, stratify=genre_idxs_filtered
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"Train: {len(X_train):,}")
    print(f"Val: {len(X_val):,}")
    print(f"Test: {len(X_test):,}")

    # Create datasets
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)
    test_dataset = EmbeddingDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print("\n=== Creating Classifier ===")
    model = GenreClassifier(input_dim, num_classes, hidden_dim, dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Classifier parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Training
    print("\n=== Training ===")
    best_val_acc = 0
    best_model_state = None
    no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test set
    print("\n=== Test Evaluation ===")
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_acc_top5 = evaluate(model, test_loader, criterion, device, return_top5=True)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy (Top-1): {test_acc:.4f}")
    print(f"Test Accuracy (Top-5): {test_acc_top5:.4f}")

    # Per-class accuracy
    print("\n=== Per-class Accuracy ===")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings = embeddings.to(device)
            logits = model(embeddings)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    for genre_idx in range(num_classes):
        mask = all_labels == genre_idx
        if mask.sum() > 0:
            acc = (all_preds[mask] == genre_idx).mean()
            genre_name = idx_to_genre[genre_idx]
            print(f"  {genre_name}: {acc:.4f} ({mask.sum()} samples)")

    # Save model
    output_dir = Path(__file__).parent.parent / "data" / "audio_embeddings"
    torch.save(best_model_state, output_dir / "genre_classifier.pt")
    print(f"\nSaved classifier to {output_dir / 'genre_classifier.pt'}")


if __name__ == "__main__":
    main()
