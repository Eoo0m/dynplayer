"""
Genre classification using PANNs MobileNetV2
Pretrained on AudioSet, only 2.4M parameters (very lightweight)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "panns", "pytorch"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import librosa

from models import MobileNetV2


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AudioGenreDataset(Dataset):
    def __init__(
        self, track_ids, genres, genre_to_idx, audio_dir, sample_rate=32000, duration=10
    ):
        self.track_ids = track_ids
        self.genres = genres
        self.genre_to_idx = genre_to_idx
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        genre = self.genres[idx]
        genre_idx = self.genre_to_idx[genre]

        audio_path = self.audio_dir / f"{track_id}.mp3"

        try:
            waveform, _ = librosa.load(
                audio_path, sr=self.sample_rate, mono=True, duration=self.duration
            )
            if len(waveform) < self.target_length:
                waveform = np.pad(waveform, (0, self.target_length - len(waveform)))
            else:
                waveform = waveform[: self.target_length]
        except:
            waveform = np.zeros(self.target_length)

        return {
            "audio": torch.FloatTensor(waveform),
            "genre_idx": genre_idx,
        }


class GenreClassifier(nn.Module):
    """Linear classifier on MobileNetV2 embeddings"""

    def __init__(self, embedding_dim=1024, num_genres=192):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_genres)

    def forward(self, x):
        return self.classifier(x)


def load_mobilenet(checkpoint_path, device):
    """Load pretrained MobileNetV2"""
    model = MobileNetV2(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    return model


def extract_embeddings(model, audio, device, training=False):
    """Extract embeddings from MobileNetV2"""
    audio = audio.to(device)
    output = model(audio)
    # embedding is before final classifier
    embeddings = output["embedding"]
    return embeddings


def load_genre_data():
    csv_path = Path(__file__).parent.parent / "data" / "spotify_genre_info_frequent.csv"
    audio_dir = Path(
        "/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node"
    )

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["genres"]:
                track_id = row["track_id"]
                if (audio_dir / f"{track_id}.mp3").exists():
                    data.append({"track_id": track_id, "genre": row["genres"]})

    df = pd.DataFrame(data)
    print(f"Tracks with audio: {len(df):,}")

    genres = sorted(df["genre"].unique())
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}

    print(f"Genres: {len(genres)}")
    return df, genre_to_idx, idx_to_genre


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = 64
    num_epochs = 10
    lr = 1e-3

    # Load MobileNetV2
    print("\n=== Loading MobileNetV2 ===")
    checkpoint_path = (
        Path(__file__).parent.parent / "panns" / "checkpoints" / "MobileNetV2.pth"
    )
    encoder = load_mobilenet(checkpoint_path, device)

    # Freeze encoder - only train classifier
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"MobileNetV2 parameters: {num_params:,} (frozen)")

    # Load data
    print("\n=== Loading Data ===")
    df, genre_to_idx, idx_to_genre = load_genre_data()

    # Split 80:10:10
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["genre"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["genre"]
    )
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    audio_dir = Path(
        "/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node"
    )

    train_dataset = AudioGenreDataset(
        train_df["track_id"].values, train_df["genre"].values, genre_to_idx, audio_dir
    )
    val_dataset = AudioGenreDataset(
        val_df["track_id"].values, val_df["genre"].values, genre_to_idx, audio_dir
    )
    test_dataset = AudioGenreDataset(
        test_df["track_id"].values, test_df["genre"].values, genre_to_idx, audio_dir
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Classifier
    print("\n=== Classifier ===")
    classifier = GenreClassifier(embedding_dim=1024, num_genres=len(genre_to_idx)).to(
        device
    )
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # Training
    print("\n=== Training ===")
    best_val_acc = 0

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        total_loss = 0
        all_preds, all_labels = [], []
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            audio = batch["audio"].to(device)
            labels = batch["genre_idx"].to(device)

            with torch.no_grad():
                embeddings = extract_embeddings(encoder, audio, device)

            optimizer.zero_grad()
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                acc = accuracy_score(all_labels, all_preds)
                print(
                    f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f} Acc: {acc:.4f}"
                )

        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = total_loss / num_batches

        # Validation
        classifier.eval()
        val_preds, val_labels, val_logits_all = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                labels = batch["genre_idx"].to(device)
                embeddings = extract_embeddings(encoder, audio, device)
                logits = classifier(embeddings)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_logits_all.append(logits.cpu())

        val_logits_all = torch.cat(val_logits_all, dim=0)
        val_labels_tensor = torch.tensor(val_labels)
        top5_preds = val_logits_all.topk(5, dim=1).indices
        val_top5_acc = (top5_preds == val_labels_tensor.unsqueeze(1)).any(dim=1).float().mean().item()

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}, Val Top5: {val_top5_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'classifier': classifier.state_dict(),
                'genre_to_idx': genre_to_idx,
                'idx_to_genre': idx_to_genre,
            }, Path(__file__).parent / "genre_mobilenet_best.pt")
            print(f"  ✅ Saved best model")

    # Test
    print("\n=== Test ===")
    checkpoint = torch.load(Path(__file__).parent / "genre_mobilenet_best.pt")
    classifier.load_state_dict(checkpoint['classifier'])
    encoder.eval()
    classifier.eval()
    test_preds, test_labels, test_logits_all = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio"].to(device)
            labels = batch["genre_idx"].to(device)
            embeddings = extract_embeddings(encoder, audio, device)
            logits = classifier(embeddings)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_logits_all.append(logits.cpu())

    test_logits_all = torch.cat(test_logits_all, dim=0)
    test_labels_tensor = torch.tensor(test_labels)
    top5_preds = test_logits_all.topk(5, dim=1).indices
    test_top5_acc = (top5_preds == test_labels_tensor.unsqueeze(1)).any(dim=1).float().mean().item()

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="weighted")
    print(f"Test Acc: {test_acc:.4f}, Test Top5: {test_top5_acc:.4f}, Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
