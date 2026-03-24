"""
Lightweight genre classification training
Uses a simple CNN on mel-spectrograms instead of heavy HTS-AT
"""

import sys
import os
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


class AudioGenreDataset(Dataset):
    """Dataset that loads audio and converts to mel-spectrogram"""

    def __init__(self, track_ids, genres, genre_to_idx, audio_dir,
                 sample_rate=22050, duration=5, n_mels=64):
        self.track_ids = track_ids
        self.genres = genres
        self.genre_to_idx = genre_to_idx
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.target_length = sample_rate * duration

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        genre = self.genres[idx]
        genre_idx = self.genre_to_idx[genre]

        audio_path = self.audio_dir / f"{track_id}.mp3"

        try:
            # Load audio (5 seconds, 22kHz - lighter than 10s @ 32kHz)
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate,
                                       mono=True, duration=self.duration)

            # Pad or trim
            if len(waveform) < self.target_length:
                waveform = np.pad(waveform, (0, self.target_length - len(waveform)))
            else:
                waveform = waveform[:self.target_length]

            # Convert to mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=waveform, sr=self.sample_rate, n_mels=self.n_mels,
                hop_length=512, n_fft=1024
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

            mel_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)  # [1, n_mels, time]

        except Exception as e:
            # Return zeros on error
            mel_tensor = torch.zeros(1, self.n_mels, 216)

        return {
            'mel': mel_tensor,
            'genre_idx': genre_idx,
        }


class LightCNN(nn.Module):
    """Simple CNN for mel-spectrogram classification"""

    def __init__(self, n_mels=64, num_genres=192):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_genres),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_genre_data():
    csv_path = Path(__file__).parent.parent / "data" / "spotify_genre_info_frequent.csv"
    audio_dir = Path("/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node")

    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['genres']:
                track_id = row['track_id']
                if (audio_dir / f"{track_id}.mp3").exists():
                    data.append({'track_id': track_id, 'genre': row['genres']})

    df = pd.DataFrame(data)
    print(f"Tracks with audio: {len(df):,}")

    genres = sorted(df['genre'].unique())
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}

    print(f"Genres: {len(genres)}")
    return df, genre_to_idx, idx_to_genre


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    lr = 1e-3

    # Load data
    print("\n=== Loading Data ===")
    df, genre_to_idx, idx_to_genre = load_genre_data()

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['genre'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['genre'])
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    audio_dir = Path("/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node")

    train_dataset = AudioGenreDataset(train_df['track_id'].values, train_df['genre'].values, genre_to_idx, audio_dir)
    val_dataset = AudioGenreDataset(val_df['track_id'].values, val_df['genre'].values, genre_to_idx, audio_dir)
    test_dataset = AudioGenreDataset(test_df['track_id'].values, test_df['genre'].values, genre_to_idx, audio_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    print("\n=== Model ===")
    model = LightCNN(n_mels=64, num_genres=len(genre_to_idx)).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    print("\n=== Training ===")
    best_val_acc = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            mel = batch['mel'].to(device)
            labels = batch['genre_idx'].to(device)

            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                acc = accuracy_score(all_labels, all_preds)
                print(f"  Epoch {epoch} Batch [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f} Acc: {acc:.4f}")

        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                mel = batch['mel'].to(device)
                labels = batch['genre_idx'].to(device)
                logits = model(mel)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(__file__).parent / 'genre_cnn_best.pt')
            print(f"  ✅ Saved best model")

    # Test
    print("\n=== Test ===")
    model.load_state_dict(torch.load(Path(__file__).parent / 'genre_cnn_best.pt'))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            mel = batch['mel'].to(device)
            labels = batch['genre_idx'].to(device)
            logits = model(mel)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
