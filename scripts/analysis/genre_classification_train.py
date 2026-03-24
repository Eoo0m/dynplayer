"""
Train genre classification model using HTS-AT audio embeddings

This script:
1. Loads the HTS-AT model (pretrained on AudioSet)
2. Uses it to extract audio embeddings
3. Trains a linear classifier for genre classification
"""

import sys
sys.path.append('HTS-Audio-Transformer')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import HTS-AT model
from models import HTSAT_Swin_Transformer
from config import get_config


class GenreDataset(Dataset):
    """Dataset for genre classification"""

    def __init__(self, track_ids, genres, genre_to_idx, audio_dir="data/audio"):
        self.track_ids = track_ids
        self.genres = genres
        self.genre_to_idx = genre_to_idx
        self.audio_dir = Path(audio_dir)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        genre = self.genres[idx]
        genre_idx = self.genre_to_idx[genre]

        # For now, return dummy audio features
        # In practice, you would load actual audio files here
        audio_path = self.audio_dir / f"{track_id}.wav"

        # Placeholder: random audio features
        # TODO: Replace with actual audio loading
        audio_features = torch.randn(1, 32000 * 10)  # 10 seconds at 32kHz

        return {
            'track_id': track_id,
            'audio': audio_features,
            'genre_idx': genre_idx,
            'genre': genre
        }


class GenreClassifier(nn.Module):
    """Linear classifier on top of HTS-AT embeddings"""

    def __init__(self, embedding_dim=768, num_genres=192):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_genres)
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)


def load_genre_data(csv_path="data/spotify_genre_info_frequent.csv"):
    """Load preprocessed genre data"""

    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['genres']:  # Only keep tracks with genres
                data.append({
                    'track_id': row['track_id'],
                    'genre': row['genres']
                })

    df = pd.DataFrame(data)
    print(f"Loaded {len(df):,} tracks with genres")

    # Create genre mapping
    genres = sorted(df['genre'].unique())
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    idx_to_genre = {idx: genre for genre, idx in genre_to_idx.items()}

    print(f"Number of genres: {len(genres)}")

    return df, genre_to_idx, idx_to_genre


def train_epoch(model, audio_encoder, dataloader, criterion, optimizer, device):
    """Train for one epoch"""

    model.train()
    audio_encoder.eval()  # Keep audio encoder frozen

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        audio = batch['audio'].to(device)
        labels = batch['genre_idx'].to(device)

        # Extract audio embeddings (frozen)
        with torch.no_grad():
            # TODO: Replace with actual HTS-AT forward pass
            # embeddings = audio_encoder(audio)
            # For now, use dummy embeddings
            embeddings = torch.randn(audio.size(0), 768).to(device)

        # Classify
        logits = model(embeddings)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, audio_encoder, dataloader, criterion, device, idx_to_genre):
    """Evaluate model"""

    model.eval()
    audio_encoder.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].to(device)
            labels = batch['genre_idx'].to(device)

            # Extract audio embeddings
            # TODO: Replace with actual HTS-AT forward pass
            # embeddings = audio_encoder(audio)
            embeddings = torch.randn(audio.size(0), 768).to(device)

            # Classify
            logits = model(embeddings)
            loss = criterion(logits, labels)

            # Track metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1, all_preds, all_labels


def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else
                         ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Device: {device}")

    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3

    # Load data
    print("\n=== Loading Genre Data ===")
    df, genre_to_idx, idx_to_genre = load_genre_data()

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['genre'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['genre'])

    print(f"\nTrain: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Create datasets
    train_dataset = GenreDataset(
        train_df['track_id'].values,
        train_df['genre'].values,
        genre_to_idx
    )
    val_dataset = GenreDataset(
        val_df['track_id'].values,
        val_df['genre'].values,
        genre_to_idx
    )
    test_dataset = GenreDataset(
        test_df['track_id'].values,
        test_df['genre'].values,
        genre_to_idx
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize models
    print("\n=== Initializing Models ===")

    # TODO: Load HTS-AT model with pretrained weights
    # For now, use placeholder
    # config = get_config()
    # audio_encoder = HTSAT_Swin_Transformer(config)
    # checkpoint = torch.load('HTS-Audio-Transformer/HTSAT_AudioSet_Saved_1.ckpt')
    # audio_encoder.load_state_dict(checkpoint)
    audio_encoder = None  # Placeholder

    # Genre classifier
    num_genres = len(genre_to_idx)
    classifier = GenreClassifier(embedding_dim=768, num_genres=num_genres).to(device)

    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training loop
    print("\n=== Training ===")
    best_val_acc = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            classifier, audio_encoder, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(
            classifier, audio_encoder, val_loader, criterion, device, idx_to_genre
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'genre_to_idx': genre_to_idx,
                'idx_to_genre': idx_to_genre,
            }, 'genre_classifier_best.pt')
            print(f"✅ Saved best model (val_acc={val_acc:.4f})")

    # Final evaluation on test set
    print("\n=== Testing ===")
    checkpoint = torch.load('genre_classifier_best.pt')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        classifier, audio_encoder, test_loader, criterion, device, idx_to_genre
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")

    # Detailed classification report
    print("\n=== Classification Report ===")
    genre_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]
    print(classification_report(test_labels, test_preds, target_names=genre_names, zero_division=0))

    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
