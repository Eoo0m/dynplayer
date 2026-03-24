"""
Train genre classification model using HTS-AT audio embeddings

This script:
1. Loads the HTS-AT model (pretrained on AudioSet)
2. Extracts audio embeddings from the model
3. Trains a linear classifier for genre classification on Spotify tracks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'HTS-Audio-Transformer'))

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
import torchaudio
import librosa

# Import HTS-AT model
try:
    from model.htsat import HTSAT_Swin_Transformer
    import config as htsat_config
except ImportError as e:
    print(f"Error importing HTS-AT modules: {e}")
    print("Make sure HTS-Audio-Transformer is in the parent directory")
    sys.exit(1)


class AudioGenreDataset(Dataset):
    """Dataset for genre classification with audio loading"""

    def __init__(self, track_ids, genres, genre_to_idx, audio_dir="data/audio",
                 sample_rate=32000, duration=10):
        self.track_ids = track_ids
        self.genres = genres
        self.genre_to_idx = genre_to_idx
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration

    def __len__(self):
        return len(self.track_ids)

    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            # Load audio with librosa
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True, duration=self.duration)

            # Pad or trim to target length
            if len(waveform) < self.target_length:
                waveform = np.pad(waveform, (0, self.target_length - len(waveform)), mode='constant')
            else:
                waveform = waveform[:self.target_length]

            return torch.FloatTensor(waveform).unsqueeze(0)  # [1, samples]

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence if loading fails
            return torch.zeros(1, self.target_length)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        genre = self.genres[idx]
        genre_idx = self.genre_to_idx[genre]

        # Try multiple audio file extensions
        audio_path = None
        for ext in ['.mp3', '.wav', '.m4a', '.flac']:
            candidate = self.audio_dir / f"{track_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        # Load audio (should always exist due to pre-filtering)
        if audio_path is not None and audio_path.exists():
            audio = self.load_audio(audio_path)
        else:
            # This shouldn't happen if data is pre-filtered
            raise FileNotFoundError(f"Audio file not found for track {track_id}")

        return {
            'track_id': track_id,
            'audio': audio,
            'genre_idx': genre_idx,
            'genre': genre
        }


class LinearGenreClassifier(nn.Module):
    """Linear classifier on top of frozen HTS-AT embeddings"""

    def __init__(self, embedding_dim=768, num_genres=192, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_genres)
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)


def load_htsat_model(checkpoint_path, device):
    """Load HTS-AT model with pretrained weights"""

    print(f"Loading HTS-AT model from {checkpoint_path}...")

    # Create model
    model = HTSAT_Swin_Transformer(
        spec_size=htsat_config.htsat_spec_size,
        patch_size=htsat_config.htsat_patch_size,
        patch_stride=htsat_config.htsat_stride,
        num_classes=htsat_config.classes_num,
        embed_dim=htsat_config.htsat_dim,
        depths=htsat_config.htsat_depth,
        num_heads=htsat_config.htsat_num_head,
        window_size=htsat_config.htsat_window_size,
        config=htsat_config,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict (handle different checkpoint formats)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("✅ HTS-AT model loaded successfully")

    return model


def extract_embeddings(model, audio_batch, device):
    """Extract embeddings from HTS-AT model"""

    with torch.no_grad():
        # Squeeze channel dimension: (batch, 1, samples) -> (batch, samples)
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(1)

        audio_batch = audio_batch.to(device)

        # Forward pass through HTS-AT
        # The model returns a dict with 'clipwise_output' as the embeddings
        output_dict = model(audio_batch, None, False)

        # Get the clipwise output (527-dim)
        embeddings = output_dict['clipwise_output']

        return embeddings


def load_genre_data(csv_path="../data/spotify_genre_info_frequent.csv", audio_dir="/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node"):
    """Load preprocessed genre data and filter tracks with available audio files"""

    csv_path = Path(__file__).parent / csv_path
    audio_dir = Path(__file__).parent / audio_dir

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

    # Filter tracks with available audio files
    def has_audio(track_id):
        for ext in ['.mp3', '.wav', '.m4a', '.flac']:
            if (audio_dir / f"{track_id}{ext}").exists():
                return True
        return False

    original_count = len(df)
    df['has_audio'] = df['track_id'].apply(has_audio)
    df = df[df['has_audio']].drop(columns=['has_audio']).reset_index(drop=True)

    print(f"Tracks with audio files: {len(df):,} / {original_count:,} ({len(df)/original_count*100:.1f}%)")

    if len(df) == 0:
        print("\n⚠️  WARNING: No audio files found!")
        print(f"Please check that audio files exist in: {audio_dir}")
        print("Expected extensions: .mp3, .wav, .m4a, .flac")

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
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        audio = batch['audio'].to(device)
        labels = batch['genre_idx'].to(device)

        # Extract audio embeddings (frozen)
        embeddings = extract_embeddings(audio_encoder, audio, device)

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

        # Print batch progress
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            running_acc = accuracy_score(all_labels, all_preds)
            print(f"  Batch [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f} Acc: {running_acc:.4f}")

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
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            audio = batch['audio'].to(device)
            labels = batch['genre_idx'].to(device)

            # Extract audio embeddings
            embeddings = extract_embeddings(audio_encoder, audio, device)

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
    # Note: MPS has compatibility issues with torchlibrosa, so we use CPU for now
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    batch_size = 16  # Smaller batch size for audio
    num_epochs = 20
    learning_rate = 1e-3

    checkpoint_path = Path(__file__).parent.parent / "HTS-Audio-Transformer" / "HTSAT_AudioSet_Saved_6.ckpt"

    # Load HTS-AT model
    print("\n=== Loading HTS-AT Model ===")
    audio_encoder = load_htsat_model(checkpoint_path, device)

    # Freeze audio encoder
    for param in audio_encoder.parameters():
        param.requires_grad = False

    # Load data
    print("\n=== Loading Genre Data ===")
    df, genre_to_idx, idx_to_genre = load_genre_data()

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['genre'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['genre'])

    print(f"\nTrain: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Create datasets
    audio_dir = Path("/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node")

    train_dataset = AudioGenreDataset(
        train_df['track_id'].values,
        train_df['genre'].values,
        genre_to_idx,
        audio_dir=audio_dir
    )
    val_dataset = AudioGenreDataset(
        val_df['track_id'].values,
        val_df['genre'].values,
        genre_to_idx,
        audio_dir=audio_dir
    )
    test_dataset = AudioGenreDataset(
        test_df['track_id'].values,
        test_df['genre'].values,
        genre_to_idx,
        audio_dir=audio_dir
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Genre classifier
    print("\n=== Initializing Classifier ===")
    num_genres = len(genre_to_idx)
    # HTS-AT outputs 527-dim features (AudioSet classes)
    classifier = LinearGenreClassifier(embedding_dim=527, num_genres=num_genres).to(device)

    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training loop
    print("\n=== Training ===")
    best_val_acc = 0

    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

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
            save_path = output_dir / 'genre_classifier_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'genre_to_idx': genre_to_idx,
                'idx_to_genre': idx_to_genre,
            }, save_path)
            print(f"✅ Saved best model to {save_path} (val_acc={val_acc:.4f})")

    # Final evaluation on test set
    print("\n=== Testing ===")
    checkpoint = torch.load(output_dir / 'genre_classifier_best.pt')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        classifier, audio_encoder, test_loader, criterion, device, idx_to_genre
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")

    # Save test results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'predictions': test_preds,
        'labels': test_labels,
        'idx_to_genre': idx_to_genre
    }
    np.save(output_dir / 'test_results.npy', results)

    print("\n✅ Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
