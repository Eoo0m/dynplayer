"""
Genre classification using MERT-v1-330M pretrained audio model
Linear evaluation: freeze MERT encoder, train only classifier head
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
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T

from transformers import Wav2Vec2FeatureExtractor, AutoModel


class MERTGenreDataset(Dataset):
    """Dataset that loads audio for MERT processing"""

    def __init__(self, track_ids, genres, genre_to_idx, audio_dir,
                 processor, sample_rate=24000, max_duration=30):
        self.track_ids = track_ids
        self.genres = genres
        self.genre_to_idx = genre_to_idx
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_length = sample_rate * max_duration  # 30초 최대

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        genre = self.genres[idx]
        genre_idx = self.genre_to_idx[genre]

        audio_path = self.audio_dir / f"{track_id}.mp3"

        try:
            # Load audio with torchaudio (전체 길이)
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

            # Resample to 24kHz if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Trim to max length if longer (30초 이상이면 자름)
            if waveform.shape[0] > self.max_length:
                waveform = waveform[:self.max_length]

            waveform = waveform.numpy()
            length = len(waveform)

        except Exception as e:
            # Return 1초 zeros on error
            waveform = np.zeros(self.sample_rate, dtype=np.float32)
            length = self.sample_rate

        return {
            'waveform': waveform,
            'length': length,
            'genre_idx': genre_idx,
        }


def collate_fn(batch, processor):
    """Custom collate function to process audio with MERT processor"""
    waveforms = [item['waveform'] for item in batch]
    lengths = [item['length'] for item in batch]
    genre_idxs = [item['genre_idx'] for item in batch]

    # Process with MERT processor (padding=True로 가장 긴 것에 맞춤)
    inputs = processor(
        waveforms,
        sampling_rate=24000,
        return_tensors="pt",
        padding=True
    )

    # 길이 정보로 attention mask 생성 (hidden states 차원에 맞게 변환 필요)
    return {
        'input_values': inputs.input_values,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'genre_idx': torch.tensor(genre_idxs, dtype=torch.long)
    }


class MERTClassifier(nn.Module):
    """MERT encoder with linear classification head"""

    def __init__(self, mert_model, num_genres, freeze_encoder=True, layer_idx=-1):
        super().__init__()
        self.mert = mert_model
        self.freeze_encoder = freeze_encoder
        self.layer_idx = layer_idx  # Which layer to use (-1 = last layer)

        # MERT-v1-330M has 1024-dim hidden states
        self.hidden_size = 1024

        # MERT의 feature rate (75Hz, 즉 1초당 75 프레임)
        self.feature_rate = 75

        # Classification head (mean + max pooling = 2048 dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_genres)
        )

        if freeze_encoder:
            for param in self.mert.parameters():
                param.requires_grad = False

    def forward(self, input_values, lengths=None):
        # Get MERT outputs
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.mert(
                    input_values,
                    output_hidden_states=True
                )
        else:
            outputs = self.mert(
                input_values,
                output_hidden_states=True
            )

        # Get hidden states from specified layer
        # hidden_states: tuple of (num_layers + 1) tensors of shape [batch, time, hidden_size]
        hidden_states = outputs.hidden_states[self.layer_idx]  # [batch, time, 1024]

        batch_size, time_steps, hidden_dim = hidden_states.shape

        # Mean + Max pooling with length masking
        if lengths is not None:
            # 오디오 샘플 길이를 hidden state 프레임 수로 변환 (24kHz / 320 hop = 75Hz)
            frame_lengths = (lengths / 320).long().clamp(min=1, max=time_steps)

            # Create mask for valid frames
            mask = torch.arange(time_steps, device=hidden_states.device).unsqueeze(0) < frame_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(hidden_states).float()

            # Masked mean pooling
            mean_pool = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Masked max pooling
            max_pool = (hidden_states.masked_fill(mask == 0, -1e9)).max(dim=1).values

            # Concatenate mean and max pooling
            pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [batch, 2048]
        else:
            mean_pool = hidden_states.mean(dim=1)
            max_pool = hidden_states.max(dim=1).values
            pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [batch, 2048]

        # Classify
        logits = self.classifier(pooled)
        return logits


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparameters
    batch_size = 8  # Smaller batch size for large model
    num_epochs = 10
    lr = 1e-3  # Higher LR since only training classifier
    freeze_encoder = True  # Linear evaluation
    layer_idx = -1  # Use last layer

    # Load MERT model and processor
    print("\n=== Loading MERT-v1-330M ===")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    print(f"MERT sample rate: {processor.sampling_rate}")

    # Load data
    print("\n=== Loading Data ===")
    df, genre_to_idx, idx_to_genre = load_genre_data()

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['genre'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['genre'])
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    audio_dir = Path("/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node")

    train_dataset = MERTGenreDataset(
        train_df['track_id'].values, train_df['genre'].values,
        genre_to_idx, audio_dir, processor
    )
    val_dataset = MERTGenreDataset(
        val_df['track_id'].values, val_df['genre'].values,
        genre_to_idx, audio_dir, processor
    )
    test_dataset = MERTGenreDataset(
        test_df['track_id'].values, test_df['genre'].values,
        genre_to_idx, audio_dir, processor
    )

    # Create collate function with processor
    def collate_with_processor(batch):
        return collate_fn(batch, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_with_processor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_with_processor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_with_processor)

    # Model
    print("\n=== Model ===")
    model = MERTClassifier(
        mert_model, num_genres=len(genre_to_idx),
        freeze_encoder=freeze_encoder, layer_idx=layer_idx
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Encoder frozen: {freeze_encoder}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training
    print("\n=== Training ===")
    best_val_acc = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_values = batch['input_values'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['genre_idx'].to(device)

            optimizer.zero_grad()
            logits = model(input_values, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_values = batch['input_values'].to(device)
                lengths = batch['lengths'].to(device)
                labels = batch['genre_idx'].to(device)

                logits = model(input_values, lengths)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.classifier.state_dict(), Path(__file__).parent / 'genre_mert_classifier.pt')
            print(f"  Saved best model")

    # Test
    print("\n=== Test ===")
    model.classifier.load_state_dict(torch.load(Path(__file__).parent / 'genre_mert_classifier.pt'))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_values = batch['input_values'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['genre_idx'].to(device)

            logits = model(input_values, lengths)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
