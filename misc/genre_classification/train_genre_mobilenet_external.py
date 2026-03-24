"""
Genre classification using PANNs MobileNetV2
For external GPU training (Colab) - configure paths at the top
Preprocesses audio to numpy on first run for faster loading
"""

import os

# ============================================================
# CONFIGURATION - Set these paths before running
# ============================================================
BASE_DIR = "/content/drive/MyDrive/dynplayer/genre_classification"
AUDIO_DIR = "/content/drive/MyDrive/preview_audio_node"
NPY_DIR = "/content/audio_npy"  # Local storage for preprocessed numpy
# ============================================================

CSV_PATH = os.path.join(BASE_DIR, "spotify_genre_info_frequent.csv")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "MobileNetV2.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torchaudio
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

SAMPLE_RATE = 32000
DURATION = 10
TARGET_LENGTH = SAMPLE_RATE * DURATION


# ============================================================
# PANNs Model Code (MobileNetV2)
# ============================================================

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super(MobileNetV2, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True
        )

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2
        )

        self.bn0 = nn.BatchNorm2d(64)

        width_mult = 1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.fc1 = nn.Linear(1280, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.features(x)

        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict


# ============================================================
# Dataset and Training Code
# ============================================================

class AudioGenreDataset(Dataset):
    def __init__(self, track_ids, genres, genre_to_idx, npy_dir):
        self.track_ids = track_ids
        self.genres = genres
        self.genre_to_idx = genre_to_idx
        self.npy_dir = Path(npy_dir)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        genre = self.genres[idx]
        genre_idx = self.genre_to_idx[genre]

        npy_path = self.npy_dir / f"{track_id}.npy"
        waveform = np.load(npy_path)

        return {
            "audio": torch.from_numpy(waveform),
            "genre_idx": genre_idx,
        }


class GenreClassifier(nn.Module):
    def __init__(self, embedding_dim=1024, num_genres=192):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_genres),
        )

    def forward(self, x):
        return self.classifier(x)


def load_mobilenet(checkpoint_path, device):
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
    return model


def process_single_audio(args):
    """Process a single audio file (for parallel processing)"""
    track_id, audio_dir, npy_dir = args
    audio_path = os.path.join(audio_dir, f"{track_id}.mp3")
    npy_path = os.path.join(npy_dir, f"{track_id}.npy")

    try:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = waveform.squeeze().numpy()

        if len(waveform) < TARGET_LENGTH:
            waveform = np.pad(waveform, (0, TARGET_LENGTH - len(waveform)))
        else:
            waveform = waveform[:TARGET_LENGTH]

        np.save(npy_path, waveform.astype(np.float32))
        return track_id, True
    except Exception as e:
        return track_id, False


def preprocess_audio(df, audio_dir, npy_dir, num_workers=8):
    """Convert mp3 to numpy arrays for faster loading (parallel)"""
    audio_dir = Path(audio_dir)
    npy_dir = Path(npy_dir)
    npy_dir.mkdir(parents=True, exist_ok=True)

    # Get existing audio files (fast listdir instead of glob)
    print(f"Scanning audio directory: {audio_dir}...")
    audio_files = set(f[:-4] for f in os.listdir(audio_dir) if f.endswith(".mp3"))
    print(f"Found {len(audio_files):,} audio files")

    # Get already processed files
    processed_files = set(f[:-4] for f in os.listdir(npy_dir) if f.endswith(".npy")) if npy_dir.exists() else set()
    print(f"Already processed: {len(processed_files):,}")

    # Filter tracks that have audio
    track_ids = set(df["track_id"].values)
    valid_track_ids = list(track_ids & audio_files)
    print(f"Tracks with audio: {len(valid_track_ids):,}")

    # Find tracks that need processing
    to_process = [t for t in valid_track_ids if t not in processed_files]
    print(f"To process: {len(to_process):,}")

    if to_process:
        print(f"Preprocessing audio to {npy_dir} with {num_workers} workers...")
        args_list = [(t, str(audio_dir), str(npy_dir)) for t in to_process]

        failed = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_audio, args): args[0] for args in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                track_id, success = future.result()
                if not success:
                    failed.append(track_id)

        # Remove failed tracks
        for t in failed:
            if t in valid_track_ids:
                valid_track_ids.remove(t)
        print(f"Failed: {len(failed)}")

    return valid_track_ids


def load_genre_data():
    """Load genre data from CSV"""
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df):,}")

    # Filter rows with genres
    df = df[df["genres"].notna() & (df["genres"] != "")]
    df = df.rename(columns={"genres": "genre"})
    print(f"Tracks with genres: {len(df):,}")

    return df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Hyperparameters
    batch_size = 64  # Safe batch size for 22GB GPU
    num_epochs = 10
    lr = 1e-3

    # Load MobileNetV2 (frozen)
    print("\n=== Loading MobileNetV2 ===")
    encoder = load_mobilenet(CHECKPOINT_PATH, device)

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"MobileNetV2 parameters: {num_params:,} (frozen)")

    # Load data
    print("\n=== Loading Data ===")
    df = load_genre_data()

    # Preprocess audio to numpy
    print("\n=== Preprocessing Audio ===")
    valid_track_ids = preprocess_audio(df, AUDIO_DIR, NPY_DIR)
    print(f"Valid tracks: {len(valid_track_ids):,}")

    # Filter to valid tracks
    df = df[df["track_id"].isin(valid_track_ids)]
    print(f"Tracks with audio: {len(df):,}")

    genres = sorted(df["genre"].unique())
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}
    print(f"Genres: {len(genres)}")

    # Split 80:10:10
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["genre"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["genre"])
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    train_dataset = AudioGenreDataset(train_df["track_id"].values, train_df["genre"].values, genre_to_idx, NPY_DIR)
    val_dataset = AudioGenreDataset(val_df["track_id"].values, val_df["genre"].values, genre_to_idx, NPY_DIR)
    test_dataset = AudioGenreDataset(test_df["track_id"].values, test_df["genre"].values, genre_to_idx, NPY_DIR)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Classifier
    print("\n=== Classifier ===")
    classifier = GenreClassifier(embedding_dim=1024, num_genres=len(genre_to_idx)).to(device)
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)  # Only classifier


    # Training
    print("\n=== Training ===")
    best_val_acc = 0

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        total_loss = 0
        all_preds, all_labels = [], []
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            audio = batch["audio"].to(device, non_blocking=True)
            labels = batch["genre_idx"].to(device, non_blocking=True)

            with torch.no_grad():
                output = encoder(audio)
                embeddings = output["embedding"]

            optimizer.zero_grad()
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 20 == 0:
                acc = accuracy_score(all_labels, all_preds)
                print(f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f} Acc: {acc:.4f}")

        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = total_loss / num_batches

        # Validation
        encoder.eval()
        classifier.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                labels = batch["genre_idx"].to(device)

                output = encoder(audio)
                embeddings = output["embedding"]
                logits = classifier(embeddings)

                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'classifier': classifier.state_dict(),
                'genre_to_idx': genre_to_idx,
                'idx_to_genre': idx_to_genre,
            }, os.path.join(OUTPUT_DIR, "genre_mobilenet_best.pt"))
            print(f"  Saved best model")

    # Test
    print("\n=== Test ===")
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "genre_mobilenet_best.pt"))
    classifier.load_state_dict(checkpoint['classifier'])
    encoder.eval()
    classifier.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio"].to(device)
            labels = batch["genre_idx"].to(device)

            output = encoder(audio)
            embeddings = output["embedding"]
            logits = classifier(embeddings)

            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="weighted")
    print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
