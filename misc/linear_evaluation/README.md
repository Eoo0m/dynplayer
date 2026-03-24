# Linear Evaluation: Genre Classification with HTS-AT

This directory contains code for training a genre classifier using frozen HTS-AT audio embeddings.

## Overview

**Task**: Multi-class genre classification on Spotify tracks

**Model Architecture**:
- **Audio Encoder**: HTS-AT (Hierarchical Token-Semantic Audio Transformer) pretrained on AudioSet
  - Frozen during training
  - Extracts 527-dimensional audio features
- **Classifier**: 3-layer MLP
  - Input: 527-dim HTS-AT embeddings
  - Hidden layers: 512 → 256
  - Output: 192 genre classes
  - Dropout: 0.3

## Data Preprocessing

### Step 1: Preprocess Genre Data

```bash
cd /Users/eomjoonseo/dynplayer
python preprocess_genre_data.py
```

**What it does**:
- Reads `/data/spotify_genre_info_sampled.csv`
- Extracts first genre for each track (e.g., "k-pop, pop" → "k-pop")
- Filters to keep only genres with ≥50 occurrences
- Outputs:
  - `data/spotify_genre_info_frequent.csv`: Filtered track-genre pairs
  - `data/genre_list.txt`: List of all genres

**Results**:
- Total tracks: 81,701
- Tracks with frequent genres: 53,617 (65.6%)
- Number of genres: 192

**Top genres**:
1. k-pop: 5,832 tracks
2. k-ballad: 4,716 tracks
3. k-rap: 2,525 tracks
4. phonk: 1,138 tracks
5. k-rock: 1,113 tracks

## Model Setup

### HTS-AT Checkpoint

Download the pretrained HTS-AT checkpoint:
- **File**: `HTSAT_AudioSet_Saved_6.ckpt` (121 MB)
- **Source**: [HTS-Audio-Transformer repository](https://github.com/RetroCirce/HTS-Audio-Transformer)
- **Location**: `/Users/eomjoonseo/dynplayer/HTS-Audio-Transformer/HTSAT_AudioSet_Saved_6.ckpt`

The checkpoint is already placed in the correct location.

## Training

### Requirements

Install dependencies:
```bash
pip install torch torchaudio librosa scikit-learn pandas tqdm
```

### Run Training

```bash
cd /Users/eomjoonseo/dynplayer/linear_evaluation
python train_genre_classifier.py
```

**Training Configuration**:
- **Batch size**: 16
- **Learning rate**: 1e-3
- **Epochs**: 20
- **Optimizer**: Adam
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Loss**: CrossEntropyLoss

**Data Split**:
- Train: 72%
- Validation: 8%
- Test: 20%
- Stratified by genre

### Important Notes

1. **Audio Files**: The script automatically filters out tracks without audio files
   - Audio directory: `/data/preview_audio_node/`
   - Format: `.mp3`
   - Total available: 16,721 audio files
   - Tracks with both genre and audio: **15,114** (28.2% of genre-labeled tracks)
   - All 192 genres are represented in the audio dataset

2. **Audio Preprocessing**:
   - Sample rate: 32 kHz
   - Duration: 10 seconds
   - Padding/trimming applied to match target length

3. **Frozen Encoder**: The HTS-AT model is frozen during training. Only the linear classifier is trained.

## Outputs

After training, the following files will be generated in `linear_evaluation/`:

- **`genre_classifier_best.pt`**: Best model checkpoint
  - Contains: model state, optimizer state, genre mappings, validation metrics

- **`test_results.npy`**: Test set evaluation results
  - Contains: predictions, labels, test accuracy, F1 score

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **F1 Score (weighted)**: Accounts for class imbalance
- **Classification Report**: Per-class precision, recall, F1

## Using the Trained Model

Load the trained model for inference:

```python
import torch
from train_genre_classifier import LinearGenreClassifier

# Load checkpoint
checkpoint = torch.load('linear_evaluation/genre_classifier_best.pt')

# Create model
num_genres = len(checkpoint['genre_to_idx'])
model = LinearGenreClassifier(embedding_dim=527, num_genres=num_genres)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get genre mappings
genre_to_idx = checkpoint['genre_to_idx']
idx_to_genre = checkpoint['idx_to_genre']

# Use for inference
# embeddings = extract_embeddings_from_audio(audio)
# logits = model(embeddings)
# predicted_genre_idx = logits.argmax(dim=1).item()
# predicted_genre = idx_to_genre[predicted_genre_idx]
```

## Dataset Statistics

**Genre Distribution** (Top 20):
| Rank | Genre | Count |
|------|-------|-------|
| 1 | k-pop | 5,832 |
| 2 | k-ballad | 4,716 |
| 3 | k-rap | 2,525 |
| 4 | phonk | 1,138 |
| 5 | k-rock | 1,113 |
| 6 | j-pop | 1,008 |
| 7 | jazz | 956 |
| 8 | country | 884 |
| 9 | rap | 877 |
| 10 | reggaeton | 807 |
| 11 | anime | 790 |
| 12 | bollywood | 721 |
| 13 | classic rock | 690 |
| 14 | motown | 671 |
| 15 | christmas | 656 |
| 16 | r&b | 653 |
| 17 | new wave | 652 |
| 18 | edm | 603 |
| 19 | opm | 583 |
| 20 | vocaloid | 579 |

## Troubleshooting

### No audio files found

If you see the warning "No audio files found", ensure:
1. Audio files are in `/Users/eomjoonseo/dynplayer/data/audio/`
2. Files are named as `{track_id}.{ext}` where ext is mp3, wav, m4a, or flac
3. Track IDs match those in the CSV file

### Out of memory

If you encounter OOM errors:
- Reduce `batch_size` in the training script
- Use CPU instead of GPU if available

### HTS-AT import errors

If you see import errors for HTS-AT modules:
- Ensure `HTS-Audio-Transformer` is in the parent directory
- Check that all required dependencies are installed

## References

- **HTS-AT Paper**: [HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection](https://arxiv.org/abs/2202.00874)
- **HTS-AT Code**: https://github.com/RetroCirce/HTS-Audio-Transformer
- **AudioSet**: https://research.google.com/audioset/
