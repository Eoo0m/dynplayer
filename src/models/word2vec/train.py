"""
Train Word2Vec model on playlist sequences with LOO evaluation.

- Train on train playlists only
- Evaluate every 5 epochs using LOO (Leave-One-Out) test set
- Save best model based on Recall@10

Usage:
    python word2vec/train.py --data-dir word2vec/data --epochs 50 --eval-every 5
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class EvaluationCallback(CallbackAny2Vec):
    """Callback to evaluate and log metrics during training."""

    def __init__(self, eval_every, test_observed, test_masked, track_to_idx, num_tracks, k_values):
        self.eval_every = eval_every
        self.test_observed = test_observed
        self.test_masked = test_masked
        self.track_to_idx = track_to_idx
        self.idx_to_track = {v: k for k, v in track_to_idx.items()}
        self.num_tracks = num_tracks
        self.k_values = k_values

        self.epoch = 0
        self.loss_prev = 0
        self.metrics_history = []
        self.best_recall = -1
        self.best_model_wv = None

    def on_epoch_end(self, model):
        # Log loss (cumulative -> per-epoch)
        loss = model.get_latest_training_loss()
        epoch_loss = loss - self.loss_prev
        self.loss_prev = loss

        self.epoch += 1
        print(f"[Epoch {self.epoch}] Loss = {epoch_loss:,.0f}")

        # Evaluate every N epochs
        if self.epoch % self.eval_every == 0:
            metrics = self.evaluate(model)

            record = {"epoch": self.epoch, "loss": epoch_loss}
            for k in self.k_values:
                record[f"recall@{k}"] = metrics[f"recall@{k}"]
                record[f"ndcg@{k}"] = metrics[f"ndcg@{k}"]
            self.metrics_history.append(record)

            # Print metrics
            print(f"\n--- Evaluation @ Epoch {self.epoch} ---")
            for k in self.k_values:
                print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}, NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")

            # Save best model
            if metrics[f"recall@{self.k_values[0]}"] > self.best_recall:
                self.best_recall = metrics[f"recall@{self.k_values[0]}"]
                # Deep copy word vectors
                self.best_model_wv = {word: model.wv[word].copy() for word in model.wv.index_to_key}
                print(f"✅ New best model (Recall@{self.k_values[0]}={self.best_recall:.4f})")
            print()

    def evaluate(self, model):
        """
        Evaluate using LOO test set.

        For each test playlist:
        1. Compute playlist embedding as average of observed track embeddings
        2. Rank all tracks by similarity
        3. Check if masked track is in top-K
        """
        recalls = {k: [] for k in self.k_values}
        ndcgs = {k: [] for k in self.k_values}

        # Build embedding matrix once (much faster than per-query lookup)
        vocab_list = model.wv.index_to_key
        vocab_set = set(vocab_list)
        track_to_vocab_idx = {t: i for i, t in enumerate(vocab_list)}

        # Get normalized embedding matrix
        embedding_matrix = model.wv.vectors.copy()
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-10
        embedding_matrix = embedding_matrix / norms

        max_k = max(self.k_values)

        for pid, observed_indices in tqdm(self.test_observed.items(), desc="Evaluating", leave=False):
            masked_idx = self.test_masked[pid]

            # Convert indices to track_ids
            observed_tracks = [self.idx_to_track[idx] for idx in observed_indices if idx in self.idx_to_track]
            masked_track = self.idx_to_track.get(masked_idx)

            if masked_track is None or masked_track not in vocab_set:
                continue

            # Filter observed tracks in vocab
            observed_in_vocab = [t for t in observed_tracks if t in vocab_set]
            if len(observed_in_vocab) == 0:
                continue

            # Compute playlist embedding (average of observed tracks)
            observed_vocab_indices = [track_to_vocab_idx[t] for t in observed_in_vocab]
            playlist_emb = embedding_matrix[observed_vocab_indices].mean(axis=0)
            playlist_emb = playlist_emb / (np.linalg.norm(playlist_emb) + 1e-10)

            # Compute similarities to all tracks (matrix multiplication)
            scores = embedding_matrix @ playlist_emb

            # Mask observed tracks
            for idx in observed_vocab_indices:
                scores[idx] = -np.inf

            # Get top-K indices
            top_indices = np.argpartition(scores, -max_k)[-max_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            topk_tracks = [vocab_list[i] for i in top_indices]

            for k in self.k_values:
                top_k = topk_tracks[:k]
                hit = 1 if masked_track in top_k else 0
                recalls[k].append(hit)

                # NDCG
                if masked_track in top_k:
                    rank = top_k.index(masked_track) + 1
                    ndcg = 1.0 / np.log2(rank + 1)
                else:
                    ndcg = 0.0
                ndcgs[k].append(ndcg)

        metrics = {}
        for k in self.k_values:
            metrics[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
            metrics[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0

        return metrics


def load_data(data_dir):
    """Load sequences and LOO split."""
    data_dir = Path(data_dir)

    # Load sequences
    with open(data_dir / "sequences.pkl", 'rb') as f:
        sequences = pickle.load(f)
    print(f"Loaded {len(sequences):,} sequences")

    # Load LOO split
    split = np.load(data_dir / "loo_split.npz", allow_pickle=True)
    test_observed = split["test_playlist_observed"].item()
    test_masked = split["test_playlist_masked"].item()
    track_to_idx = split["track_to_idx"].item()
    num_tracks = int(split["num_tracks"])

    print(f"Test playlists: {len(test_observed):,}")
    print(f"Total tracks: {num_tracks:,}")

    # Load track info
    with open(data_dir / "track_info.pkl", 'rb') as f:
        track_info = pickle.load(f)

    return sequences, test_observed, test_masked, track_to_idx, num_tracks, track_info


def train_word2vec(
    sequences,
    test_observed,
    test_masked,
    track_to_idx,
    num_tracks,
    vector_size=128,
    window=10,
    min_count=5,
    sg=1,
    negative=10,
    epochs=50,
    workers=4,
    seed=42,
    eval_every=5,
    k_values=[10, 20],
):
    """Train Word2Vec with periodic evaluation."""
    print(f"\n=== Training Word2Vec ===")
    print(f"  vector_size: {vector_size}")
    print(f"  window: {window}")
    print(f"  min_count: {min_count}")
    print(f"  sg (Skip-gram): {sg}")
    print(f"  negative: {negative}")
    print(f"  epochs: {epochs}")
    print(f"  eval_every: {eval_every}")

    callback = EvaluationCallback(
        eval_every=eval_every,
        test_observed=test_observed,
        test_masked=test_masked,
        track_to_idx=track_to_idx,
        num_tracks=num_tracks,
        k_values=k_values,
    )

    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        epochs=epochs,
        workers=workers,
        seed=seed,
        compute_loss=True,
        callbacks=[callback]
    )

    print(f"\nVocabulary size: {len(model.wv):,} tracks")

    return model, callback


def save_model(model, callback, output_dir, track_info):
    """Save trained model, best model, and embeddings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full model (for continued training)
    model_path = output_dir / "word2vec.model"
    model.save(str(model_path))
    print(f"Saved model to {model_path}")

    # Save final embeddings
    track_ids = list(model.wv.index_to_key)
    embeddings = np.array([model.wv[tid] for tid in track_ids])

    np.save(output_dir / "track_embeddings.npy", embeddings)
    np.save(output_dir / "track_ids.npy", np.array(track_ids))
    print(f"Saved final embeddings: {embeddings.shape}")

    # Save best embeddings
    if callback.best_model_wv:
        best_track_ids = list(callback.best_model_wv.keys())
        best_embeddings = np.array([callback.best_model_wv[tid] for tid in best_track_ids])
        np.save(output_dir / "track_embeddings_best.npy", best_embeddings)
        np.save(output_dir / "track_ids_best.npy", np.array(best_track_ids))
        print(f"Saved best embeddings: {best_embeddings.shape}")

    # Save track_id to index mapping
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    with open(output_dir / "track_to_idx.pkl", 'wb') as f:
        pickle.dump(track_to_idx, f)

    # Save config
    config = {
        'vector_size': model.wv.vector_size,
        'window': model.window,
        'min_count': model.min_count,
        'sg': model.sg,
        'negative': model.negative,
        'epochs': model.epochs,
        'vocab_size': len(model.wv),
        'best_recall': callback.best_recall,
    }
    with open(output_dir / "config.pkl", 'wb') as f:
        pickle.dump(config, f)

    # Save metrics history
    np.save(output_dir / "metrics_history.npy", callback.metrics_history, allow_pickle=True)
    print(f"Saved metrics history: {len(callback.metrics_history)} records")


def test_similarity(model, track_info):
    """Test model by finding similar tracks."""
    print("\n=== Testing Similarity ===")

    sample_tracks = list(model.wv.index_to_key)[:5]

    for track_id in sample_tracks:
        info = track_info.get(track_id, {})
        track_name = info.get('track', 'Unknown')
        artist = info.get('artist', 'Unknown')

        print(f"\nQuery: {track_name} - {artist}")
        print("Similar tracks:")

        similar = model.wv.most_similar(track_id, topn=5)
        for sim_id, score in similar:
            sim_info = track_info.get(sim_id, {})
            sim_name = sim_info.get('track', 'Unknown')
            sim_artist = sim_info.get('artist', 'Unknown')
            print(f"  {score:.4f} | {sim_name} - {sim_artist}")


def main():
    parser = argparse.ArgumentParser(description="Train Word2Vec with LOO evaluation")
    parser.add_argument("--data-dir", type=str, default="word2vec/data",
                        help="Directory with prepared sequences and LOO split")
    parser.add_argument("--output-dir", type=str, default="word2vec/outputs",
                        help="Output directory for model")
    parser.add_argument("--vector-size", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--window", type=int, default=10,
                        help="Context window size")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Minimum track frequency")
    parser.add_argument("--sg", type=int, default=1,
                        help="1 for Skip-gram, 0 for CBOW")
    parser.add_argument("--negative", type=int, default=10,
                        help="Number of negative samples")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 20],
                        help="K values for Recall/NDCG")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Load data
    sequences, test_observed, test_masked, track_to_idx, num_tracks, track_info = load_data(args.data_dir)

    # Train model
    model, callback = train_word2vec(
        sequences,
        test_observed,
        test_masked,
        track_to_idx,
        num_tracks,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        negative=args.negative,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
        eval_every=args.eval_every,
        k_values=args.k_values,
    )

    # Save model
    save_model(model, callback, args.output_dir, track_info)

    # Test similarity
    test_similarity(model, track_info)

    print(f"\n✅ Training complete!")
    print(f"Best Recall@{args.k_values[0]}: {callback.best_recall:.4f}")


if __name__ == "__main__":
    main()
