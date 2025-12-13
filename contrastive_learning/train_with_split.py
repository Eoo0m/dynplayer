import argparse
from pathlib import Path
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from model import (
    PositiveBatcher,
    NormalEmbedding,
    build_pos_mask,
    mp_infonce_loss,
)


# --------------------------
# Metrics
# --------------------------
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def recall_at_k(r, k, n_relevant):
    r = np.asfarray(r)[:k]
    if n_relevant == 0:
        return 0.0
    return np.sum(r) / n_relevant


def get_playlist_embedding(track_indices, track_embeddings):
    if len(track_indices) == 0:
        return np.zeros(track_embeddings.shape[1])
    return np.mean(track_embeddings[track_indices], axis=0)


# --------------------------
# Retrieval Evaluation
# --------------------------
def evaluate_retrieval(
    model, train_dict, test_dict, tid2idx, k_values=[10, 20], device="cpu"
):
    model.eval()

    n_tracks = len(tid2idx)
    with torch.no_grad():
        all_idx = torch.arange(n_tracks, device=device)
        track_embeddings = model(all_idx).cpu().numpy()

    track_emb_tensor = torch.tensor(track_embeddings, dtype=torch.float32).to(device)

    metrics = {f"ndcg@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})

    max_k = max(k_values)

    for playlist_idx in tqdm(test_dict.keys(), desc="Evaluating", leave=False):
        train_tracks = train_dict.get(playlist_idx, [])
        test_tracks = test_dict[playlist_idx]

        if len(train_tracks) == 0 or len(test_tracks) == 0:
            continue

        playlist_emb = get_playlist_embedding(train_tracks, track_embeddings)
        playlist_emb = torch.tensor(playlist_emb, dtype=torch.float32).to(device)

        playlist_emb_norm = playlist_emb / (torch.norm(playlist_emb) + 1e-10)
        track_emb_norm = track_emb_tensor / (
            torch.norm(track_emb_tensor, dim=1, keepdim=True) + 1e-10
        )

        sim = torch.matmul(track_emb_norm, playlist_emb_norm).cpu().numpy()

        # mask out train tracks
        for idx in train_tracks:
            if 0 <= idx < len(sim):
                sim[idx] = -float("inf")

        top_idx = np.argsort(sim)[::-1][:max_k]

        test_set = set(test_tracks)
        relevance = np.array([1 if idx in test_set else 0 for idx in top_idx])

        for k in k_values:
            ndcg = ndcg_at_k(relevance, k)
            rec = recall_at_k(relevance, k, len(test_tracks))
            metrics[f"ndcg@{k}"].append(ndcg)
            metrics[f"recall@{k}"].append(rec)

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


# --------------------------
# Build neighbors ONLY from train
# --------------------------
def build_track_neighbors_from_train(train_dict, idx2tid):

    all_track_indices = set()
    for lst in train_dict.values():
        all_track_indices.update(lst)

    tids = [idx2tid[idx] for idx in sorted(all_track_indices)]
    tid_set = set(tids)

    pos_sets = defaultdict(set)

    for lst in tqdm(train_dict.values(), desc="Building neighbors"):
        track_ids = [idx2tid[idx] for idx in lst]

        for i, t1 in enumerate(track_ids):
            for t2 in track_ids[i + 1 :]:
                pos_sets[t1].add(t2)
                pos_sets[t2].add(t1)

    pos_count = {tid: len(pos_sets.get(tid, set())) for tid in tid_set}
    return tids, pos_sets, pos_count


# ========================
# Retrieval Metrics
# ========================
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def recall_at_k(r, k, n_relevant):
    r = np.asfarray(r)[:k]
    if n_relevant == 0:
        return 0.0
    return np.sum(r) / n_relevant


def get_playlist_embedding(track_indices, track_embeddings):
    if len(track_indices) == 0:
        return np.zeros(track_embeddings.shape[1])
    return np.mean(track_embeddings[track_indices], axis=0)


def evaluate_retrieval(
    model, train_dict, test_dict, tid2idx, k_values=[10, 20], device="cpu"
):
    model.eval()

    n_tracks = len(tid2idx)
    with torch.no_grad():
        all_idx = torch.arange(n_tracks, device=device)
        track_embeddings = model(all_idx).cpu().numpy()

    track_emb_tensor = torch.tensor(track_embeddings, dtype=torch.float32).to(device)

    metrics = {f"ndcg@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})

    max_k = max(k_values)

    for playlist_idx in tqdm(test_dict.keys(), desc="Evaluating", leave=False):
        train_tracks = train_dict.get(playlist_idx, [])
        test_tracks = test_dict[playlist_idx]

        if len(train_tracks) == 0 or len(test_tracks) == 0:
            continue

        playlist_emb = get_playlist_embedding(train_tracks, track_embeddings)
        playlist_emb = torch.tensor(playlist_emb, dtype=torch.float32).to(device)

        playlist_emb_norm = playlist_emb / (torch.norm(playlist_emb) + 1e-10)
        track_emb_norm = track_emb_tensor / (
            torch.norm(track_emb_tensor, dim=1, keepdim=True) + 1e-10
        )

        sim = torch.matmul(track_emb_norm, playlist_emb_norm).cpu().numpy()

        for idx in train_tracks:
            if 0 <= idx < len(sim):
                sim[idx] = -float("inf")

        top_idx = np.argsort(sim)[::-1][:max_k]

        test_set = set(test_tracks)
        relevance = np.array([1 if idx in test_set else 0 for idx in top_idx])

        for k in k_values:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(relevance, k))
            metrics[f"recall@{k}"].append(recall_at_k(relevance, k, len(test_tracks)))

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


# ========================
# Build neighbors (Train Only)
# ========================
def build_track_neighbors_from_train(train_dict, idx2tid):
    all_track_indices = set()
    for lst in train_dict.values():
        all_track_indices.update(lst)

    tids = [idx2tid[idx] for idx in sorted(all_track_indices)]
    tid_set = set(tids)

    pos_sets = defaultdict(set)

    for lst in tqdm(train_dict.values(), desc="Building neighbors"):
        track_ids = [idx2tid[idx] for idx in lst]

        for i, t1 in enumerate(track_ids):
            for t2 in track_ids[i + 1 :]:
                pos_sets[t1].add(t2)
                pos_sets[t2].add(t1)

    pos_count = {tid: len(pos_sets.get(tid, set())) for tid in tid_set}
    return tids, pos_sets, pos_count


# ========================
# MAIN
# ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default="contrastive_learning/train_test_split.npz")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--k-values", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-prefix", default="contrastive_learning/contrastive_eval")
    ap.add_argument("--uniformity-weight", type=float, default=1.0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    split = np.load(args.split_file, allow_pickle=True)
    train_dict = split["train_dict"].item()
    test_dict = split["test_dict"].item()
    all_track_ids = split["all_track_ids"]

    tid2idx_original = {tid: i for i, tid in enumerate(all_track_ids)}
    idx2tid_original = {i: tid for tid, i in tid2idx_original.items()}

    tids, pos_sets, pos_count = build_track_neighbors_from_train(
        train_dict, idx2tid_original
    )

    tid2idx = {t: i for i, t in enumerate(tids)}
    anchors = [t for t in tids if pos_count[t] > 0]

    train_mapped, test_mapped = {}, {}
    for pid, lst in train_dict.items():
        ids = [idx2tid_original[x] for x in lst if idx2tid_original[x] in tid2idx]
        if ids:
            train_mapped[pid] = [tid2idx[t] for t in ids]

    for pid, lst in test_dict.items():
        ids = [idx2tid_original[x] for x in lst if idx2tid_original[x] in tid2idx]
        if ids and pid in train_mapped:
            test_mapped[pid] = [tid2idx[t] for t in ids]

    model = NormalEmbedding(len(tids), args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_ndcg = -1
    metrics_history = []

    # ========================
    # TRAINING LOOP
    # ========================
    for ep in range(1, args.epochs + 1):
        model.train()
        batcher = PositiveBatcher(
            anchors, pos_sets, pos_count, args.batch, tids, args.seed + ep
        )

        total_loss = total_used = iters = 0
        total_pos_sim = total_neg_sim = total_uni = 0.0

        for batch_ids in tqdm(batcher, desc=f"Epoch {ep}/{args.epochs}"):
            idx = torch.tensor([tid2idx[t] for t in batch_ids], device=device)
            z = model(idx)
            pos_mask = build_pos_mask(batch_ids, pos_sets).to(device)

            loss, used, avg_pos_sim, avg_neg_sim, uni_loss = mp_infonce_loss(
                z, args.tau, pos_mask, uniformity_weight=args.uniformity_weight
            )

            if used == 0:
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_used += used
            total_pos_sim += avg_pos_sim
            total_neg_sim += avg_neg_sim
            total_uni += uni_loss
            iters += 1

        avg_loss = total_loss / max(1, iters)
        avg_pos = total_pos_sim / max(1, iters)
        avg_neg = total_neg_sim / max(1, iters)
        avg_uni = total_uni / max(1, iters)

        print(
            f"[Epoch {ep}] loss={avg_loss:.4f}, pos={avg_pos:.4f}, neg={avg_neg:.4f}, uni={avg_uni:.4f}"
        )

        # ========================
        # EVALUATION
        # ========================
        if ep % args.eval_every == 0:
            print(f"\n--- Evaluation @ Epoch {ep} ---")
            results = evaluate_retrieval(
                model,
                train_mapped,
                test_mapped,
                tid2idx,
                k_values=args.k_values,
                device=device,
            )

            for k in sorted(args.k_values):
                print(
                    f"NDCG@{k}: {results[f'ndcg@{k}']:.4f}, Recall@{k}: {results[f'recall@{k}']:.4f}"
                )

            # Save metrics
            record = {
                "epoch": ep,
                "loss": avg_loss,
                "pos_sim": avg_pos,
                "neg_sim": avg_neg,
                "uniformity_loss": avg_uni,
            }

            for k in args.k_values:
                record[f"ndcg@{k}"] = results[f"ndcg@{k}"]
                record[f"recall@{k}"] = results[f"recall@{k}"]

            metrics_history.append(record)

    # ========================
    # SAVE EMBEDDINGS / METRICS
    # ========================
    prefix = Path(args.save_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        Z = model(torch.arange(len(tids), device=device)).cpu().numpy()

    np.save(f"{prefix}_embeddings.npy", Z)
    np.save(f"{prefix}_keys.npy", np.array(tids))
    np.save(f"{prefix}_metrics.npy", metrics_history, allow_pickle=True)

    print("Saved embeddings, keys, metrics, best model.")


if __name__ == "__main__":
    main()
