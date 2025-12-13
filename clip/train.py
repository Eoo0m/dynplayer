import random
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from model import PlaylistCaptionDataset, CaptionPlaylistCLIP


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def embed_text_batch(client, texts, model="text-embedding-3-large"):
    """배치로 텍스트 임베딩"""
    res = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in res.data])


def train_test_split(playlist_ids, title_emb_dict, test_ratio=0.2, seed=42):
    """
    플레이리스트를 train/test로 분리

    Args:
        playlist_ids: 플레이리스트 ID 리스트
        title_emb_dict: 타이틀 임베딩 딕셔너리
        test_ratio: 테스트 비율 (0이면 전량 학습)
        seed: random seed

    Returns:
        train_ids: 학습용 플레이리스트 ID
        test_ids: 테스트용 플레이리스트 ID
        test_titles: 테스트용 플레이리스트 타이틀
    """
    random.seed(seed)
    np.random.seed(seed)

    # 타이틀이 있는 플레이리스트만 필터링
    valid_ids = [pid for pid in playlist_ids if pid in title_emb_dict]

    if test_ratio == 0:
        print(f"Train playlists: {len(valid_ids):,}")
        print("Test playlists: 0 (test_ratio=0, using all data for training)")
        return valid_ids, [], {}

    # Shuffle and split
    shuffled = valid_ids.copy()
    random.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_ids = shuffled[:n_test]
    train_ids = shuffled[n_test:]

    # 테스트 타이틀 저장 (평가용)
    test_titles = {pid: title_emb_dict[pid] for pid in test_ids}

    print(f"Train playlists: {len(train_ids):,}")
    print(f"Test playlists: {len(test_ids):,}")

    return train_ids, test_ids, test_titles


def recall_at_k(
    model,
    test_titles,
    playlist_embs,
    playlist_index,
    device,
    client,
    k_list=[1, 5, 10, 20],
):
    """
    Recall@K 평가: 텍스트 검색으로 플레이리스트를 찾는 정확도

    Args:
        model: 학습된 CLIP 모델
        test_titles: {playlist_id: title_embedding} 테스트 플레이리스트
        playlist_embs: 전체 플레이리스트 임베딩 (LightGCN)
        playlist_index: {playlist_id: index} 매핑
        device: 디바이스
        client: OpenAI 클라이언트
        k_list: 평가할 K 값들

    Returns:
        recall_scores: {k: recall@k} 딕셔너리
    """
    model.eval()

    # 전체 플레이리스트 임베딩 프로젝션
    all_playlist_ids = list(playlist_index.keys())
    all_projected = []

    batch_size = 512
    with torch.no_grad():
        for i in tqdm(
            range(0, len(all_playlist_ids), batch_size),
            desc="Projecting playlists",
        ):
            batch_ids = all_playlist_ids[i : i + batch_size]
            batch_embs = torch.stack(
                [
                    torch.tensor(
                        playlist_embs[playlist_index[pid]], dtype=torch.float32
                    )
                    for pid in batch_ids
                ]
            ).to(device)

            projected = model.playlist_proj(batch_embs)
            all_projected.append(projected)

    all_projected = torch.cat(all_projected, dim=0)  # (N, out_dim)

    # 플레이리스트 ID 매핑
    pid_to_idx = {pid: i for i, pid in enumerate(all_playlist_ids)}

    # 테스트 타이틀 평가
    recall_scores = {k: [] for k in k_list}

    for test_pid, title_emb in tqdm(test_titles.items(), desc="Evaluating Recall@K"):
        if test_pid not in pid_to_idx:
            continue

        # 타이틀 임베딩 프로젝션
        with torch.no_grad():
            title_tensor = (
                torch.tensor(title_emb, dtype=torch.float32).unsqueeze(0).to(device)
            )
            projected_title = model.caption_proj(title_tensor)  # (1, out_dim)

            # 모든 플레이리스트와 유사도 계산
            similarities = torch.matmul(projected_title, all_projected.T).squeeze(
                0
            )  # (N,)

            # Top-K 추출
            max_k = max(k_list)
            _, top_indices = torch.topk(similarities, k=max_k)
            top_indices = top_indices.cpu().numpy()

        # 정답 인덱스
        target_idx = pid_to_idx[test_pid]

        # 각 K에 대해 Recall 계산
        for k in k_list:
            if target_idx in top_indices[:k]:
                recall_scores[k].append(1.0)
            else:
                recall_scores[k].append(0.0)

    # 평균 계산
    avg_recall = {k: np.mean(scores) for k, scores in recall_scores.items()}

    return avg_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--playlist-ids",
        default="lightgcn/baseline_playlist_embeddings_ids.npy",
        help="Playlist IDs corresponding to embeddings",
    )
    parser.add_argument(
        "--playlist-embs",
        default="lightgcn/baseline_playlist_embeddings.npy",
        help="Playlist embeddings (baseline or LightGCN output)",
    )
    parser.add_argument(
        "--playlist-csv",
        default="clip/filtered_playlist_tracks.csv",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default="clip/clip")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ======================================================
    # 1) 플레이리스트 임베딩 로드 (LightGCN)
    # ======================================================
    print("\n=== Loading Playlist Embeddings ===")
    playlist_ids = np.load(args.playlist_ids, allow_pickle=True)
    playlist_embs = np.load(args.playlist_embs)
    playlist_index = {pid: i for i, pid in enumerate(playlist_ids)}

    print(f"Playlist embeddings: {playlist_embs.shape}")
    print(f"Total playlists: {len(playlist_index)}")

    # ======================================================
    # 2) 플레이리스트 메타데이터 로드 (타이틀 포함)
    # ======================================================
    print("\n=== Loading Playlist Metadata ===")
    playlist_df = pd.read_csv(args.playlist_csv)
    print(f"Total playlist records: {len(playlist_df)}")

    # 타이틀이 있고 임베딩이 있는 플레이리스트만 필터링
    playlist_df = playlist_df[playlist_df["playlist_title"].notna()]
    playlist_df = playlist_df[playlist_df["playlist_title"] != ""]
    playlist_df = playlist_df[playlist_df["playlist_id"].isin(playlist_index)]

    print(f"Playlists with titles and embeddings: {len(playlist_df)}")

    # ======================================================
    # 3) GPT로 플레이리스트 타이틀 임베딩 생성
    # ======================================================
    print("\n=== Embedding Playlist Titles with GPT ===")

    all_playlist_ids = playlist_df["playlist_id"].tolist()
    all_titles = playlist_df["playlist_title"].tolist()

    title_embeddings = []
    BATCH = 512
    for i in tqdm(range(0, len(all_titles), BATCH), desc="Embedding titles"):
        batch_titles = all_titles[i : i + BATCH]
        batch_embs = embed_text_batch(client, batch_titles)
        title_embeddings.append(batch_embs)

    title_embeddings = np.vstack(title_embeddings)
    title_emb_dict = {
        pid: title_embeddings[i] for i, pid in enumerate(all_playlist_ids)
    }

    caption_dim = title_embeddings.shape[1]
    print(f"Playlist title embeddings: {title_embeddings.shape}")
    print(f"Caption dim: {caption_dim}")

    # ======================================================
    # 4) Train/Test Split
    # ======================================================
    print(f"\n=== Train/Test Split (test_ratio={args.test_ratio}) ===")
    train_ids, test_ids, test_titles = train_test_split(
        all_playlist_ids, title_emb_dict, test_ratio=args.test_ratio, seed=args.seed
    )

    # ======================================================
    # 5) Model & Dataset
    # ======================================================
    playlist_dim = playlist_embs.shape[1]
    model = CaptionPlaylistCLIP(
        caption_dim, playlist_dim, out_dim=args.output_dim, temperature=0.07
    ).to(device)

    print("\n" + "=" * 80)
    print("🎯 Playlist Caption ↔ Playlist Embedding Contrastive Learning")
    print("=" * 80)
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Strategy: Playlist Title (GPT) ↔ Playlist Embedding (LightGCN)")

    # Dataset 생성 (train만 사용)
    dataset = PlaylistCaptionDataset(
        train_ids, title_emb_dict, playlist_embs, playlist_index
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nDataset size: {len(dataset)} playlists")

    # ======================================================
    # 6) Training
    # ======================================================
    best_recall = 0.0
    best_recall_scores = {}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        avg_pos = 0
        avg_neg = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for caption_emb, playlist_emb in pbar:
            caption_emb = caption_emb.to(device)
            playlist_emb = playlist_emb.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, pos_sim, neg_sim = model(caption_emb, playlist_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_pos += pos_sim
            avg_neg += neg_sim

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "pos": f"{pos_sim:.4f}",
                    "neg": f"{neg_sim:.4f}",
                }
            )

        batches = len(loader)
        print(
            f"[Epoch {epoch+1}] loss={total_loss/batches:.4f} | pos_sim={avg_pos/batches:.4f} | neg_sim={avg_neg/batches:.4f}"
        )

        # Evaluation (skip if test_ratio is 0)
        if args.test_ratio > 0 and (
            (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs
        ):
            print(f"\n--- Evaluation at Epoch {epoch+1} ---")
            recall_scores = recall_at_k(
                model,
                test_titles,
                playlist_embs,
                playlist_index,
                device,
                client,
                k_list=[1, 5, 10, 20],
            )

            print("Recall@K:")
            for k, score in recall_scores.items():
                print(f"  Recall@{k}: {score:.4f}")

            # Best model 저장
            if recall_scores[10] > best_recall:
                best_recall = recall_scores[10]
                best_recall_scores = recall_scores.copy()
                print(f"\n✅ New best Recall@10: {best_recall:.4f}")
                torch.save(model.state_dict(), f"{args.output_prefix}_best.pt")

    # ======================================================
    # 7) Final Evaluation
    # ======================================================
    if args.test_ratio > 0:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        final_recall = recall_at_k(
            model,
            test_titles,
            playlist_embs,
            playlist_index,
            device,
            client,
            k_list=[1, 5, 10, 20, 50],
        )

        print("\nFinal Recall@K:")
        for k, score in final_recall.items():
            print(f"  Recall@{k}: {score:.4f}")

        if best_recall_scores:
            print(f"\nBest Epoch Recall@K:")
            for k, score in best_recall_scores.items():
                print(f"  Recall@{k}: {score:.4f}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Training completed (no evaluation, test_ratio=0)")
        print("=" * 60)

    # 모델 저장
    torch.save(model.state_dict(), f"{args.output_prefix}.pt")
    print(f"\n✅ Model saved: {args.output_prefix}.pt")

    # ======================================================
    # 8) 플레이리스트 임베딩 프로젝션 저장
    # ======================================================
    print("\n=== Generating Projected Playlist Embeddings ===")
    model.eval()

    all_projected = []
    batch_size_infer = 512
    with torch.no_grad():
        for i in tqdm(range(0, len(playlist_ids), batch_size_infer)):
            batch_ids = playlist_ids[i : i + batch_size_infer]

            batch_embs = torch.stack(
                [
                    torch.tensor(
                        playlist_embs[playlist_index[pid]], dtype=torch.float32
                    )
                    for pid in batch_ids
                ]
            ).to(device)

            projected = model.playlist_proj(batch_embs).cpu().numpy()
            all_projected.append(projected)

    all_projected = np.vstack(all_projected)

    # 저장
    np.save(f"{args.output_prefix}_playlist_ids.npy", playlist_ids)
    np.save(f"{args.output_prefix}_playlist_projected.npy", all_projected)

    print(f"\n Saved playlist embeddings: {all_projected.shape}")
    print(f"   - {args.output_prefix}_playlist_ids.npy")
    print(f"   - {args.output_prefix}_playlist_projected.npy")


if __name__ == "__main__":
    main()
