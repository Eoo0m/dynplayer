"""
Compare Contrastive Learning (basic) vs LightGCN on two datasets
- Uses existing code: contrastive_learning new/ and lightgcn_sparse/
- Just runs commands sequentially and collects results
"""

import json
import subprocess
from pathlib import Path
import numpy as np
from datetime import datetime

# Configuration
DATASETS = [
    {
        "name": "min2_win10",
        "csv_path": "/Users/eomjoonseo/dynplayer_crawler/preprocessing/track_playlist_counts_min2_win10.csv",
    },
    {
        "name": "min5_win10",
        "csv_path": "/Users/eomjoonseo/dynplayer_crawler/preprocessing/track_playlist_counts_min5_win10.csv",
    },
]

EPOCHS = 100
EMBEDDING_DIM = 128
SEED = 42

OUTPUT_DIR = Path("/Users/eomjoonseo/dynplayer/comparison_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_contrastive_experiment(dataset_name, csv_path):
    """Run contrastive learning experiment using existing code"""
    print(f"\n{'='*80}")
    print(f"🚀 Running Contrastive Learning on {dataset_name}")
    print(f"{'='*80}")

    base_dir = Path(f"contrastive_learning new")
    output_prefix = OUTPUT_DIR / f"{dataset_name}_contrastive"
    split_file = OUTPUT_DIR / f"{dataset_name}_contrastive_split.npz"

    # Step 1: Create train/test split
    print(f"\n[1/2] Creating train/test split...")
    cmd_split = [
        "python",
        str(base_dir / "make_train_test_split.py"),
        "--csv",
        csv_path,
        "--output",
        str(split_file),
        "--seed",
        str(SEED),
    ]
    print(f"  Command: {' '.join(cmd_split)}")
    result = subprocess.run(cmd_split)
    if result.returncode != 0:
        print(f"  ❌ Failed to create split")
        return None

    # Step 2: Train model
    print(f"\n[2/2] Training model...")
    cmd_train = [
        "python",
        str(base_dir / "train_with_split.py"),
        "--split-file",
        str(split_file),
        "--dim",
        str(EMBEDDING_DIM),
        "--epochs",
        str(EPOCHS),
        "--eval-every",
        "10",
        "--seed",
        str(SEED),
        "--save-prefix",
        str(output_prefix),
        "--uniformity-weight",
        "0.0",  # Basic version
    ]
    print(f"  Command: {' '.join(cmd_train)}")
    result = subprocess.run(cmd_train)
    if result.returncode != 0:
        print(f"  ❌ Failed to train")
        return None

    print(f"\n✅ Contrastive experiment completed")

    # Load metrics
    metrics_file = f"{output_prefix}_metrics.npy"
    if not Path(metrics_file).exists():
        print(f"  ⚠️  Metrics file not found: {metrics_file}")
        return None

    metrics = np.load(metrics_file, allow_pickle=True)
    best_epoch = max(metrics, key=lambda x: x.get("ndcg@10", 0))

    return {
        "dataset": dataset_name,
        "model": "contrastive_basic",
        "best_epoch": int(best_epoch["epoch"]),
        "best_metrics": {
            "ndcg@10": float(best_epoch.get("ndcg@10", 0)),
            "ndcg@20": float(best_epoch.get("ndcg@20", 0)),
            "recall@10": float(best_epoch.get("recall@10", 0)),
            "recall@20": float(best_epoch.get("recall@20", 0)),
        },
        "all_epochs": [
            {
                "epoch": int(m["epoch"]),
                "ndcg@10": float(m.get("ndcg@10", 0)),
                "ndcg@20": float(m.get("ndcg@20", 0)),
                "recall@10": float(m.get("recall@10", 0)),
                "recall@20": float(m.get("recall@20", 0)),
            }
            for m in metrics
        ],
    }


def run_lightgcn_experiment(dataset_name, csv_path):
    """Run LightGCN experiment using existing code"""
    print(f"\n{'='*80}")
    print(f"🚀 Running LightGCN on {dataset_name}")
    print(f"{'='*80}")

    base_dir = Path("lightgcn_sparse")
    graph_dir = OUTPUT_DIR / f"{dataset_name}_lightgcn_graph"
    graph_dir.mkdir(exist_ok=True)
    output_prefix = OUTPUT_DIR / f"{dataset_name}_lightgcn"

    # Step 1: Prepare graph
    print(f"\n[1/2] Preparing graph...")
    cmd_prepare = [
        "python",
        str(base_dir / "prepare_graph.py"),
        "--csv",
        csv_path,
        "--output-dir",
        str(graph_dir),
        "--seed",
        str(SEED),
    ]
    print(f"  Command: {' '.join(cmd_prepare)}")
    result = subprocess.run(cmd_prepare)
    if result.returncode != 0:
        print(f"  ❌ Failed to prepare graph")
        return None

    # Step 2: Train model
    print(f"\n[2/2] Training model...")
    cmd_train = [
        "python",
        str(base_dir / "train.py"),
        "--data-dir",
        str(graph_dir),
        "--dim",
        str(EMBEDDING_DIM),
        "--epochs",
        str(EPOCHS),
        "--seed",
        str(SEED),
        "--output-prefix",
        str(output_prefix),
    ]
    print(f"  Command: {' '.join(cmd_train)}")
    result = subprocess.run(cmd_train)
    if result.returncode != 0:
        print(f"  ❌ Failed to train")
        return None

    print(f"\n✅ LightGCN experiment completed")

    # Load metrics
    metrics_file = f"{output_prefix}_metrics.npy"
    if not Path(metrics_file).exists():
        print(f"  ⚠️  Metrics file not found: {metrics_file}")
        return None

    metrics = np.load(metrics_file, allow_pickle=True)
    best_epoch = max(metrics, key=lambda x: x.get("ndcg@10", 0))

    return {
        "dataset": dataset_name,
        "model": "lightgcn",
        "best_epoch": int(best_epoch["epoch"]),
        "best_metrics": {
            "ndcg@10": float(best_epoch.get("ndcg@10", 0)),
            "ndcg@20": float(best_epoch.get("ndcg@20", 0)),
            "recall@10": float(best_epoch.get("recall@10", 0)),
            "recall@20": float(best_epoch.get("recall@20", 0)),
        },
        "all_epochs": [
            {
                "epoch": int(m["epoch"]),
                "ndcg@10": float(m.get("ndcg@10", 0)),
                "ndcg@20": float(m.get("ndcg@20", 0)),
                "recall@10": float(m.get("recall@10", 0)),
                "recall@20": float(m.get("recall@20", 0)),
            }
            for m in metrics
        ],
    }


def main():
    print("=" * 80)
    print("🎯 Model Comparison: Contrastive (basic) vs LightGCN")
    print("=" * 80)

    results = []

    for dataset in DATASETS:
        dataset_name = dataset["name"]
        csv_path = dataset["csv_path"]

        print(f"\n\n{'#'*80}")
        print(f"# Dataset: {dataset_name}")
        print(f"# CSV: {csv_path}")
        print(f"{'#'*80}")

        # Run experiments
        contrastive_result = run_contrastive_experiment(dataset_name, csv_path)
        if contrastive_result:
            results.append(contrastive_result)

        lightgcn_result = run_lightgcn_experiment(dataset_name, csv_path)
        if lightgcn_result:
            results.append(lightgcn_result)

    # Save results to JSON
    output_file = (
        OUTPUT_DIR
        / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Results saved to {output_file}")
    print(f"{'='*80}")

    # Print summary
    print("\n📊 Summary:")
    for result in results:
        print(f"\n{result['dataset']} - {result['model']}:")
        if "best_metrics" in result:
            print(f"  Best epoch: {result['best_epoch']}")
            for k, v in result["best_metrics"].items():
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
