"""
Compare metrics between contrastive_new and simgcl models
"""
import numpy as np
from pathlib import Path

def load_and_print_metrics(file_path, model_name):
    """Load metrics from .npy file and print results"""
    if not Path(file_path).exists():
        print(f"❌ {model_name}: File not found - {file_path}")
        return None

    try:
        metrics = np.load(file_path, allow_pickle=True)
        print(f"\n{'='*60}")
        print(f"{model_name}: {file_path}")
        print(f"{'='*60}")

        if isinstance(metrics, np.ndarray) and len(metrics) > 0:
            # Get last epoch metrics (best performance usually)
            last_metrics = metrics[-1] if hasattr(metrics[-1], 'item') else metrics[-1]

            if isinstance(last_metrics, dict):
                print(f"Epoch: {last_metrics.get('epoch', 'N/A')}")
                print(f"Loss: {last_metrics.get('loss', 0):.4f}")

                # Extract NDCG and Recall
                for k in [10, 20]:
                    ndcg_key = f'ndcg@{k}'
                    recall_key = f'recall@{k}'
                    if ndcg_key in last_metrics:
                        print(f"NDCG@{k}: {last_metrics[ndcg_key]:.4f}")
                    if recall_key in last_metrics:
                        print(f"Recall@{k}: {last_metrics[recall_key]:.4f}")

                # Show all epochs for progression
                print(f"\nAll epochs:")
                for i, m in enumerate(metrics):
                    if isinstance(m, dict):
                        ep = m.get('epoch', i+1)
                        ndcg10 = m.get('ndcg@10', 0)
                        recall10 = m.get('recall@10', 0)
                        print(f"  Epoch {ep}: NDCG@10={ndcg10:.4f}, Recall@10={recall10:.4f}")

                return last_metrics
            else:
                print(f"Unexpected format: {type(last_metrics)}")
                print(metrics)
        else:
            print(f"Empty or invalid metrics file")

        return None
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None

# Compare models
print("\n" + "="*60)
print("MODEL COMPARISON: contrastive_new vs SimGCL")
print("="*60)

# Load contrastive_new metrics
contrastive_files = [
    ("contrastive_learning/outputs/min2_win10/model_metrics.npy", "Contrastive_new (min2_win10)"),
    ("contrastive_learning new/u00_metrics.npy", "Contrastive_new (u00)"),
    ("contrastive_learning new/u05_metrics.npy", "Contrastive_new (u05)"),
    ("contrastive_learning new/u10_metrics.npy", "Contrastive_new (u10)"),
]

# Load simgcl metrics
simgcl_files = [
    ("simgcl/outputs/min2_win10/model_metrics.npy", "SimGCL (min2_win10)"),
]

print("\n" + "🔵 CONTRASTIVE_NEW RESULTS " + "="*40)
contrastive_results = {}
for file_path, name in contrastive_files:
    result = load_and_print_metrics(file_path, name)
    if result:
        contrastive_results[name] = result

print("\n" + "🔴 SIMGCL RESULTS " + "="*40)
simgcl_results = {}
for file_path, name in simgcl_files:
    result = load_and_print_metrics(file_path, name)
    if result:
        simgcl_results[name] = result

# Summary comparison
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)

if contrastive_results and simgcl_results:
    print("\n📊 Best Results:")

    # Find best contrastive
    best_contrastive = max(contrastive_results.items(),
                          key=lambda x: x[1].get('ndcg@10', 0))
    best_simgcl = max(simgcl_results.items(),
                     key=lambda x: x[1].get('ndcg@10', 0))

    print(f"\n🔵 Best Contrastive_new: {best_contrastive[0]}")
    print(f"   NDCG@10: {best_contrastive[1].get('ndcg@10', 0):.4f}")
    print(f"   Recall@10: {best_contrastive[1].get('recall@10', 0):.4f}")

    print(f"\n🔴 Best SimGCL: {best_simgcl[0]}")
    print(f"   NDCG@10: {best_simgcl[1].get('ndcg@10', 0):.4f}")
    print(f"   Recall@10: {best_simgcl[1].get('recall@10', 0):.4f}")

    # Calculate improvement
    contrastive_ndcg = best_contrastive[1].get('ndcg@10', 0)
    simgcl_ndcg = best_simgcl[1].get('ndcg@10', 0)

    if simgcl_ndcg > 0:
        improvement = ((contrastive_ndcg - simgcl_ndcg) / simgcl_ndcg) * 100
        print(f"\n📈 Improvement: {improvement:+.2f}%")

        if improvement > 0:
            print(f"   ✅ Contrastive_new is better by {improvement:.2f}%")
        else:
            print(f"   ⚠️ SimGCL is better by {-improvement:.2f}%")
else:
    print("⚠️ Not enough data for comparison")
