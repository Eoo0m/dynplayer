import numpy as np
import matplotlib.pyplot as plt

paths = {
    "u00": "/Users/eomjoonseo/projects/dynplayer-1/contrastive_learning/u00_metrics.npy",
    "u05": "/Users/eomjoonseo/projects/dynplayer-1/contrastive_learning/u05_metrics.npy",
    "u10": "/Users/eomjoonseo/projects/dynplayer-1/contrastive_learning/u10_metrics.npy",
}

legend_name = {
    "u00": "Uniformity OFF (w=0)",
    "u05": "Uniformity w=0.5",
    "u10": "Uniformity w=1.0",
}

metrics = {}
for name, path in paths.items():
    arr = np.load(path, allow_pickle=True)
    metrics[name] = list(arr)  # 리스트로 변환


def extract(metric_list, key):
    return [d[key] for d in metric_list]


plt.figure(figsize=(12, 5))

# --------------------------
# 1) NDCG@20
# --------------------------
plt.subplot(1, 2, 1)
for name in metrics:
    epochs = extract(metrics[name], "epoch")
    ndcg20 = extract(metrics[name], "ndcg@20")
    plt.plot(epochs, ndcg20, label=legend_name[name], linewidth=2)

plt.title("NDCG@20 vs Uniformity Weight")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.grid()
plt.legend()

# --------------------------
# 2) Recall@20
# --------------------------
plt.subplot(1, 2, 2)
for name in metrics:
    epochs = extract(metrics[name], "epoch")
    recall20 = extract(metrics[name], "recall@20")
    plt.plot(epochs, recall20, label=legend_name[name], linewidth=2)

plt.title("Recall@20 vs Uniformity Weight")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
