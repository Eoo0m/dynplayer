"""Check train/test split statistics and sparsity."""

import numpy as np
from pathlib import Path

split_file = "/Users/eomjoonseo/dynplayer/lightgcn_sparse/train_test_split.npz"

print(f"=== Analyzing {split_file} ===\n")

# Load data
data = np.load(split_file, allow_pickle=True)

train_dict = data["train_dict"].item()
test_dict = data["test_dict"].item()
num_users = int(data["num_users"])
num_items = int(data["num_items"])

print(f"📊 Matrix Dimensions:")
print(f"  Rows (Users/Playlists): {num_users:,}")
print(f"  Cols (Items/Tracks): {num_items:,}")
print(f"  Total possible entries: {num_users * num_items:,}")

# Count actual interactions
train_interactions = sum(len(tracks) for tracks in train_dict.values())
test_interactions = sum(len(tracks) for tracks in test_dict.values())
total_interactions = train_interactions + test_interactions

print(f"\n📈 Interactions:")
print(f"  Train interactions: {train_interactions:,}")
print(f"  Test interactions: {test_interactions:,}")
print(f"  Total interactions: {total_interactions:,}")

# Calculate sparsity
total_possible = num_users * num_items
train_sparsity = 1 - (train_interactions / total_possible)
test_sparsity = 1 - (test_interactions / total_possible)
overall_sparsity = 1 - (total_interactions / total_possible)

print(f"\n💫 Sparsity:")
print(f"  Train sparsity: {train_sparsity:.6f} ({train_sparsity*100:.4f}%)")
print(f"  Test sparsity: {test_sparsity:.6f} ({test_sparsity*100:.4f}%)")
print(f"  Overall sparsity: {overall_sparsity:.6f} ({overall_sparsity*100:.4f}%)")

# Density (inverse of sparsity)
train_density = train_interactions / total_possible
test_density = test_interactions / total_possible
overall_density = total_interactions / total_possible

print(f"\n🎯 Density (filled ratio):")
print(f"  Train density: {train_density:.6f} ({train_density*100:.4f}%)")
print(f"  Test density: {test_density:.6f} ({test_density*100:.4f}%)")
print(f"  Overall density: {overall_density:.6f} ({overall_density*100:.4f}%)")

# Additional stats
avg_train_per_user = train_interactions / num_users
avg_test_per_user = test_interactions / num_users
avg_total_per_user = total_interactions / num_users

print(f"\n📊 Per-user statistics:")
print(f"  Avg train tracks per playlist: {avg_train_per_user:.2f}")
print(f"  Avg test tracks per playlist: {avg_test_per_user:.2f}")
print(f"  Avg total tracks per playlist: {avg_total_per_user:.2f}")

# Distribution stats
train_lens = [len(v) for v in train_dict.values()]
test_lens = [len(v) for v in test_dict.values()]

print(f"\n📈 Distribution:")
print(f"  Train - min: {min(train_lens)}, max: {max(train_lens)}, median: {np.median(train_lens):.1f}")
print(f"  Test - min: {min(test_lens)}, max: {max(test_lens)}, median: {np.median(test_lens):.1f}")
