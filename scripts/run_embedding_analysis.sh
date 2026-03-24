#!/bin/bash

# Embedding analysis script for all models
# Usage: bash run_embedding_analysis.sh

DATASET="min5_win10"
OUTPUT_DIR="comparison_results"
mkdir -p $OUTPUT_DIR

echo "==================================="
echo "Embedding Analysis for All Models"
echo "==================================="

# Define models and their paths
declare -A MODELS=(
    ["Contrastive"]="contrastive_learning_loo/outputs/${DATASET}/model_loo_track_embeddings.npy"
    ["Contrastive-U1.0"]="contrastive_learning_loo/outputs/${DATASET}_u1.0/model_loo_track_embeddings.npy"
    ["LightGCN"]="lightgcn_loo/outputs/${DATASET}/model_loo_track_embeddings.npy"
    ["SimGCL"]="simgcl_loo/outputs/${DATASET}/model_loo_track_embeddings.npy"
)

echo ""
echo "Models to analyze:"
for model in "${!MODELS[@]}"; do
    echo "  - $model: ${MODELS[$model]}"
done

# ===================================
# 1. Popularity-based Similarity Analysis
# ===================================
echo ""
echo "==================================="
echo "1. Popularity-based Similarity Analysis"
echo "==================================="

for model in "${!MODELS[@]}"; do
    path="${MODELS[$model]}"

    # Check if file exists
    if [ ! -f "$path" ]; then
        echo "⚠️  Skipping $model: file not found at $path"
        continue
    fi

    echo ""
    echo "Analyzing $model..."
    python analyze_popularity_similarity.py \
        --model-path "$path" \
        --model-name "$model" \
        --dataset $DATASET \
        --popular-threshold 100 \
        --unpopular-threshold 3 \
        --sample-pairs 10000 \
        --output "${OUTPUT_DIR}/popularity_${model// /-}.png"
done

# ===================================
# 2. Embedding Visualization (PCA)
# ===================================
echo ""
echo "==================================="
echo "2. Pairwise Embedding Visualization"
echo "==================================="

# Compare Contrastive vs Contrastive-U1.0
if [ -f "${MODELS[Contrastive]}" ] && [ -f "${MODELS[Contrastive-U1.0]}" ]; then
    echo ""
    echo "Comparing: Contrastive vs Contrastive-U1.0"
    python visualize_embeddings.py \
        --model1-path "${MODELS[Contrastive]}" \
        --model1-name "Contrastive" \
        --model2-path "${MODELS[Contrastive-U1.0]}" \
        --model2-name "Contrastive-U1.0" \
        --method pca \
        --n-samples 5000 \
        --output "${OUTPUT_DIR}/viz_contrastive_vs_u1.0.png"
fi

# Compare Contrastive vs LightGCN
if [ -f "${MODELS[Contrastive]}" ] && [ -f "${MODELS[LightGCN]}" ]; then
    echo ""
    echo "Comparing: Contrastive vs LightGCN"
    python visualize_embeddings.py \
        --model1-path "${MODELS[Contrastive]}" \
        --model1-name "Contrastive" \
        --model2-path "${MODELS[LightGCN]}" \
        --model2-name "LightGCN" \
        --method pca \
        --n-samples 5000 \
        --output "${OUTPUT_DIR}/viz_contrastive_vs_lightgcn.png"
fi

# Compare Contrastive vs SimGCL
if [ -f "${MODELS[Contrastive]}" ] && [ -f "${MODELS[SimGCL]}" ]; then
    echo ""
    echo "Comparing: Contrastive vs SimGCL"
    python visualize_embeddings.py \
        --model1-path "${MODELS[Contrastive]}" \
        --model1-name "Contrastive" \
        --model2-path "${MODELS[SimGCL]}" \
        --model2-name "SimGCL" \
        --method pca \
        --n-samples 5000 \
        --output "${OUTPUT_DIR}/viz_contrastive_vs_simgcl.png"
fi

# Compare LightGCN vs SimGCL
if [ -f "${MODELS[LightGCN]}" ] && [ -f "${MODELS[SimGCL]}" ]; then
    echo ""
    echo "Comparing: LightGCN vs SimGCL"
    python visualize_embeddings.py \
        --model1-path "${MODELS[LightGCN]}" \
        --model1-name "LightGCN" \
        --model2-path "${MODELS[SimGCL]}" \
        --model2-name "SimGCL" \
        --method pca \
        --n-samples 5000 \
        --output "${OUTPUT_DIR}/viz_lightgcn_vs_simgcl.png"
fi

echo ""
echo "==================================="
echo "Analysis Complete!"
echo "==================================="
echo "Results saved to: $OUTPUT_DIR/"
ls -lh $OUTPUT_DIR/
