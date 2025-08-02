#!/bin/bash
# Example script to run DyGIE++ + SPECTER2 baseline

echo "=== DyGIE++ + SPECTER2 Bridger Baseline Example ==="

# Set paths (modify these for your data)
EVALUATION_DATA="/data/jx4237data/Graph-CoT/Pipeline/step1_process/strict_0.88_remove_case1_year2-5/paper_levels_0.88_year2-5.csv"
EMBEDDING_DIR="../data/embeddings"

cd scripts/

echo "Step 1: Setup (one-time)"
echo "python setup_dygie_specter2.py"
echo ""

echo "Step 2: Generate embeddings (one-time, may take hours)"
echo "python embedding_generator.py \\"
echo "  --evaluation-data $EVALUATION_DATA \\"
echo "  --storage-dir $EMBEDDING_DIR \\"
echo "  --force-regenerate"
echo ""

echo "Step 3: Run evaluation (fast, seconds)"
echo "python bridger_baselines_improved.py \\"
echo "  --evaluation-data $EVALUATION_DATA \\"
echo "  --embedding-dir $EMBEDDING_DIR"
echo ""

echo "Step 4: Compare with original baseline"
echo "python bridger_baselines_improved.py \\"
echo "  --evaluation-data $EVALUATION_DATA \\"
echo "  --embedding-dir $EMBEDDING_DIR \\"
echo "  --compare-original"
echo ""

echo "To actually run these commands, uncomment them in this script!"

# Uncomment to actually run:
# python setup_dygie_specter2.py
# python embedding_generator.py --evaluation-data $EVALUATION_DATA --storage-dir $EMBEDDING_DIR --force-regenerate  
# python bridger_baselines_improved.py --evaluation-data $EVALUATION_DATA --embedding-dir $EMBEDDING_DIR --compare-original