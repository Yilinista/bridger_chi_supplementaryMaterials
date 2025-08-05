#!/bin/bash
# Final embedding generation script

echo "Starting Bridger embedding generation with spaCy + SPECTER2..."
echo "Started at: $(date)"

# Activate ScienceBeam environment
source /home/jx4237/miniconda3/bin/activate ScienceBeam

# Run the embedding generation
python scripts/multiprocess_embedding_generator_final.py \
    --evaluation-data dummy_eval.csv \
    --paper-nodes /data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json \
    --author-kg /data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json \
    --storage-dir ./global_embeddings_final \
    --force-regenerate

echo "Completed at: $(date)"
echo "Check results in: ./global_embeddings_final/"