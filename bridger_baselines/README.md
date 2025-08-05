# Bridger Baselines Implementation

Implementation of ST and sTdM baselines from "Bursting Scientific Filter Bubbles" (CHI 2022) paper for research evaluation.

## Quick Start

1. **Generate Embeddings** (using correct author nodes):
```bash
cd dygie_specter2_baseline
./run_complete_embedding_generation.sh
```

2. **Run Evaluation**:
```python
from bridger_baselines import run_bridger_evaluation

results = run_bridger_evaluation(
    evaluation_data_path="/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
)

print(f"ST Hit@10: {results['ST']['Hit@10']:.4f}")
print(f"sTdM Hit@10: {results['sTdM']['Hit@10']:.4f}")
```

## Core Files

### Root Directory
- `bridger_baselines.py` - Main baseline implementation and evaluation functions
- `CLAUDE.md` - Project documentation and implementation notes

### dygie_specter2_baseline/
- `generate_correct_embeddings.py` - Content-based embedding generation using proper data structure
- `run_complete_embedding_generation.sh` - Complete embedding generation workflow
- `full_dataset_test.py` - Full evaluation script
- `recommend.py` - CLI interface for getting recommendations

## Algorithm Overview

**ST (Similar Tasks)**: Recommends collaborators based on task embedding similarity.

**sTdM (Similar Tasks, distant Methods)**: 
1. Filter candidates by task similarity (top-k)
2. Re-rank by method dissimilarity
3. Return authors with similar tasks but different methods

## Data Requirements

- **Author Nodes**: `/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/updated_author_nodes_with_papers.json`
- **Paper Nodes**: `/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/paper_nodes_2024dec.json`
- **Evaluation Data**: CSV file with author teams and ground truth collaborators

## Implementation Features

- **Content-based Embeddings**: Uses actual paper titles and abstracts
- **Multi-processing**: Parallel embedding generation for all 44,673 authors
- **High Coverage**: 100% paper matching rate using correct data structure
- **Vectorized Operations**: Optimized similarity computations
- **Standard Metrics**: Hit@k and MRR evaluation

## Usage Examples

### Generate Recommendations
```bash
python dygie_specter2_baseline/recommend.py --author-id 1234567 --method ST --top-k 10
```

### Custom Evaluation
```python
from bridger_baselines import BridgerBaselines
import pickle

# Load pre-computed embeddings
with open('dygie_specter2_baseline/correct_embeddings/task_embeddings.pkl', 'rb') as f:
    task_emb = pickle.load(f)
with open('dygie_specter2_baseline/correct_embeddings/method_embeddings.pkl', 'rb') as f:
    method_emb = pickle.load(f)

# Initialize baselines
baselines = BridgerBaselines(task_emb, method_emb)

# Get recommendations
recs = baselines.st_baseline(focal_author="1234567", top_k=10)
print(f"ST recommendations: {recs}")

recs = baselines.stdm_baseline(focal_author="1234567", top_k=10)  
print(f"sTdM recommendations: {recs}")
```

## Performance Notes

- Embedding generation: ~30-60 minutes for all authors (32 processes)
- Memory usage: ~6GB for full dataset
- Evaluation: ~1-2 minutes for 986 test pairs
- Coverage: 75%+ on evaluation dataset