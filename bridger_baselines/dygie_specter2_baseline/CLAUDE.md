# Bridger Baseline Implementation Project

## Project Overview

This project implements a baseline version of the Bridger recommendation algorithm from the CHI 2022 paper "Bursting Scientific Filter Bubbles: Boosting Innovation via Novel Author Discovery". The core idea is to recommend novel collaborators by finding scholars with **similar tasks** but **different methods**.

### Original vs Our Implementation

| Component | Original Paper | Our Implementation |
|-----------|----------------|-------------------|
| Term Extraction | DyGIE++ | spaCy (simplified) |
| Embeddings | cs-roberta | SPECTER2 |
| Purpose | Research prototype | Production baseline |

## Project Background

**Task**: Adapt and reproduce the core recommendation algorithm from the published Bridger paper as a baseline for our own project.

**Core Algorithm**: The Bridger algorithm recommends novel collaborators by finding scholars whose work has **similar "Tasks"** but **different "Methods"**.

## Implementation Pipeline

### 1. Term Extraction
- **Tool**: spaCy with scientific term classification
- **Input**: Paper titles and abstracts from our dataset
- **Output**: Extracted Task and Method keywords
- **Categories**:
  - Task terms: research problems, domains, applications
  - Method terms: algorithms, techniques, tools, frameworks

### 2. Embedding Generation
- **Tool**: SPECTER2 model (`allenai/specter2_base`)
- **Input**: Extracted terms from step 1
- **Output**: 768-dimensional semantic embeddings
- **Process**: Weighted averaging with author position and paper importance weights

### 3. Algorithm Implementation
- **Author Embeddings**: Weighted average of all term embeddings per author
- **Two Recommendation Methods**:
  - **ST (Similar Tasks)**: Rank by task similarity only
  - **sTdM (Similar Tasks, distant Methods)**: Filter by task similarity, then rank by method dissimilarity

## Key Features

### Persona Mode Support
- **Clustering**: Uses Ward hierarchical clustering with distance threshold 88.0
- **Paper Grouping**: Groups each author's papers into research personas (A, B, C...)
- **Fine-grained Matching**: Enables better recommendations for multi-domain researchers

### Advanced Weighting System
Following the original paper methodology:

#### Author Position Weighting
- First or last author: **1.0**
- Middle author: **0.75**

#### Paper Importance Weighting
- Uses MinMaxScaler with feature_range=(0.5, 1.0)
- Based on citation counts from paper metadata
- **Combined Weight** = position_weight × importance_weight

## File Structure

```
dygie_specter2_baseline/
├── scripts/
│   ├── bridger_baselines_improved.py    # Enhanced Bridger with persona support
│   ├── embedding_generator.py           # DyGIE++ + SPECTER2 pipeline
│   └── setup_dygie_specter2.py         # Environment setup
├── persona_bridger.py                   # Persona clustering implementation
├── recommend.py                         # CLI recommendation interface
├── test_improved_weighting.py          # Weighting system tests
└── docs/
    └── README_IMPROVED.md              # Detailed documentation
```

## Recent Improvements (Latest Session)

### 1. Proper MinMaxScaler Implementation
- **Before**: Hardcoded linear interpolation `0.5 + (cited_count/100) * 0.5`
- **After**: True sklearn MinMaxScaler with global citation distribution fitting

### 2. Matrix-based Term Weighting
- **Before**: Uniform weights for all terms
- **After**: Term weights based on paper-level author position and citation importance

### 3. Enhanced Persona Mode
- **Before**: No weighting support in persona mode
- **After**: Full weighting support with focal_author_id tracking

### 4. Two-pass Processing
- **Pass 1**: Collect all citation counts to fit MinMaxScaler
- **Pass 2**: Generate embeddings with proper weighting

## Usage

### Generate Embeddings
```bash
python scripts/bridger_baselines_improved.py --enable-personas --min-papers 4
```

### Get Recommendations
```bash
python recommend.py --author-id <author_id> --method ST --top-k 10
```

### Available Methods
- `ST`: Similar Tasks baseline
- `sTdM`: Similar Tasks, distant Methods baseline

## Technical Notes

### Weighting Formula (Following Original Paper)
```python
combined_weight = author_position_weight * paper_importance_weight

where:
- author_position_weight ∈ {0.75, 1.0}
- paper_importance_weight ∈ [0.5, 1.0] (MinMaxScaler)
```

### Clustering Parameters
- **Algorithm**: Ward linkage with Euclidean distance
- **Distance Threshold**: 88.0
- **Minimum Papers per Persona**: 4
- **Paper Embeddings**: SPECTER2 on "title. abstract"

### Data Dependencies
- Paper nodes: `/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json`
- Author knowledge graph: `/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json`

## Goal Achievement

**Target**: A script that accepts author ID and outputs ranked recommendation list for evaluation on our dataset.

**Status**: ✅ **Achieved**
- CLI interface available via `recommend.py`
- Support for both standard and persona modes
- Proper weighting following original paper methodology
- Ready for evaluation pipeline integration

## Next Steps

1. Integration with evaluation pipeline
2. Performance optimization for large-scale datasets
3. Hyperparameter tuning for clustering thresholds
4. Comparison with other baseline methods