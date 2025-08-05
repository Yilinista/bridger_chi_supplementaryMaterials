# DyGIE++ + SPECTER2 Bridger Baseline Documentation

This implementation follows the original Bridger paper methodology using:
- **DyGIE++** for scientific term extraction
- **SPECTER2** for semantic embeddings (replaces CS-RoBERTa)
- **Precomputed embeddings** for fast evaluation

## Setup

### Automated Setup
```bash
python setup_dygie_specter2.py
```

### Manual Setup
```bash
# Install dependencies
pip install torch sentence-transformers transformers numpy pandas scikit-learn spacy scispacy

# Install spaCy models
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz

# Clone and setup DyGIE++
git clone https://github.com/dwadden/dygiepp.git
cd dygiepp
pip install -r requirements.txt

# Download SciERC model
mkdir -p pretrained_models
cd pretrained_models
wget https://ai2-s2-dygiepp.s3.amazonaws.com/models/scierc.tar.gz
tar -xzf scierc.tar.gz
```

## Usage

### Step 1: Generate Embeddings 

```bash
python embedding_generator.py \
  --evaluation-data /path/to/evaluation_data.csv \
  --force-regenerate
```

**Parameters:**
- `--evaluation-data`: Path to BetterTeaming evaluation CSV
- `--paper-nodes`: Path to paper nodes JSON (default: Graph-CoT path)
- `--author-kg`: Path to author knowledge graph JSON (default: Graph-CoT path)
- `--storage-dir`: Directory to store embeddings (default: `./bridger_embeddings`)
- `--force-regenerate`: Force regeneration even if embeddings exist
- `--stats-only`: Show embedding statistics only

### Step 2: Run Evaluation 

```bash
python bridger_baselines_improved.py \
  --evaluation-data /path/to/evaluation_data.csv \
  --enable-personas \
  --min-papers 4
```

**Parameters:**
- `--evaluation-data`: Path to evaluation CSV
- `--embedding-dir`: Directory containing embeddings (default: `./bridger_embeddings`)
- `--enable-personas`: Enable persona clustering mode
- `--min-papers`: Minimum papers per persona (default: 4)
- `--compare-original`: Compare with original random-vector baseline
- `--batch-mode file1.csv file2.csv`: Evaluate on multiple datasets
- `--force-regenerate`: Force regeneration of embeddings
- `--stats-only`: Show embedding statistics only

### Step 3: Get Direct Recommendations

```bash
python ../recommend.py \
  --author-id <author_id> \
  --method ST \
  --top-k 10
```

**Parameters:**
- `--author-id`: Target author ID for recommendations
- `--method`: Recommendation method (ST or sTdM)
- `--top-k`: Number of recommendations to return (default: 10)
- `--embedding-dir`: Directory containing embeddings


## Architecture Comparison

| Component | Original Paper | Our Implementation |
|-----------|----------------|-------------------|
| **Term Extraction** | DyGIE++ | DyGIE++ |
| **Term Embedding** | CS-RoBERTa | SPECTER2 |
| **Algorithm Logic** | ST/sTdM cosine similarity | ST/sTdM cosine similarity |

## File Structure

```
dygie_specter2_baseline/
├── scripts/
│   ├── embedding_generator.py       # DyGIE++ + SPECTER2 pipeline
│   ├── bridger_baselines_improved.py # Enhanced evaluation with persona support
│   └── setup_dygie_specter2.py      # Environment setup script
├── persona_bridger.py                # Persona clustering implementation
├── recommend.py                      # CLI recommendation interface
├── data/
│   └── embeddings/                  # Generated embeddings storage
├── models/
│   └── dygiepp/                     # DyGIE++ repository
├── results/                         # Evaluation results
├── global_embeddings/               # Precomputed global embeddings
│   ├── metadata.json               # Embedding metadata
│   ├── task_embeddings.pkl         # Task embeddings
│   └── method_embeddings.pkl       # Method embeddings
└── docs/
    └── README_IMPROVED.md          # This detailed documentation
```

## Algorithm Details

### Weighting System
Following the original Bridger paper methodology:

**Author Position Weighting:**
- First or last author: 1.0
- Middle author: 0.75

**Paper Importance Weighting:**
- Uses MinMaxScaler with feature_range=(0.5, 1.0)
- Based on citation counts from paper metadata
- Combined Weight = position_weight × importance_weight

### Persona Clustering (Optional)
- Algorithm: Ward hierarchical clustering with Euclidean distance
- Distance Threshold: 88.0
- Minimum Papers per Persona: 4 (configurable)
- Paper Embeddings: SPECTER2 on "title. abstract"
- Persona Assignment: Letters A, B, C... by paper count

### Recommendation Methods

**ST (Similar Tasks):**
- Rank candidates by task embedding cosine similarity
- Returns top-k most similar task profiles

**sTdM (Similar Tasks, distant Methods):**
- Filter candidates by task similarity (top 50%)
- Rank filtered candidates by method embedding dissimilarity
- Implements the core Bridger "similar tasks, different methods" concept

## Data Dependencies

The system requires access to these data files:
- Paper nodes: `/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json`
- Author knowledge graph: `/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json`

## Embedding Storage Format

```python
{
    "metadata": {
        "dygie_model": "scierc",
        "embedding_model": "allenai/specter2_base",
        "creation_time": "2024-01-15T10:30:00",
        "total_authors": 15000,
        "embedding_dim": 768,
        "weighting_enabled": true,
        "persona_mode": false
    },
    "task_embeddings": {
        "author_123": np.array([...]),  # 768-dim vector
    },
    "method_embeddings": {
        "author_123": np.array([...]),  # 768-dim vector
    },
    "persona_embeddings": {  # Only if persona mode enabled
        "author_123_A": np.array([...]),
        "author_123_B": np.array([...])
    }
}
```

## Troubleshooting

### DyGIE++ Model Not Found
```bash
cd dygiepp/pretrained_models
wget https://ai2-s2-dygiepp.s3.amazonaws.com/models/scierc.tar.gz
tar -xzf scierc.tar.gz
```

### Scientific spaCy Model Issues
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
```

### Memory Issues
Process datasets in smaller batches or increase system memory.

### GPU/CUDA Issues
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Performance Notes

### System Requirements
- Memory: 8GB+ RAM for medium datasets (10K authors)
- Storage: 1-5GB for embeddings depending on dataset size
- GPU: Optional, CPU-only mode supported

### Processing Times (Approximate)
- Setup: 5-10 minutes (model downloads)
- Embedding generation: 30-60 minutes for 10K authors
- Evaluation: 5-15 minutes depending on query size
- Persona clustering: Additional 10-20 minutes

### Optimization Tips
- Use `--stats-only` to check embedding status without regeneration
- Enable persona mode only if needed (adds computation overhead)
- Use precomputed embeddings in `global_embeddings/` for repeated evaluations

## CLI Usage Examples

### Basic Evaluation
```bash
python scripts/bridger_baselines_improved.py --evaluation-data evaluation_data.csv
```

### Full Pipeline with Personas
```bash
python scripts/bridger_baselines_improved.py \
  --evaluation-data evaluation_data.csv \
  --enable-personas \
  --min-papers 4 \
  --force-regenerate
```

### Batch Evaluation
```bash
python scripts/bridger_baselines_improved.py \
  --batch-mode dataset1.csv dataset2.csv dataset3.csv
```

### Direct Recommendations
```bash
python recommend.py --author-id 12345 --method sTdM --top-k 20
```

## Integration with Evaluation Pipeline

The system outputs standardized recommendation lists compatible with evaluation frameworks:

```python
# Example output format
{
    "query_author": "12345",
    "method": "sTdM",
    "recommendations": [
        {"author_id": "67890", "score": 0.85},
        {"author_id": "11111", "score": 0.82},
        # ... more recommendations
    ],
    "metadata": {
        "persona_mode": true,
        "total_candidates": 15000,
        "generation_time": "2024-01-15T10:30:00"
    }
}
```

## References

- **Original Bridger Paper**: "Bursting Scientific Filter Bubbles: Boosting Innovation via Novel Author Discovery" (CHI 2022)
- **DyGIE++**: https://github.com/dwadden/dygiepp
- **SPECTER2**: https://huggingface.co/allenai/specter2_base
- **SciSpaCy**: https://allenai.github.io/scispacy/