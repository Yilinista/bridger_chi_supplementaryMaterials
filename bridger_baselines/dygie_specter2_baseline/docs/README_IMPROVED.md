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
  --evaluation-data /path/to/evaluation_data.csv
```

**Parameters:**
- `--evaluation-data`: Path to evaluation CSV
- `--embedding-dir`: Directory containing embeddings (default: `./bridger_embeddings`)
- `--compare-original`: Compare with original random-vector baseline
- `--batch-mode file1.csv file2.csv`: Evaluate on multiple datasets
- `--force-regenerate`: Force regeneration of embeddings
- `--stats-only`: Show embedding statistics only


## Architecture Comparison

| Component | Original Paper | Our Implementation |
|-----------|----------------|-------------------|
| **Term Extraction** | DyGIE++ | DyGIE++ |
| **Term Embedding** | CS-RoBERTa | SPECTER2 |
| **Algorithm Logic** | ST/sTdM cosine similarity | ST/sTdM cosine similarity |

## File Structure

```
├── scripts/
│   ├── embedding_generator.py       # Generate embeddings
│   ├── bridger_baselines_improved.py # Run evaluations
│   └── setup_dygie_specter2.py      # Setup script
├── data/
│   └── embeddings/                  # Generated embeddings
├── models/
│   └── dygiepp/                     # DyGIE++ repository
└── results/                         # Evaluation results
```

## Embedding Storage Format

```python
{
    "metadata": {
        "dygie_model": "scierc",
        "embedding_model": "allenai/specter2_base",
        "creation_time": "2024-01-15T10:30:00",
        "total_authors": 15000,
        "embedding_dim": 768
    },
    "task_embeddings": {
        "author_123": np.array([...]),  # 768-dim vector
    },
    "method_embeddings": {
        "author_123": np.array([...]),  # 768-dim vector
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


## References

- **Original Bridger Paper**: "Bursting Scientific Filter Bubbles" (CHI 2022)
- **DyGIE++**: https://github.com/dwadden/dygiepp
- **SPECTER2**: https://huggingface.co/allenai/specter2_base
- **SciSpaCy**: https://allenai.github.io/scispacy/