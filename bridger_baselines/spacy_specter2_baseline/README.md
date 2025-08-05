# spaCy + SPECTER2 Bridger Baseline

Implementation of the Bridger recommendation algorithm from CHI 2022 paper "Bursting Scientific Filter Bubbles: Boosting Innovation via Novel Author Discovery". 

**Core Concept**: Recommend novel collaborators by finding scholars with **similar tasks** but **different methods**.

## Architecture Comparison

| Component | Original Paper | Our Implementation |
|-----------|----------------|-------------------|
| **Term Extraction** | DyGIE++ | **spaCy (configurable)** |
| **Embeddings** | CS-RoBERTa | **SPECTER2** |
| **Purpose** | Research prototype | Production baseline |

## System Components

### Core Scripts
- `scripts/setup_dygie_specter2.py` - Environment setup and model download
- `scripts/embedding_generator.py` - DyGIE++ term extraction + SPECTER2 embedding generation
- `scripts/bridger_baselines_improved.py` - Main evaluation script with persona support
- `persona_bridger.py` - Persona clustering for multi-domain researchers
- `recommend.py` - CLI recommendation interface

### Algorithm Pipeline

1. **Term Extraction** (DyGIE++)
   - Extract Task terms: `['Task', 'OtherScientificTerm']`
   - Extract Method terms: `['Method', 'Material', 'Metric']`

2. **Embedding Generation** (SPECTER2)
   - Generate 768-dim semantic vectors
   - Apply author position weighting (first/last: 1.0, middle: 0.75)
   - Apply paper importance weighting (MinMaxScaler on citation counts)

3. **Recommendation Methods**
   - **ST**: Similar Tasks baseline
   - **sTdM**: Similar Tasks + distant Methods baseline

4. **Persona Mode** (Optional)
   - Ward hierarchical clustering (distance threshold: 88.0)
   - Group papers by research domains (A, B, C...)
   - Enable fine-grained matching for multi-domain researchers

## Quick Start

1. **Setup Environment**:
   ```bash
   cd scripts/
   python setup_dygie_specter2.py
   ```

2. **Generate Embeddings**:
   ```bash
   python embedding_generator.py --evaluation-data /path/to/data.csv --force-regenerate
   ```

3. **Run Evaluation**:
   ```bash
   python bridger_baselines_improved.py --evaluation-data /path/to/data.csv --enable-personas
   ```

4. **Get Recommendations**:
   ```bash
   python ../recommend.py --author-id <author_id> --method ST --top-k 10
   ```

## Key Features

- **Advanced Weighting**: Author position × paper importance weighting
- **Persona Clustering**: Multi-domain researcher support
- **Precomputed Embeddings**: Fast evaluation with caching
- **CLI Interface**: Direct recommendation queries
- **Batch Processing**: Multiple dataset evaluation

## Documentation

See `docs/README_IMPROVED.md` for comprehensive documentation, parameters, troubleshooting, and advanced usage.
