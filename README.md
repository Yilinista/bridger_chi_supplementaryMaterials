# Bridger Algorithm Implementation - Supplementary Materials

This repository contains the baseline implementation of the Bridger recommendation algorithm from the CHI 2022 paper "Bursting Scientific Filter Bubbles: Boosting Innovation via Novel Author Discovery".

## Project Structure

```
bridger_chi_supplementaryMaterials/
├── README.md                           # This file
├── bridger_baselines/
│   └── spacy_specter2_baseline/       # Main implementation
│       ├── README.md                   # Implementation details
│       ├── CLAUDE.md                   # Project memory
│       ├── STATUS.md                   # Current status
│       ├── requirements.txt            # Python dependencies
│       ├── recommend.py                # CLI recommendation interface
│       ├── config/                     # Configuration files
│       │   ├── stopwords.json         # Stopwords for term extraction
│       │   └── term_classification.json # Term classification rules
│       ├── scripts/                    # Core implementation
│       │   ├── multiprocess_persona_embedding_generator.py  # Main embedding generator
│       │   ├── bridger_baselines_improved.py              # Evaluation script
│       │   └── spacy_term_extractor.py                    # Term extraction
│       └── persona_embeddings/        # Generated embeddings output
└── experiment1Script.pdf              # Original paper materials
└── experiment2Script.pdf
└── tutorialSlides.pdf
```

## Core Algorithm

The Bridger algorithm recommends novel collaborators by finding scholars with **similar tasks** but **different methods**:

1. **Term Extraction**: Uses spaCy to extract scientific terms from paper titles/abstracts
2. **Classification**: Categorizes terms as "Tasks" (research problems) or "Methods" (techniques)  
3. **Embeddings**: Generates SPECTER2 semantic embeddings with author position and citation weighting
4. **Persona Clustering**: Groups authors' papers into research personas using hierarchical clustering
5. **Recommendation**: Ranks candidates by task similarity and method dissimilarity

## Quick Start

```bash
# Install dependencies
pip install -r bridger_baselines/spacy_specter2_baseline/requirements.txt

# Generate embeddings (takes several hours for full dataset)
cd bridger_baselines/spacy_specter2_baseline
python scripts/multiprocess_persona_embedding_generator.py

# Get recommendations
python recommend.py --author-id <author_id> --method sTdM --top-k 10
```

## Key Features

- **Multi-process embedding generation** with CUDA optimization
- **Persona-based clustering** for multi-domain researchers  
- **Advanced weighting** based on author position and paper citations
- **Command-line interface** for easy recommendations
- **Production-ready** baseline for research evaluation

## Data Dependencies

The implementation expects data files at:
- Paper nodes: `/data/.../paper_nodes_2024dec.json`  
- Author nodes: `/data/.../updated_author_nodes_with_papers.json`

## Implementation Details

See `bridger_baselines/spacy_specter2_baseline/README.md` for detailed technical documentation.

## Original Paper Materials

- `experiment1Script.pdf`: User study script for Experiment 1
- `experiment2Script.pdf`: User study script for Experiment 2  
- `tutorialSlides.pdf`: Tutorial slides for Experiment 2

## Original Paper

Portenoy, J., West, J. D., & Howe, B. (2022). Bursting Scientific Filter Bubbles: Boosting Innovation via Novel Author Discovery. CHI Conference on Human Factors in Computing Systems.
