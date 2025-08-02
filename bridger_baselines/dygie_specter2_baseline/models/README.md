# Models Directory

This directory contains downloaded models and repositories.

## Structure (created during setup)

```
models/
├── dygiepp/                        # DyGIE++ repository (cloned)
│   ├── predict.py                  # Main prediction script
│   ├── pretrained_models/          # Downloaded models
│   │   └── scierc/                 # SciERC model for term extraction
│   ├── requirements.txt            # DyGIE++ dependencies
│   └── ...                         # Other DyGIE++ files
└── cache/                          # Model cache (HuggingFace, etc.)
```

## Models Used

1. **DyGIE++ SciERC Model**
   - Purpose: Scientific term extraction
   - Source: https://ai2-s2-dygiepp.s3.amazonaws.com/models/scierc.tar.gz
   - Size: ~500 MB
   - Description: Pre-trained on scientific literature for NER and relation extraction

2. **SPECTER2**
   - Purpose: Semantic embeddings for scientific text
   - Source: `allenai/specter2_base` (HuggingFace)
   - Size: ~500 MB (downloaded automatically)
   - Description: Scientific document embeddings model

## Setup

Models are automatically downloaded by `setup_dygie_specter2.py`:

```bash
cd ../scripts/
python setup_dygie_specter2.py
```

## Manual Download (if needed)

```bash
# DyGIE++ repository
git clone https://github.com/dwadden/dygiepp.git

# SciERC model
cd dygiepp/pretrained_models
wget https://ai2-s2-dygiepp.s3.amazonaws.com/models/scierc.tar.gz
tar -xzf scierc.tar.gz
```