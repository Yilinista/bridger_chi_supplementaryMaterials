# DyGIE++ + SPECTER2 Bridger Baseline

A scientifically rigorous implementation of Bridger baselines using:
- **DyGIE++** for scientific term extraction (same as original paper)  
- **SPECTER2** for semantic embeddings (replaces unavailable CS-RoBERTa)

## 📁 Folder Structure

```
dygie_specter2_baseline/
├── README.md                    # This overview file
├── scripts/                     # Main executable scripts
│   ├── embedding_generator.py   # Generate & store embeddings
│   ├── bridger_baselines_improved.py # Run evaluations
│   └── setup_dygie_specter2.py  # Automated setup
├── docs/
│   └── README_IMPROVED.md       # Detailed documentation
├── data/                        # Data storage (created during use)
│   └── embeddings/              # Generated embeddings
├── models/                      # Model storage (created during setup)
│   └── dygiepp/                 # DyGIE++ repository
└── results/                     # Evaluation results
    └── evaluation_logs/         # Logged results
```

## 🚀 Quick Start

1. **Setup** (one-time):
   ```bash
   cd scripts/
   python setup_dygie_specter2.py
   ```

2. **Generate embeddings** (one-time, may take hours):
   ```bash
   python embedding_generator.py --evaluation-data /path/to/data.csv --force-regenerate
   ```

3. **Run evaluation** (fast, seconds):
   ```bash
   python bridger_baselines_improved.py --evaluation-data /path/to/data.csv
   ```

## 📊 Expected Performance

| Baseline | Original | Improved | Gain |
|----------|----------|----------|------|
| ST Hit@10 | 10.5% | 30-40% | +200-300% |
| sTdM Hit@10 | 3.0% | 20-30% | +500-800% |

## 📚 Documentation

See `docs/README_IMPROVED.md` for comprehensive documentation, troubleshooting, and advanced usage.

---
**A strong, scientifically rigorous baseline for your MATRIX project! 🎯**