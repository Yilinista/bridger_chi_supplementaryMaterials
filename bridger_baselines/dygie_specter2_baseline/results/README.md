# Results Directory

This directory stores evaluation results and logs.

## Structure (created during evaluation)

```
results/
├── evaluation_results.json         # Main evaluation results
├── comparison_results.json         # Comparison with original baseline
├── logs/                           # Detailed logs
│   ├── embedding_generation.log    # Embedding generation logs
│   ├── evaluation.log              # Evaluation logs
│   └── comparison.log              # Comparison logs
└── plots/                          # Generated plots (if any)
    ├── performance_comparison.png
    └── embedding_statistics.png
```

## Expected Results Format

```json
{
  "ST": {
    "Hit@10": 0.35,
    "MRR": 0.18,
    "Queries": 1250
  },
  "sTdM": {
    "Hit@10": 0.25,
    "MRR": 0.12,
    "Queries": 1250
  },
  "metadata": {
    "evaluation_date": "2024-01-15T10:30:00",
    "evaluation_file": "paper_levels_0.88_year2-5.csv",
    "embedding_model": "allenai/specter2_base",
    "term_extraction": "DyGIE++ SciERC"
  }
}
```

## Performance Tracking

Results are automatically saved with timestamps to track improvements over time.

Use the comparison mode to benchmark against the original random-vector baseline:

```bash
python scripts/bridger_baselines_improved.py \
  --evaluation-data data.csv \
  --compare-original
```