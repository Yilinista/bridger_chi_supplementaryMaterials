# Data Directory

This directory stores generated embeddings and intermediate data.

## Structure (created during use)

```
data/
├── embeddings/                      # Generated embeddings
│   ├── author_embeddings.pkl        # Main embedding file
│   ├── extracted_terms.json         # Terms extracted by DyGIE++
│   ├── metadata.json               # Embedding metadata
│   ├── task_embeddings.npz         # Task embeddings (numpy)
│   └── method_embeddings.npz       # Method embeddings (numpy)
├── cache/                          # Temporary files during processing
└── logs/                           # Processing logs
```

## Storage Requirements

- **Small dataset** (~1K authors): ~10-50 MB
- **Medium dataset** (~10K authors): ~100-500 MB  
- **Large dataset** (~100K authors): ~1-5 GB

## Notes

- Embeddings are automatically generated on first run
- Use `--force-regenerate` to recreate embeddings
- Storage location can be changed with `--storage-dir` parameter