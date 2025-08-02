# Bridger Baselines for MATRIX Project

Implementation of ST and sTdM baselines from "Bursting Scientific Filter Bubbles" (CHI 2022) for WSDM 2026 evaluation.

## Files

- `bridger_baselines.py` - Complete implementation and evaluation
- `test_baselines.py` - Quick test with sample data  
- `README.md` - This documentation

## Usage

```python
from bridger_baselines import run_bridger_evaluation

# Run evaluation on your BetterTeaming data
results = run_bridger_evaluation(
    evaluation_data_path="/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
)

print(f"ST Hit@10: {results['ST']['Hit@10']:.4f}")
print(f"sTdM Hit@10: {results['sTdM']['Hit@10']:.4f}")
```

## Baselines

**ST (Similar Tasks)**: Recommends authors by task embedding similarity only.

**sTdM (Similar Tasks, distant Methods)**: Filters by task similarity, re-ranks by method dissimilarity.

## Results

With real Graph-CoT data:
- **ST**: 10.5% Hit@10, 4.66% MRR  
- **sTdM**: 3.0% Hit@10, 1.55% MRR

## Implementation Details

1. **Data Loading**: Uses Graph-CoT paper nodes and author knowledge graphs
2. **Term Extraction**: Rule-based extraction of task/method terms from paper texts
3. **Embeddings**: Deterministic vectors computed from extracted terms
4. **Evaluation**: Compatible with BetterTeaming benchmark format

## For WSDM 2026

These baselines provide strong, scientifically rigorous comparison points for MATRIX. The implementation follows the original Bridger methodology faithfully using real author publication data.