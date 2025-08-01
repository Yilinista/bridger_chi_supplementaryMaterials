# Bridger Baselines Implementation

This directory contains the implementation of the ST and sTdM baselines from the CHI 2022 paper "Bursting Scientific Filter Bubbles" (the Bridger system), adapted for evaluation on the BetterTeaming benchmark for the MATRIX project.

## Files

- `bridger_baselines.py` - Main implementation of the ST and sTdM baselines
- `test_bridger_baselines.py` - Testing script with synthetic data
- `README_BASELINES.md` - This documentation file

## Baselines Implemented

### 1. ST (Similar Tasks)
- **Description**: Recommend authors based only on the similarity of their "Tasks" facet vectors
- **Implementation**: Uses cosine similarity between focal author's task embedding and all candidate authors' task embeddings
- **Key Function**: `st_baseline()`

### 2. sTdM (Similar Tasks, distant Methods)  
- **Description**: Two-step process that first filters for the top-K authors by "Tasks" similarity, then re-ranks this group by "Methods" dissimilarity (most different first)
- **Implementation**: 
  1. Filter top-K authors by task similarity (smallest cosine distances)
  2. Re-rank by method dissimilarity (largest cosine distances) 
- **Key Function**: `stdm_baseline()`

## Key Code Mappings from Bridger

The implementation is based on analysis of the original Bridger codebase:

- **Embedding aggregation**: Based on `get_avg_embeddings` in `average_embeddings.py:73-84`
- **ST logic**: Based on `_get_avg_embedding_cosine_distance` in `user_study.py:780-790` 
- **sTdM logic**: Based on `sort_distance_df` in `util.py:695-718`
- **Workflow**: Based on `get_user_study_single_author_id.py`

## Usage

```python
from bridger_baselines import BridgerBaselines

# Initialize with author embeddings
baselines = BridgerBaselines(
    task_embeddings=task_embeddings_dict,     # Dict[author_id, np.ndarray]
    method_embeddings=method_embeddings_dict, # Dict[author_id, np.ndarray] 
    author_ids=list_of_author_ids            # List[int]
)

# Get ST recommendations
st_recs = baselines.st_baseline(
    focal_author_id=123,
    exclude_authors=[124, 125],  # e.g., coauthors
    top_k=10
)

# Get sTdM recommendations  
stdm_recs = baselines.stdm_baseline(
    focal_author_id=123,
    exclude_authors=[124, 125],
    filter_k=1000,  # First filter to top-1000 by task similarity
    top_k=10        # Then re-rank and return top-10
)
```

## Testing

Run the test script to see the baselines in action with synthetic data:

```bash
python test_bridger_baselines.py
```

The test script demonstrates:
- Individual baseline functionality
- Comparison between ST and sTdM recommendations
- Simulated evaluation on a BetterTeaming-style benchmark

## Integration Steps

To integrate with actual data and evaluation:

1. **Load Bridger Embeddings**: Implement the `load_bridger_embeddings()` function to load from the actual Bridger data pipeline outputs:
   - `average_author_embeddings_task_pandas.pickle`
   - `average_author_embeddings_method_pandas.pickle`  
   - `mat_author_task_row_labels.npy`
   - `mat_author_method_row_labels.npy`

2. **BetterTeaming Integration**: Implement the `evaluate_baselines_on_betterteaming()` function to:
   - Load BetterTeaming test queries
   - Compute MRR, Hit@k, NDCG@k metrics
   - Compare against MATRIX performance

3. **Adapt Author ID Mappings**: Ensure author IDs are consistent between Bridger embeddings and BetterTeaming benchmark data.

## Expected Results

Based on the Bridger paper, we expect:
- **ST**: Good performance on task-similar recommendations but may miss expertise gaps
- **sTdM**: Better at finding complementary expertise (distant methods) while maintaining task relevance
- **MATRIX**: Should outperform both by more precisely targeting expertise gaps

## Implementation Notes

- Cosine distance is used for similarity computation (matching Bridger's approach)
- Embeddings are L2-normalized for consistent similarity calculations
- Exclusion lists handle coauthor filtering (standard in author recommendation)
- The filter_k parameter in sTdM defaults to 1000 (matching Bridger's implementation)

## Citation

If you use this implementation, please cite both:

1. The original Bridger paper:
   ```
   [Bridger CHI 2022 citation]
   ```

2. The MATRIX paper (when published):
   ```  
   [MATRIX WSDM 2026 citation]
   ```