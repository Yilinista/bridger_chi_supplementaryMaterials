#!/usr/bin/env python3
"""
Quick test for Bridger baselines with sample data.
"""

import numpy as np
from bridger_baselines import BridgerBaselines

def test_with_sample_data():
    """Test baselines with small sample dataset."""
    
    # Create sample embeddings
    np.random.seed(42)
    authors = ['author1', 'author2', 'author3', 'author4', 'author5']
    
    task_embeddings = {author: np.random.randn(768) for author in authors}
    method_embeddings = {author: np.random.randn(768) for author in authors}
    
    # Normalize embeddings
    for author in authors:
        task_embeddings[author] = task_embeddings[author] / np.linalg.norm(task_embeddings[author])
        method_embeddings[author] = method_embeddings[author] / np.linalg.norm(method_embeddings[author])
    
    # Initialize baselines
    baselines = BridgerBaselines(task_embeddings, method_embeddings)
    
    # Test ST baseline
    st_recs = baselines.st_baseline('author1', exclude_authors=['author2'], top_k=3)
    print("ST recommendations for author1:")
    for i, (author, score) in enumerate(st_recs, 1):
        print(f"  {i}. {author}: {score:.4f}")
    
    # Test sTdM baseline  
    stdm_recs = baselines.stdm_baseline('author1', exclude_authors=['author2'], top_k=3)
    print("\nsTdM recommendations for author1:")
    for i, (author, score) in enumerate(stdm_recs, 1):
        print(f"  {i}. {author}: {score:.4f}")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_with_sample_data()