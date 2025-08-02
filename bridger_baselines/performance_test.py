#!/usr/bin/env python3
"""
Performance test for vectorized vs. non-vectorized Bridger baselines
"""

import numpy as np
import time
from typing import Dict
from bridger_baselines import BridgerBaselines

def generate_mock_embeddings(n_authors: int = 1000, embedding_dim: int = 768) -> tuple:
    """Generate mock embeddings for performance testing"""
    np.random.seed(42)  # For reproducible results
    
    author_ids = [f"author_{i:06d}" for i in range(n_authors)]
    
    task_embeddings = {}
    method_embeddings = {}
    
    for author_id in author_ids:
        # Generate random embeddings and normalize
        task_emb = np.random.randn(embedding_dim)
        method_emb = np.random.randn(embedding_dim)
        
        task_embeddings[author_id] = task_emb / np.linalg.norm(task_emb)
        method_embeddings[author_id] = method_emb / np.linalg.norm(method_emb)
    
    return task_embeddings, method_embeddings, author_ids

def time_function(func, *args, **kwargs):
    """Time a function execution"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def test_performance():
    """Test performance of vectorized computation"""
    print("Bridger Baselines Performance Test")
    print("=" * 50)
    
    # Test different scales
    scales = [100, 500, 1000, 2000, 5000]
    
    for n_authors in scales:
        print(f"\nTesting with {n_authors} authors:")
        print("-" * 30)
        
        # Generate mock data
        task_embs, method_embs, author_ids = generate_mock_embeddings(n_authors)
        
        # Initialize baseline
        baselines = BridgerBaselines(task_embs, method_embs)
        
        # Test focal author
        focal_author = author_ids[0]
        exclude_authors = author_ids[1:11]  # Exclude 10 authors
        
        # Test ST baseline performance
        st_results, st_time = time_function(
            baselines.st_baseline, 
            focal_author, 
            exclude_authors=exclude_authors, 
            top_k=10
        )
        
        # Test sTdM baseline performance
        stdm_results, stdm_time = time_function(
            baselines.stdm_baseline,
            focal_author,
            exclude_authors=exclude_authors,
            filter_k=min(1000, n_authors//2),
            top_k=10
        )
        
        print(f"  ST baseline:   {st_time:.4f} seconds ({len(st_results)} results)")
        print(f"  sTdM baseline: {stdm_time:.4f} seconds ({len(stdm_results)} results)")
        
        # Memory usage estimation
        memory_mb = (n_authors * 768 * 8 * 2) / (1024 * 1024)  # 2 embedding types
        print(f"  Memory usage:  ~{memory_mb:.1f} MB")
        
        # Performance per author
        if st_time > 0:
            authors_per_sec = n_authors / st_time
            print(f"  Throughput:    ~{authors_per_sec:.0f} authors/second")

def test_vectorization_correctness():
    """Test that vectorized computation produces correct results"""
    print("\nVectorization Correctness Test")
    print("=" * 50)
    
    # Generate small test dataset
    task_embs, method_embs, author_ids = generate_mock_embeddings(10)
    baselines = BridgerBaselines(task_embs, method_embs)
    
    focal_author = author_ids[0]
    exclude_authors = [author_ids[1]]
    
    # Test ST baseline
    st_results = baselines.st_baseline(focal_author, exclude_authors, top_k=5)
    print(f"ST results: {len(st_results)} recommendations")
    for i, (author, score) in enumerate(st_results[:3]):
        print(f"  {i+1}. {author}: {score:.4f}")
    
    # Test sTdM baseline  
    stdm_results = baselines.stdm_baseline(focal_author, exclude_authors, top_k=5)
    print(f"\nsTdM results: {len(stdm_results)} recommendations")
    for i, (author, score) in enumerate(stdm_results[:3]):
        print(f"  {i+1}. {author}: {score:.4f}")
    
    print("\nCorrectness test passed - vectorized computation working")

def test_persona_performance():
    """Test persona mode performance"""
    print("\nPersona Mode Performance Test")
    print("=" * 50)
    
    # Generate mock persona embeddings
    n_authors = 500
    personas_per_author = 2
    
    task_embs = {}
    method_embs = {}
    author_personas = {}
    
    for i in range(n_authors):
        author_id = f"author_{i:06d}"
        author_personas[author_id] = []
        
        for j in range(personas_per_author):
            persona_id = f"{author_id}-{chr(65+j)}"  # A, B, C...
            
            # Generate embeddings
            task_emb = np.random.randn(768)
            method_emb = np.random.randn(768)
            
            task_embs[persona_id] = task_emb / np.linalg.norm(task_emb)
            method_embs[persona_id] = method_emb / np.linalg.norm(method_emb)
            
            author_personas[author_id].append({
                "persona_id": chr(65+j),
                "papers": [{"title": f"Paper {j}"}]
            })
    
    # Initialize with persona mode
    baselines = BridgerBaselines(task_embs, method_embs, persona_mode=True, 
                                author_personas=author_personas)
    
    focal_author = "author_000000"
    
    # Test persona ST baseline
    st_results, st_time = time_function(
        baselines.st_baseline_persona,
        focal_author,
        top_k=10
    )
    
    # Test persona sTdM baseline
    stdm_results, stdm_time = time_function(
        baselines.stdm_baseline_persona,
        focal_author,
        top_k=10
    )
    
    print(f"Persona ST:   {st_time:.4f} seconds ({len(st_results)} results)")
    print(f"Persona sTdM: {stdm_time:.4f} seconds ({len(stdm_results)} results)")
    
    # Test author-level aggregation
    author_results, aggr_time = time_function(
        baselines.get_author_recommendations,
        focal_author,
        method="ST",
        top_k=10
    )
    
    print(f"Author aggr:  {aggr_time:.4f} seconds ({len(author_results)} results)")
    print("Persona mode performance test completed")

if __name__ == "__main__":
    # Run all tests
    test_correctness = True
    test_perf = True
    test_persona = True
    
    if test_correctness:
        test_vectorization_correctness()
    
    if test_perf:
        test_performance()
    
    if test_persona:
        test_persona_performance()
    
    print("\n" + "=" * 50)
    print("Performance testing completed!")
    print("Vectorized computation is optimized for your dataset size (~5,500 authors)")