#!/usr/bin/env python3
"""
Test script for Bridger baselines with synthetic data.

This demonstrates how the ST and sTdM baselines work using synthetic embeddings,
since we don't have access to the actual Bridger embedding files.
"""

import numpy as np
import logging
from bridger_baselines import BridgerBaselines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_embeddings(num_authors: int = 100, 
                                embedding_dim: int = 768,
                                seed: int = 42) -> tuple:
    """Generate synthetic task and method embeddings for testing."""
    np.random.seed(seed)
    
    author_ids = list(range(1, num_authors + 1))
    
    # Generate synthetic task embeddings (some clusters)
    task_embeddings = {}
    for i, author_id in enumerate(author_ids):
        # Create some clustering structure
        cluster_id = i % 5
        base_vector = np.random.randn(embedding_dim) * 0.1
        cluster_center = np.random.randn(embedding_dim) 
        task_embeddings[author_id] = base_vector + cluster_center * (cluster_id / 5.0)
        task_embeddings[author_id] = task_embeddings[author_id] / np.linalg.norm(task_embeddings[author_id])
    
    # Generate synthetic method embeddings (different clustering)
    method_embeddings = {}
    for i, author_id in enumerate(author_ids):
        # Different clustering structure for methods
        cluster_id = (i + 2) % 4
        base_vector = np.random.randn(embedding_dim) * 0.1  
        cluster_center = np.random.randn(embedding_dim)
        method_embeddings[author_id] = base_vector + cluster_center * (cluster_id / 4.0)
        method_embeddings[author_id] = method_embeddings[author_id] / np.linalg.norm(method_embeddings[author_id])
    
    return task_embeddings, method_embeddings, author_ids


def test_st_baseline():
    """Test the ST (Similar Tasks) baseline."""
    logger.info("Testing ST (Similar Tasks) baseline...")
    
    # Generate synthetic data
    task_embeddings, method_embeddings, author_ids = generate_synthetic_embeddings()
    
    # Initialize baselines
    baselines = BridgerBaselines(task_embeddings, method_embeddings, author_ids)
    
    # Print summary
    summary = baselines.get_author_embeddings_summary()
    logger.info(f"Embeddings summary: {summary}")
    
    # Test recommendations for a focal author
    focal_author = 1
    recommendations = baselines.st_baseline(focal_author, top_k=5)
    
    logger.info(f"ST recommendations for author {focal_author}:")
    for i, (author_id, score) in enumerate(recommendations, 1):
        logger.info(f"  {i}. Author {author_id}: similarity = {score:.4f}")
    
    return recommendations


def test_stdm_baseline():
    """Test the sTdM (Similar Tasks, distant Methods) baseline."""
    logger.info("\nTesting sTdM (Similar Tasks, distant Methods) baseline...")
    
    # Generate synthetic data
    task_embeddings, method_embeddings, author_ids = generate_synthetic_embeddings()
    
    # Initialize baselines
    baselines = BridgerBaselines(task_embeddings, method_embeddings, author_ids)
    
    # Test recommendations for a focal author
    focal_author = 1
    recommendations = baselines.stdm_baseline(focal_author, filter_k=50, top_k=5)
    
    logger.info(f"sTdM recommendations for author {focal_author}:")
    for i, (author_id, score) in enumerate(recommendations, 1):
        logger.info(f"  {i}. Author {author_id}: combined score = {score:.4f}")
    
    return recommendations


def compare_baselines():
    """Compare ST and sTdM recommendations for the same focal author."""
    logger.info("\n" + "="*60)
    logger.info("COMPARING ST vs sTdM BASELINES")
    logger.info("="*60)
    
    # Generate synthetic data
    task_embeddings, method_embeddings, author_ids = generate_synthetic_embeddings()
    baselines = BridgerBaselines(task_embeddings, method_embeddings, author_ids)
    
    focal_author = 1
    exclude_coauthors = [2, 3, 4]  # Simulate excluding coauthors
    
    # Get recommendations from both baselines
    st_recs = baselines.st_baseline(focal_author, exclude_authors=exclude_coauthors, top_k=10)
    stdm_recs = baselines.stdm_baseline(focal_author, exclude_authors=exclude_coauthors, top_k=10)
    
    logger.info(f"\nFocal Author: {focal_author}")
    logger.info(f"Excluded Authors: {exclude_coauthors}")
    logger.info("\nTop 10 Recommendations Comparison:")
    logger.info(f"{'Rank':<4} {'ST Baseline':<15} {'sTdM Baseline':<15} {'Overlap':<8}")
    logger.info("-" * 50)
    
    st_authors = [rec[0] for rec in st_recs]
    stdm_authors = [rec[0] for rec in stdm_recs]
    
    for i in range(10):
        st_author = st_authors[i] if i < len(st_authors) else "N/A"
        stdm_author = stdm_authors[i] if i < len(stdm_authors) else "N/A"
        overlap = "✓" if st_author == stdm_author and st_author != "N/A" else ""
        logger.info(f"{i+1:<4} {st_author:<15} {stdm_author:<15} {overlap:<8}")
    
    # Calculate overlap statistics
    st_set = set(st_authors)
    stdm_set = set(stdm_authors)
    overlap_count = len(st_set.intersection(stdm_set))
    
    logger.info(f"\nOverlap Statistics:")
    logger.info(f"  Common recommendations: {overlap_count}/10")
    logger.info(f"  ST unique: {len(st_set - stdm_set)}")
    logger.info(f"  sTdM unique: {len(stdm_set - st_set)}")


def simulate_betterteaming_evaluation():
    """Simulate evaluation on BetterTeaming-style benchmark."""
    logger.info("\n" + "="*60)
    logger.info("SIMULATING BETTERTEAMING EVALUATION")  
    logger.info("="*60)
    
    # Generate synthetic data
    task_embeddings, method_embeddings, author_ids = generate_synthetic_embeddings(num_authors=500)
    baselines = BridgerBaselines(task_embeddings, method_embeddings, author_ids)
    
    # Simulate some test queries
    test_queries = [
        {'focal_author': 1, 'ground_truth': [15, 32, 47, 89, 156]},
        {'focal_author': 25, 'ground_truth': [8, 91, 134, 267, 301]}, 
        {'focal_author': 50, 'ground_truth': [12, 67, 123, 234, 445]},
    ]
    
    logger.info(f"Evaluating on {len(test_queries)} test queries...")
    
    st_hits = 0
    stdm_hits = 0
    
    for query in test_queries:
        focal = query['focal_author']
        ground_truth = set(query['ground_truth'])
        
        # Get recommendations (top-10)
        st_recs = baselines.st_baseline(focal, top_k=10)
        stdm_recs = baselines.stdm_baseline(focal, top_k=10)
        
        st_recommended = set([rec[0] for rec in st_recs])
        stdm_recommended = set([rec[0] for rec in stdm_recs])
        
        # Count hits@10
        st_hit = len(st_recommended.intersection(ground_truth))
        stdm_hit = len(stdm_recommended.intersection(ground_truth))
        
        st_hits += st_hit
        stdm_hits += stdm_hit
        
        logger.info(f"Query {focal}: ST hits = {st_hit}, sTdM hits = {stdm_hit}")
    
    # Simple metrics
    total_possible = sum(len(q['ground_truth']) for q in test_queries)
    st_recall = st_hits / total_possible
    stdm_recall = stdm_hits / total_possible
    
    logger.info(f"\nSimple Evaluation Results:")
    logger.info(f"  ST Baseline - Total Hits: {st_hits}/{total_possible} (Recall: {st_recall:.3f})")
    logger.info(f"  sTdM Baseline - Total Hits: {stdm_hits}/{total_possible} (Recall: {stdm_recall:.3f})")


if __name__ == "__main__":
    logger.info("Bridger Baselines Testing Script")
    logger.info("="*60)
    
    # Run individual tests
    test_st_baseline()
    test_stdm_baseline()
    
    # Compare baselines
    compare_baselines()
    
    # Simulate evaluation
    simulate_betterteaming_evaluation()
    
    logger.info("\n" + "="*60)
    logger.info("Testing completed!")
    logger.info("Ready for integration with actual Bridger embeddings and BetterTeaming evaluation.")