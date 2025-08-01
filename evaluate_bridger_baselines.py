#!/usr/bin/env python3
"""
Comprehensive evaluation of Bridger baselines on BetterTeaming benchmark.

This script evaluates both ST and sTdM baselines with multiple metrics
commonly used in recommendation system evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set
import logging
from pathlib import Path
import argparse

from bridger_adapter import BridgerBetterTeamingAdapter, create_synthetic_embeddings_from_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_ndcg_at_k(ground_truth: Set[str], recommendations: List[str], k: int) -> float:
    """Calculate NDCG@k metric."""
    if not ground_truth or not recommendations:
        return 0.0
    
    # DCG@k
    dcg = 0.0
    for i, rec in enumerate(recommendations[:k]):
        if rec in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
    
    # IDCG@k - ideal DCG
    ideal_k = min(k, len(ground_truth))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_at_k(ground_truth: Set[str], recommendations: List[str], k: int) -> float:
    """Calculate Precision@k metric."""
    if not recommendations[:k]:
        return 0.0
    
    relevant_recs = sum(1 for rec in recommendations[:k] if rec in ground_truth)
    return relevant_recs / min(k, len(recommendations))


def calculate_recall_at_k(ground_truth: Set[str], recommendations: List[str], k: int) -> float:
    """Calculate Recall@k metric."""
    if not ground_truth:
        return 0.0
    
    relevant_recs = sum(1 for rec in recommendations[:k] if rec in ground_truth)
    return relevant_recs / len(ground_truth)


def comprehensive_evaluation(adapter: BridgerBetterTeamingAdapter,
                           data_path: str,
                           baseline: str = 'st',
                           k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """
    Run comprehensive evaluation with multiple metrics.
    
    Args:
        adapter: BridgerBetterTeamingAdapter instance
        data_path: Path to BetterTeaming CSV file
        baseline: 'st' or 'stdm'
        k_values: List of k values for @k metrics
        
    Returns:
        Dict with all evaluation metrics
    """
    df = adapter.load_betterteaming_data(data_path)
    
    metrics = {f'Hit@{k}': [] for k in k_values}
    metrics.update({f'NDCG@{k}': [] for k in k_values})
    metrics.update({f'Precision@{k}': [] for k in k_values})
    metrics.update({f'Recall@{k}': [] for k in k_values})
    metrics['MRR'] = []
    
    valid_queries = 0
    
    for idx, row in df.iterrows():
        # Parse team authors
        team_authors = adapter.parse_authors_list(row['author2'])
        if not team_authors:
            continue
            
        # Parse ground truth
        ground_truth = adapter.parse_ground_truth(row['ground_truth_authors'])
        if not ground_truth:
            continue
        
        # Get recommendations (use max k value)
        max_k = max(k_values)
        try:
            recommendations = adapter.get_recommendations_for_team(
                team_authors, 
                baseline=baseline,
                top_k=max_k
            )
            
            if not recommendations:
                continue
            
            valid_queries += 1
            rec_authors = [rec[0] for rec in recommendations]
            gt_set = set(ground_truth)
            
            # Calculate metrics for each k
            for k in k_values:
                # Hit@k (binary: did we get at least one relevant item?)
                hits = len(set(rec_authors[:k]).intersection(gt_set))
                metrics[f'Hit@{k}'].append(1.0 if hits > 0 else 0.0)
                
                # NDCG@k
                ndcg = calculate_ndcg_at_k(gt_set, rec_authors, k)
                metrics[f'NDCG@{k}'].append(ndcg)
                
                # Precision@k
                precision = calculate_precision_at_k(gt_set, rec_authors, k)
                metrics[f'Precision@{k}'].append(precision)
                
                # Recall@k
                recall = calculate_recall_at_k(gt_set, rec_authors, k)
                metrics[f'Recall@{k}'].append(recall)
            
            # MRR - Mean Reciprocal Rank
            rr = 0.0
            for i, rec_author in enumerate(rec_authors):
                if rec_author in gt_set:
                    rr = 1.0 / (i + 1)
                    break
            metrics['MRR'].append(rr)
            
            if idx % 100 == 0:
                logger.info(f"Processed {idx} queries (valid: {valid_queries})...")
                
        except Exception as e:
            logger.warning(f"Error processing query {idx}: {e}")
            continue
    
    # Calculate averages
    results = {}
    for metric_name, values in metrics.items():
        if values:
            results[metric_name] = np.mean(values)
        else:
            results[metric_name] = 0.0
    
    results['Valid_Queries'] = valid_queries
    results['Total_Queries'] = len(df)
    
    logger.info(f"Evaluation completed: {valid_queries}/{len(df)} valid queries")
    return results


def print_results_table(st_results: Dict[str, float], 
                       stdm_results: Dict[str, float],
                       data_name: str):
    """Print formatted results table."""
    print("\n" + "="*80)
    print("BRIDGER BASELINES COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Dataset: {data_name}")
    print(f"Valid Queries: {int(st_results.get('Valid_Queries', 0))}/{int(st_results.get('Total_Queries', 0))}")
    print()
    
    # Header
    print(f"{'Metric':<15} {'ST Baseline':<15} {'sTdM Baseline':<15} {'Improvement':<15}")
    print("-" * 65)
    
    # Key metrics to display
    key_metrics = ['MRR', 'Hit@1', 'Hit@3', 'Hit@5', 'Hit@10', 
                   'NDCG@5', 'NDCG@10', 'Precision@5', 'Precision@10', 
                   'Recall@5', 'Recall@10']
    
    for metric in key_metrics:
        if metric in st_results and metric in stdm_results:
            st_val = st_results[metric]
            stdm_val = stdm_results[metric]
            
            # Calculate improvement (positive means sTdM is better)
            if st_val > 0:
                improvement = ((stdm_val - st_val) / st_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{metric:<15} {st_val:<15.4f} {stdm_val:<15.4f} {improvement_str:<15}")
    
    print()


def analyze_performance_by_team_size(adapter: BridgerBetterTeamingAdapter,
                                   data_path: str) -> Dict[str, Dict]:
    """Analyze performance by team size."""
    df = adapter.load_betterteaming_data(data_path)
    
    team_size_results = {}
    
    for idx, row in df.iterrows():
        team_authors = adapter.parse_authors_list(row['author2'])
        ground_truth = adapter.parse_ground_truth(row['ground_truth_authors'])
        
        if not team_authors or not ground_truth:
            continue
        
        team_size = len(team_authors)
        if team_size not in team_size_results:
            team_size_results[team_size] = {'count': 0, 'avg_gt_size': []}
        
        team_size_results[team_size]['count'] += 1
        team_size_results[team_size]['avg_gt_size'].append(len(ground_truth))
    
    # Calculate averages
    for size, data in team_size_results.items():
        data['avg_gt_size'] = np.mean(data['avg_gt_size'])
    
    return team_size_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Bridger baselines on BetterTeaming benchmark')
    parser.add_argument('--data', 
                       default='/data/jx4237data/Graph-CoT/Pipeline/step1_process/strict_0.88_remove_case1_year2-5/paper_levels_0.88_year2-5.csv',
                       help='Path to BetterTeaming CSV file')
    parser.add_argument('--embedding-dim', type=int, default=768,
                       help='Embedding dimension for synthetic embeddings')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 3, 5, 10],
                       help='K values for evaluation metrics')
    
    args = parser.parse_args()
    
    logger.info("Creating synthetic embeddings from data...")
    task_embeddings, method_embeddings = create_synthetic_embeddings_from_data(
        args.data, 
        embedding_dim=args.embedding_dim
    )
    
    logger.info("Initializing adapter...")
    adapter = BridgerBetterTeamingAdapter(task_embeddings, method_embeddings)
    
    logger.info("Running ST baseline evaluation...")
    st_results = comprehensive_evaluation(
        adapter, 
        args.data, 
        baseline='st', 
        k_values=args.k_values
    )
    
    logger.info("Running sTdM baseline evaluation...")
    stdm_results = comprehensive_evaluation(
        adapter, 
        args.data, 
        baseline='stdm', 
        k_values=args.k_values
    )
    
    # Print results
    data_name = Path(args.data).name
    print_results_table(st_results, stdm_results, data_name)
    
    # Analyze by team size
    logger.info("Analyzing performance by team size...")
    team_size_analysis = analyze_performance_by_team_size(adapter, args.data)
    
    print("TEAM SIZE ANALYSIS")
    print("-" * 40)
    print(f"{'Team Size':<12} {'Count':<8} {'Avg GT Size':<12}")
    print("-" * 40)
    for size in sorted(team_size_analysis.keys()):
        data = team_size_analysis[size]
        print(f"{size:<12} {data['count']:<8} {data['avg_gt_size']:<12.2f}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("Next steps:")
    print("1. Replace synthetic embeddings with real Bridger embeddings")
    print("2. Compare these results with MATRIX performance")
    print("3. The baselines are ready for your WSDM 2026 paper!")


if __name__ == "__main__":
    main()