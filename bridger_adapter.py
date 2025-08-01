#!/usr/bin/env python3
"""
Adapter for Bridger baselines to work with MATRIX/BetterTeaming data format.

This module adapts the Bridger baselines to work with the specific data format
used in the MATRIX project's BetterTeaming benchmark.
"""

import pandas as pd
import numpy as np
import ast
from typing import Dict, List, Tuple, Set
import logging
from pathlib import Path

from bridger_baselines import BridgerBaselines

logger = logging.getLogger(__name__)


class BridgerBetterTeamingAdapter:
    """Adapter to run Bridger baselines on BetterTeaming-format data."""
    
    def __init__(self, 
                 task_embeddings: Dict[str, np.ndarray],
                 method_embeddings: Dict[str, np.ndarray]):
        """
        Initialize adapter with author embeddings.
        
        Args:
            task_embeddings: Dict mapping author_id (str) -> task embedding vector
            method_embeddings: Dict mapping author_id (str) -> method embedding vector
        """
        self.task_embeddings = task_embeddings
        self.method_embeddings = method_embeddings
        
        # Get all author IDs (union of both embedding sets)
        all_authors = set(task_embeddings.keys()) | set(method_embeddings.keys())
        self.author_ids = sorted(list(all_authors))
        
        # Convert author IDs to integers for BridgerBaselines
        self.author_id_to_int = {aid: i for i, aid in enumerate(self.author_ids)}
        self.int_to_author_id = {i: aid for aid, i in self.author_id_to_int.items()}
        
        # Convert embeddings to use integer keys
        task_emb_int = {}
        method_emb_int = {}
        
        for i, author_id in enumerate(self.author_ids):
            if author_id in task_embeddings:
                task_emb_int[i] = task_embeddings[author_id]
            if author_id in method_embeddings:
                method_emb_int[i] = method_embeddings[author_id]
        
        # Initialize Bridger baselines
        self.baselines = BridgerBaselines(
            task_embeddings=task_emb_int,
            method_embeddings=method_emb_int, 
            author_ids=list(range(len(self.author_ids)))
        )
        
        logger.info(f"Initialized adapter with {len(self.author_ids)} authors")
        logger.info(f"Task embeddings: {len(task_embeddings)}, Method embeddings: {len(method_embeddings)}")
    
    def load_betterteaming_data(self, data_path: str) -> pd.DataFrame:
        """Load BetterTeaming format data."""
        logger.info(f"Loading BetterTeaming data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} queries")
        return df
    
    def parse_authors_list(self, authors_str: str) -> List[str]:
        """Parse author list from string format."""
        try:
            return ast.literal_eval(authors_str)
        except:
            logger.warning(f"Could not parse authors string: {authors_str}")
            return []
    
    def parse_ground_truth(self, gt_str: str) -> List[str]:
        """Parse ground truth authors from pipe-separated format."""
        if pd.isna(gt_str):
            return []
        return [author.strip() for author in gt_str.split('|')]
    
    def get_recommendations_for_team(self, 
                                   team_authors: List[str],
                                   baseline: str = 'st',
                                   top_k: int = 10,
                                   filter_k: int = 1000) -> List[Tuple[str, float]]:
        """
        Get recommendations for a team of authors.
        
        Args:
            team_authors: List of author IDs currently on the team
            baseline: 'st' or 'stdm' 
            top_k: Number of recommendations to return
            filter_k: Filter parameter for sTdM baseline
            
        Returns:
            List of (author_id, score) tuples
        """
        # Convert team authors to integer IDs, skip unknown authors
        team_int_ids = []
        for author_id in team_authors:
            if author_id in self.author_id_to_int:
                team_int_ids.append(self.author_id_to_int[author_id])
        
        if not team_int_ids:
            logger.warning("No valid team members found in embeddings")
            return []
        
        # For teams, we'll use the first author as focal (or could average embeddings)
        # This is a simplification - in practice you might want to aggregate team embeddings
        focal_author_int = team_int_ids[0]
        
        # Exclude all team members from recommendations
        exclude_authors = team_int_ids
        
        # Get recommendations
        if baseline == 'st':
            recs = self.baselines.st_baseline(
                focal_author_int, 
                exclude_authors=exclude_authors,
                top_k=top_k
            )
        elif baseline == 'stdm':
            recs = self.baselines.stdm_baseline(
                focal_author_int,
                exclude_authors=exclude_authors,
                filter_k=filter_k,
                top_k=top_k
            )
        else:
            raise ValueError(f"Unknown baseline: {baseline}")
        
        # Convert back to string author IDs
        str_recs = []
        for author_int, score in recs:
            author_id = self.int_to_author_id[author_int]
            str_recs.append((author_id, score))
        
        return str_recs
    
    def evaluate_on_betterteaming(self, 
                                data_path: str,
                                baseline: str = 'st',
                                top_k: int = 10) -> Dict[str, float]:
        """
        Evaluate baseline on BetterTeaming benchmark.
        
        Args:
            data_path: Path to BetterTeaming CSV file
            baseline: 'st' or 'stdm'
            top_k: Number of recommendations to evaluate
            
        Returns:
            Dict with evaluation metrics
        """
        df = self.load_betterteaming_data(data_path)
        
        total_queries = 0
        total_hits = 0
        mrr_sum = 0.0
        
        for idx, row in df.iterrows():
            # Parse team authors
            team_authors = self.parse_authors_list(row['author2'])
            if not team_authors:
                continue
                
            # Parse ground truth
            ground_truth = self.parse_ground_truth(row['ground_truth_authors'])
            if not ground_truth:
                continue
            
            # Get recommendations
            try:
                recommendations = self.get_recommendations_for_team(
                    team_authors, 
                    baseline=baseline,
                    top_k=top_k
                )
                
                if not recommendations:
                    continue
                
                total_queries += 1
                
                # Calculate metrics
                rec_authors = [rec[0] for rec in recommendations]
                gt_set = set(ground_truth)
                
                # Hit@k
                hits = len(set(rec_authors).intersection(gt_set))
                total_hits += hits
                
                # MRR - find first relevant recommendation
                rr = 0.0
                for i, rec_author in enumerate(rec_authors):
                    if rec_author in gt_set:
                        rr = 1.0 / (i + 1)
                        break
                mrr_sum += rr
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx} queries...")
                    
            except Exception as e:
                logger.warning(f"Error processing query {idx}: {e}")
                continue
        
        # Calculate final metrics
        metrics = {}
        if total_queries > 0:
            metrics[f'Hit@{top_k}'] = total_hits / (total_queries * len(ground_truth) if ground_truth else 1)
            metrics['MRR'] = mrr_sum / total_queries
            metrics['Total_Queries'] = total_queries
            metrics['Total_Hits'] = total_hits
        
        logger.info(f"Evaluation completed: {total_queries} queries processed")
        return metrics


def create_synthetic_embeddings_from_data(data_path: str, 
                                        embedding_dim: int = 768) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Create synthetic embeddings for all authors found in the dataset.
    
    This is a placeholder until real embeddings are available.
    """
    df = pd.read_csv(data_path)
    
    # Collect all unique author IDs
    all_authors = set()
    
    for idx, row in df.iterrows():
        # Authors from teams
        try:
            team_authors = ast.literal_eval(row['author2'])
            all_authors.update(team_authors)
        except:
            pass
            
        # Authors from ground truth
        if pd.notna(row['ground_truth_authors']):
            gt_authors = row['ground_truth_authors'].split('|')
            all_authors.update([a.strip() for a in gt_authors])
    
    all_authors = sorted(list(all_authors))
    logger.info(f"Found {len(all_authors)} unique authors in dataset")
    
    # Generate synthetic embeddings
    np.random.seed(42)
    
    task_embeddings = {}
    method_embeddings = {}
    
    for author_id in all_authors:
        # Create some structure in embeddings based on author ID
        author_hash = hash(author_id) % 1000
        
        # Task embeddings - some clustering
        task_cluster = author_hash % 10
        task_base = np.random.randn(embedding_dim) * 0.1
        task_center = np.random.randn(embedding_dim) 
        task_emb = task_base + task_center * (task_cluster / 10.0)
        task_embeddings[author_id] = task_emb / np.linalg.norm(task_emb)
        
        # Method embeddings - different clustering  
        method_cluster = (author_hash + 123) % 8
        method_base = np.random.randn(embedding_dim) * 0.1
        method_center = np.random.randn(embedding_dim)
        method_emb = method_base + method_center * (method_cluster / 8.0)
        method_embeddings[author_id] = method_emb / np.linalg.norm(method_emb)
    
    return task_embeddings, method_embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    data_path = "/data/jx4237data/Graph-CoT/Pipeline/step1_process/strict_0.88_remove_case1_year2-5/paper_levels_0.88_year2-5.csv"
    
    logger.info("Creating synthetic embeddings from data...")
    task_embeddings, method_embeddings = create_synthetic_embeddings_from_data(data_path)
    
    logger.info("Initializing adapter...")
    adapter = BridgerBetterTeamingAdapter(task_embeddings, method_embeddings)
    
    logger.info("Running evaluation...")
    st_results = adapter.evaluate_on_betterteaming(data_path, baseline='st', top_k=10)
    stdm_results = adapter.evaluate_on_betterteaming(data_path, baseline='stdm', top_k=10)
    
    print("\n" + "="*60)
    print("BRIDGER BASELINES EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {Path(data_path).name}")
    print(f"Queries processed: {st_results.get('Total_Queries', 0)}")
    print()
    
    print("ST (Similar Tasks) Baseline:")
    for metric, value in st_results.items():
        if metric not in ['Total_Queries', 'Total_Hits']:
            print(f"  {metric}: {value:.4f}")
    print()
    
    print("sTdM (Similar Tasks, distant Methods) Baseline:")
    for metric, value in stdm_results.items():
        if metric not in ['Total_Queries', 'Total_Hits']:
            print(f"  {metric}: {value:.4f}")