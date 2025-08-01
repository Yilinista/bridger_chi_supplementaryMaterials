#!/usr/bin/env python3
"""
Bridger Baselines Implementation for MATRIX Project

This module implements the ST (Similar Tasks) and sTdM (Similar Tasks, distant Methods) 
baselines from the CHI 2022 paper "Bursting Scientific Filter Bubbles" for evaluation 
on the BetterTeaming benchmark.

Baselines:
- ST: Recommend authors based on Tasks similarity only
- sTdM: Filter by top-K Tasks similarity, then re-rank by Methods dissimilarity
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class BridgerBaselines:
    """Implementation of Bridger recommendation baselines."""
    
    def __init__(self, 
                 task_embeddings: Dict[int, np.ndarray],
                 method_embeddings: Dict[int, np.ndarray],
                 author_ids: List[int]):
        """
        Initialize Bridger baselines.
        
        Args:
            task_embeddings: Dict mapping author_id -> task embedding vector
            method_embeddings: Dict mapping author_id -> method embedding vector  
            author_ids: List of all candidate author IDs
        """
        self.task_embeddings = task_embeddings
        self.method_embeddings = method_embeddings
        self.author_ids = author_ids
        
        # Convert to matrices for efficient computation
        self.task_matrix = self._build_embedding_matrix(task_embeddings)
        self.method_matrix = self._build_embedding_matrix(method_embeddings)
    
    def _build_embedding_matrix(self, embeddings_dict: Dict[int, np.ndarray]) -> np.ndarray:
        """Build matrix where rows are authors and columns are embedding dimensions."""
        embedding_dim = next(iter(embeddings_dict.values())).shape[0]
        matrix = np.zeros((len(self.author_ids), embedding_dim))
        
        for i, author_id in enumerate(self.author_ids):
            if author_id in embeddings_dict:
                matrix[i] = embeddings_dict[author_id]
        
        return matrix
    
    def _get_task_distances(self, focal_author_id: int) -> np.ndarray:
        """Compute cosine distances for task embeddings."""
        if focal_author_id not in self.task_embeddings:
            raise ValueError(f"Focal author {focal_author_id} not found in task embeddings")
        
        focal_embedding = self.task_embeddings[focal_author_id].reshape(1, -1)
        distances = cosine_distances(focal_embedding, self.task_matrix)[0]
        return distances
    
    def _get_method_distances(self, focal_author_id: int) -> np.ndarray:
        """Compute cosine distances for method embeddings.""" 
        if focal_author_id not in self.method_embeddings:
            raise ValueError(f"Focal author {focal_author_id} not found in method embeddings")
        
        focal_embedding = self.method_embeddings[focal_author_id].reshape(1, -1)
        distances = cosine_distances(focal_embedding, self.method_matrix)[0]
        return distances
    
    def st_baseline(self, 
                   focal_author_id: int, 
                   exclude_authors: Optional[List[int]] = None,
                   top_k: int = 10) -> List[Tuple[int, float]]:
        """
        ST (Similar Tasks) baseline: Recommend authors by task similarity only.
        
        Args:
            focal_author_id: The focal author for whom to make recommendations
            exclude_authors: List of author IDs to exclude from recommendations
            top_k: Number of recommendations to return
            
        Returns:
            List of (author_id, similarity_score) tuples, sorted by similarity
        """
        task_distances = self._get_task_distances(focal_author_id)
        
        # Create results dataframe
        results = []
        for i, author_id in enumerate(self.author_ids):
            if exclude_authors and author_id in exclude_authors:
                continue
            if author_id == focal_author_id:  # Exclude focal author
                continue
                
            similarity = 1 - task_distances[i]  # Convert distance to similarity
            results.append((author_id, similarity))
        
        # Sort by similarity (descending) and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def stdm_baseline(self, 
                     focal_author_id: int,
                     exclude_authors: Optional[List[int]] = None, 
                     filter_k: int = 1000,
                     top_k: int = 10) -> List[Tuple[int, float]]:
        """
        sTdM (Similar Tasks, distant Methods) baseline: Two-step filtering and re-ranking.
        
        Step 1: Filter top-K authors by task similarity
        Step 2: Re-rank by method dissimilarity (most different methods first)
        
        Args:
            focal_author_id: The focal author for whom to make recommendations
            exclude_authors: List of author IDs to exclude from recommendations  
            filter_k: Number of authors to keep after task similarity filtering
            top_k: Final number of recommendations to return
            
        Returns:
            List of (author_id, combined_score) tuples, sorted by the sTdM criteria
        """
        task_distances = self._get_task_distances(focal_author_id)
        method_distances = self._get_method_distances(focal_author_id)
        
        # Step 1: Build candidates dataframe
        candidates = []
        for i, author_id in enumerate(self.author_ids):
            if exclude_authors and author_id in exclude_authors:
                continue
            if author_id == focal_author_id:  # Exclude focal author
                continue
                
            candidates.append({
                'author_id': author_id,
                'task_dist': task_distances[i],
                'method_dist': method_distances[i]
            })
        
        df_candidates = pd.DataFrame(candidates)
        
        # Step 2: Filter by task similarity (keep top filter_k with smallest task distances)
        df_filtered = df_candidates.nsmallest(filter_k, 'task_dist')
        
        # Step 3: Re-rank by method dissimilarity (largest method distances first)  
        df_reranked = df_filtered.nlargest(top_k, 'method_dist')
        
        # Convert back to list of tuples with combined score
        results = []
        for _, row in df_reranked.iterrows():
            # Combined score: high task similarity + high method dissimilarity
            task_similarity = 1 - row['task_dist']
            method_dissimilarity = row['method_dist'] 
            combined_score = task_similarity + method_dissimilarity
            results.append((int(row['author_id']), combined_score))
        
        return results
    
    def get_author_embeddings_summary(self) -> Dict[str, int]:
        """Get summary statistics about the loaded embeddings."""
        return {
            'total_authors': len(self.author_ids),
            'authors_with_task_embeddings': len(self.task_embeddings),
            'authors_with_method_embeddings': len(self.method_embeddings),
            'task_embedding_dim': next(iter(self.task_embeddings.values())).shape[0] if self.task_embeddings else 0,
            'method_embedding_dim': next(iter(self.method_embeddings.values())).shape[0] if self.method_embeddings else 0
        }


def load_bridger_embeddings(embeddings_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]]:
    """
    Load Bridger author embeddings from files.
    
    This is a placeholder function - in practice, you would load the actual 
    embeddings from the Bridger data pipeline outputs.
    
    Args:
        embeddings_path: Path to directory containing embedding files
        
    Returns:
        Tuple of (task_embeddings, method_embeddings, author_ids)
    """
    # TODO: Implement actual loading logic based on Bridger file formats
    # This would load from files like:
    # - average_author_embeddings_task_pandas.pickle  
    # - average_author_embeddings_method_pandas.pickle
    # - mat_author_task_row_labels.npy
    # - mat_author_method_row_labels.npy
    
    raise NotImplementedError("Embedding loading not yet implemented - requires access to Bridger data files")


def evaluate_baselines_on_betterteaming(baselines: BridgerBaselines,
                                       test_queries: List[Dict],
                                       metrics: List[str] = ['MRR', 'Hit@5', 'Hit@10', 'NDCG@5', 'NDCG@10']) -> Dict[str, Dict[str, float]]:
    """
    Evaluate ST and sTdM baselines on BetterTeaming benchmark.
    
    Args:
        baselines: Initialized BridgerBaselines instance
        test_queries: List of test queries, each containing focal_author_id and ground_truth_authors
        metrics: List of metrics to compute
        
    Returns:
        Dict with results for each baseline and metric
    """
    # TODO: Implement evaluation logic compatible with BetterTeaming format
    # This would compute MRR, Hit@k, NDCG@k for both baselines
    
    raise NotImplementedError("BetterTeaming evaluation not yet implemented")


if __name__ == "__main__":
    # Example usage
    logger.info("Bridger Baselines Implementation")
    logger.info("ST: Similar Tasks baseline")  
    logger.info("sTdM: Similar Tasks, distant Methods baseline")
    logger.info("Ready for integration with BetterTeaming evaluation framework")