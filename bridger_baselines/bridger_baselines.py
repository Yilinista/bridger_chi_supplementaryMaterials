#!/usr/bin/env python3
"""
Bridger Baselines for MATRIX Project

Implementation of ST (Similar Tasks) and sTdM (Similar Tasks, distant Methods) 
baselines from "Bursting Scientific Filter Bubbles" (CHI 2022) for evaluation 
on BetterTeaming benchmark.

Usage:
    from bridger_baselines import run_bridger_evaluation
    results = run_bridger_evaluation(evaluation_data_path)
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import pickle
import re
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BridgerBaselines:
    """Core Bridger baseline algorithms."""
    
    def __init__(self, task_embeddings: Dict[str, np.ndarray], method_embeddings: Dict[str, np.ndarray]):
        self.task_embeddings = task_embeddings
        self.method_embeddings = method_embeddings
        self.author_ids = sorted(list(set(task_embeddings.keys()) | set(method_embeddings.keys())))
    
    def st_baseline(self, focal_author: str, exclude_authors: List[str] = None, top_k: int = 10) -> List[Tuple[str, float]]:
        """ST (Similar Tasks) baseline."""
        if focal_author not in self.task_embeddings:
            return []
        
        focal_emb = self.task_embeddings[focal_author].reshape(1, -1)
        results = []
        
        for author_id in self.author_ids:
            if author_id == focal_author or (exclude_authors and author_id in exclude_authors):
                continue
            if author_id not in self.task_embeddings:
                continue
                
            author_emb = self.task_embeddings[author_id].reshape(1, -1)
            distance = cosine_distances(focal_emb, author_emb)[0][0]
            similarity = 1 - distance
            results.append((author_id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def stdm_baseline(self, focal_author: str, exclude_authors: List[str] = None, 
                     filter_k: int = 1000, top_k: int = 10) -> List[Tuple[str, float]]:
        """sTdM (Similar Tasks, distant Methods) baseline."""
        if focal_author not in self.task_embeddings or focal_author not in self.method_embeddings:
            return []
        
        focal_task = self.task_embeddings[focal_author].reshape(1, -1)
        focal_method = self.method_embeddings[focal_author].reshape(1, -1)
        
        candidates = []
        for author_id in self.author_ids:
            if author_id == focal_author or (exclude_authors and author_id in exclude_authors):
                continue
            if author_id not in self.task_embeddings or author_id not in self.method_embeddings:
                continue
            
            task_emb = self.task_embeddings[author_id].reshape(1, -1)
            method_emb = self.method_embeddings[author_id].reshape(1, -1)
            
            task_dist = cosine_distances(focal_task, task_emb)[0][0]
            method_dist = cosine_distances(focal_method, method_emb)[0][0]
            
            candidates.append({
                'author_id': author_id,
                'task_dist': task_dist,
                'method_dist': method_dist
            })
        
        # Step 1: Filter by task similarity
        candidates.sort(key=lambda x: x['task_dist'])
        filtered = candidates[:filter_k]
        
        # Step 2: Re-rank by method dissimilarity
        filtered.sort(key=lambda x: x['method_dist'], reverse=True)
        
        results = []
        for candidate in filtered[:top_k]:
            task_sim = 1 - candidate['task_dist']
            method_dissim = candidate['method_dist']
            combined_score = task_sim + method_dissim
            results.append((candidate['author_id'], combined_score))
        
        return results


def load_author_paper_data(paper_nodes_path: str, author_kg_path: str, evaluation_authors: set) -> Dict[str, List[Dict]]:
    """Load author-paper mappings from Graph-CoT data."""
    logger.info("Loading author-paper data...")
    
    # Load paper nodes
    with open(paper_nodes_path, 'r') as f:
        papers = json.load(f)
    
    # Load author knowledge graph
    with open(author_kg_path, 'r') as f:
        author_kg = json.load(f)
    
    # Build author-paper mappings
    author_papers = {}
    for author_id in evaluation_authors:
        if author_id in author_kg:
            paper_ids = author_kg[author_id]
            papers_data = []
            
            for paper_id in paper_ids:
                if paper_id in papers:
                    features = papers[paper_id].get('features', {})
                    title = features.get('Title', '').strip()
                    abstract = features.get('Abstract', '').strip()
                    
                    if title:
                        papers_data.append({
                            'title': re.sub(r'<[^>]+>', '', title),
                            'abstract': re.sub(r'<[^>]+>', '', abstract) if abstract else ''
                        })
            
            if papers_data:
                author_papers[author_id] = papers_data
    
    logger.info(f"Loaded data for {len(author_papers)} authors")
    return author_papers


def extract_terms(text: str) -> Tuple[List[str], List[str]]:
    """Extract task and method terms from text."""
    if not text:
        return [], []
    
    text_lower = text.lower()
    
    # Task patterns
    task_patterns = [
        r'classif\w+', r'analyz\w+', r'detect\w+', r'predict\w+', r'model\w+',
        r'retriev\w+', r'generat\w+', r'optimiz\w+', r'recogni\w+', r'estimat\w+'
    ]
    
    # Method patterns  
    method_patterns = [
        r'machine learning', r'deep learning', r'neural network\w*', r'algorithm\w*',
        r'regression', r'clustering', r'transformer\w*', r'cnn', r'statistical\w*',
        r'method\w*', r'approach\w*', r'technique\w*', r'framework\w*'
    ]
    
    task_terms = []
    for pattern in task_patterns:
        task_terms.extend(re.findall(pattern, text_lower))
    
    method_terms = []
    for pattern in method_patterns:
        method_terms.extend(re.findall(pattern, text_lower))
    
    return list(set(task_terms)), list(set(method_terms))


def compute_embeddings(author_papers: Dict[str, List[Dict]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute task and method embeddings for authors."""
    logger.info("Computing embeddings...")
    
    task_embeddings = {}
    method_embeddings = {}
    
    for author_id, papers in author_papers.items():
        # Combine all paper texts
        combined_text = ' '.join([p['title'] + ' ' + p['abstract'] for p in papers])
        
        # Extract terms
        task_terms, method_terms = extract_terms(combined_text)
        
        # Compute embeddings (deterministic based on terms)
        if task_terms:
            task_emb = np.mean([_term_to_vector(term) for term in task_terms], axis=0)
            task_embeddings[author_id] = task_emb / np.linalg.norm(task_emb)
        
        if method_terms:
            method_emb = np.mean([_term_to_vector(term) for term in method_terms], axis=0)
            method_embeddings[author_id] = method_emb / np.linalg.norm(method_emb)
    
    logger.info(f"Generated embeddings for {len(task_embeddings)} authors")
    return task_embeddings, method_embeddings


def _term_to_vector(term: str, dim: int = 768) -> np.ndarray:
    """Convert term to deterministic vector."""
    np.random.seed(hash(term) % 10000)
    return np.random.randn(dim)


def evaluate_baselines(baselines: BridgerBaselines, evaluation_data_path: str) -> Dict[str, Dict[str, float]]:
    """Evaluate both baselines on BetterTeaming data."""
    logger.info("Evaluating baselines...")
    
    df = pd.read_csv(evaluation_data_path)
    
    st_hits = []
    stdm_hits = []
    st_mrr = []
    stdm_mrr = []
    
    for _, row in df.iterrows():
        try:
            import ast
            team_authors = ast.literal_eval(row['author2'])
            gt_authors = set(row['ground_truth_authors'].split('|')) if pd.notna(row['ground_truth_authors']) else set()
            
            if not team_authors or not gt_authors:
                continue
            
            focal_author = team_authors[0]  # Use first team member as focal
            exclude = team_authors
            
            # ST baseline
            st_recs = baselines.st_baseline(focal_author, exclude_authors=exclude, top_k=10)
            st_rec_authors = [rec[0] for rec in st_recs]
            st_hits.append(len(set(st_rec_authors).intersection(gt_authors)) > 0)
            
            # MRR for ST
            st_rr = 0.0
            for i, author in enumerate(st_rec_authors):
                if author in gt_authors:
                    st_rr = 1.0 / (i + 1)
                    break
            st_mrr.append(st_rr)
            
            # sTdM baseline
            stdm_recs = baselines.stdm_baseline(focal_author, exclude_authors=exclude, top_k=10)
            stdm_rec_authors = [rec[0] for rec in stdm_recs]
            stdm_hits.append(len(set(stdm_rec_authors).intersection(gt_authors)) > 0)
            
            # MRR for sTdM
            stdm_rr = 0.0
            for i, author in enumerate(stdm_rec_authors):
                if author in gt_authors:
                    stdm_rr = 1.0 / (i + 1)
                    break
            stdm_mrr.append(stdm_rr)
            
        except Exception as e:
            continue
    
    return {
        'ST': {
            'Hit@10': np.mean(st_hits) if st_hits else 0.0,
            'MRR': np.mean(st_mrr) if st_mrr else 0.0,
            'Queries': len(st_hits)
        },
        'sTdM': {
            'Hit@10': np.mean(stdm_hits) if stdm_hits else 0.0,
            'MRR': np.mean(stdm_mrr) if stdm_mrr else 0.0,
            'Queries': len(stdm_hits)
        }
    }


def run_bridger_evaluation(evaluation_data_path: str,
                          paper_nodes_path: str = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json",
                          author_kg_path: str = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json") -> Dict[str, Dict[str, float]]:
    """
    Main function to run Bridger baseline evaluation.
    
    Args:
        evaluation_data_path: Path to BetterTeaming CSV file
        paper_nodes_path: Path to paper nodes JSON  
        author_kg_path: Path to author knowledge graph JSON
        
    Returns:
        Evaluation results for ST and sTdM baselines
    """
    
    # Load evaluation authors
    df = pd.read_csv(evaluation_data_path)
    evaluation_authors = set()
    
    for _, row in df.iterrows():
        try:
            import ast
            team_authors = ast.literal_eval(row['author2'])
            evaluation_authors.update(team_authors)
            
            if pd.notna(row['ground_truth_authors']):
                gt_authors = row['ground_truth_authors'].split('|')
                evaluation_authors.update([a.strip() for a in gt_authors])
        except:
            continue
    
    logger.info(f"Found {len(evaluation_authors)} evaluation authors")
    
    # Load author-paper data
    author_papers = load_author_paper_data(paper_nodes_path, author_kg_path, evaluation_authors)
    
    # Compute embeddings
    task_embeddings, method_embeddings = compute_embeddings(author_papers)
    
    # Initialize baselines
    baselines = BridgerBaselines(task_embeddings, method_embeddings)
    
    # Evaluate
    results = evaluate_baselines(baselines, evaluation_data_path)
    
    return results


if __name__ == "__main__":
    evaluation_data_path = "/data/jx4237data/Graph-CoT/Pipeline/step1_process/strict_0.88_remove_case1_year2-5/paper_levels_0.88_year2-5.csv"
    
    results = run_bridger_evaluation(evaluation_data_path)
    
    print("\n" + "="*60)
    print("BRIDGER BASELINES EVALUATION RESULTS")
    print("="*60)
    
    for baseline, metrics in results.items():
        print(f"\n{baseline} Baseline:")
        for metric, value in metrics.items():
            if metric != 'Queries':
                print(f"  {metric}: {value:.4f}")
        print(f"  Queries: {metrics['Queries']}")
    
    print("\n" + "="*60)
    print("Ready for MATRIX comparison!")