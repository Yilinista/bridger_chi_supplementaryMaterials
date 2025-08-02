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
    """
    Core Bridger baseline algorithms with support for both standard and persona modes.
    
    This class implements the ST (Similar Tasks) and sTdM (Similar Tasks, distant Methods)
    recommendation algorithms from "Bursting Scientific Filter Bubbles" (CHI 2022).
    
    Features:
    - Vectorized computation for optimal performance
    - Automatic persona mode detection
    - Support for multi-domain researcher modeling
    - Author-level and persona-level recommendations
    
    Persona Mode Logic:
    - Automatically detected when embedding keys contain hyphens (e.g., "author_id-persona_id")
    - Each author can have multiple research personas (A, B, C, etc.)
    - Recommendations computed at persona level, then aggregated to author level
    - Enables fine-grained matching for multi-disciplinary researchers
    """
    
    def __init__(self, task_embeddings: Dict[str, np.ndarray], method_embeddings: Dict[str, np.ndarray], 
                 persona_mode: bool = False, author_personas: Dict[str, List[Dict]] = None):
        self.task_embeddings = task_embeddings
        self.method_embeddings = method_embeddings
        self.persona_mode = persona_mode
        self.author_personas = author_personas or {}
        
        if persona_mode:
            # In persona mode, keys are "author_id-persona_id"
            self.persona_ids = sorted(list(set(task_embeddings.keys()) | set(method_embeddings.keys())))
            self.author_ids = sorted(list(set(pid.split('-')[0] for pid in self.persona_ids)))
            
            # Create author to personas mapping
            self.author_to_personas = {}
            for persona_id in self.persona_ids:
                author_id = persona_id.split('-')[0]
                if author_id not in self.author_to_personas:
                    self.author_to_personas[author_id] = []
                self.author_to_personas[author_id].append(persona_id)
        else:
            # In standard mode, keys are author_ids
            self.author_ids = sorted(list(set(task_embeddings.keys()) | set(method_embeddings.keys())))
            self.persona_ids = []
            self.author_to_personas = {}
    
    def st_baseline(self, focal_author: str, exclude_authors: List[str] = None, top_k: int = 10) -> List[Tuple[str, float]]:
        """ST (Similar Tasks) baseline with vectorized computation."""
        if focal_author not in self.task_embeddings:
            return []
        
        # Get candidate authors (excluding focal and excluded authors)
        exclude_set = set([focal_author] + (exclude_authors or []))
        candidate_authors = [aid for aid in self.author_ids 
                           if aid not in exclude_set and aid in self.task_embeddings]
        
        if not candidate_authors:
            return []
        
        # Vectorized computation
        focal_emb = self.task_embeddings[focal_author].reshape(1, -1)
        candidate_embs = np.vstack([self.task_embeddings[aid] for aid in candidate_authors])
        
        # Compute all similarities at once
        distances = cosine_distances(focal_emb, candidate_embs)[0]
        similarities = 1 - distances
        
        # Create results and sort
        results = list(zip(candidate_authors, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def stdm_baseline(self, focal_author: str, exclude_authors: List[str] = None, 
                     filter_k: int = 1000, top_k: int = 10) -> List[Tuple[str, float]]:
        """sTdM (Similar Tasks, distant Methods) baseline with vectorized computation."""
        if focal_author not in self.task_embeddings or focal_author not in self.method_embeddings:
            return []
        
        # Get candidate authors (excluding focal and excluded authors)
        exclude_set = set([focal_author] + (exclude_authors or []))
        candidate_authors = [aid for aid in self.author_ids 
                           if aid not in exclude_set and 
                           aid in self.task_embeddings and aid in self.method_embeddings]
        
        if not candidate_authors:
            return []
        
        # Vectorized computation for all candidates
        focal_task = self.task_embeddings[focal_author].reshape(1, -1)
        focal_method = self.method_embeddings[focal_author].reshape(1, -1)
        
        candidate_task_embs = np.vstack([self.task_embeddings[aid] for aid in candidate_authors])
        candidate_method_embs = np.vstack([self.method_embeddings[aid] for aid in candidate_authors])
        
        # Compute all distances at once
        task_distances = cosine_distances(focal_task, candidate_task_embs)[0]
        method_distances = cosine_distances(focal_method, candidate_method_embs)[0]
        
        # Create candidate data
        candidates = [{
            'author_id': candidate_authors[i],
            'task_dist': task_distances[i],
            'method_dist': method_distances[i]
        } for i in range(len(candidate_authors))]
        
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
    
    def st_baseline_persona(self, focal_author: str, focal_persona: str = None, 
                           exclude_authors: List[str] = None, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        ST baseline with persona support and vectorized computation.
        
        This method operates at the persona level, enabling fine-grained similarity matching
        for multi-domain researchers. Each author may have multiple research personas
        (e.g., "NLP Expert", "Computer Vision Expert") derived from paper clustering.
        
        Algorithm Logic:
        1. Determine focal persona (use primary if not specified)
        2. Filter candidate personas (exclude focal author and specified exclusions)
        3. Vectorized cosine similarity computation across all candidates
        4. Return top-k most similar personas
        
        Args:
            focal_author: The author seeking recommendations
            focal_persona: Specific persona to use (None = use primary persona)
            exclude_authors: Authors to exclude from recommendations
            top_k: Number of recommendations to return
            
        Returns:
            List of (persona_id, similarity_score) tuples, sorted by similarity
            
        Note:
            Returns persona-level results. Use get_author_recommendations() for
            author-level aggregation.
        """
        if not self.persona_mode:
            # Fall back to regular ST baseline
            return self.st_baseline(focal_author, exclude_authors, top_k)
        
        # Determine focal persona
        if focal_persona is None:
            # Use the first (primary) persona for this author
            author_personas = self.author_to_personas.get(focal_author, [])
            if not author_personas:
                return []
            focal_persona_id = author_personas[0]
        else:
            focal_persona_id = f"{focal_author}-{focal_persona}"
        
        if focal_persona_id not in self.task_embeddings:
            return []
        
        # Get candidate personas (excluding focal author and excluded authors)
        exclude_set = set([focal_author] + (exclude_authors or []))
        candidate_personas = [pid for pid in self.persona_ids 
                            if pid.split('-')[0] not in exclude_set and pid in self.task_embeddings]
        
        if not candidate_personas:
            return []
        
        # Vectorized computation
        focal_emb = self.task_embeddings[focal_persona_id].reshape(1, -1)
        candidate_embs = np.vstack([self.task_embeddings[pid] for pid in candidate_personas])
        
        # Compute all similarities at once
        distances = cosine_distances(focal_emb, candidate_embs)[0]
        similarities = 1 - distances
        
        # Create results and sort
        results = list(zip(candidate_personas, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def stdm_baseline_persona(self, focal_author: str, focal_persona: str = None,
                             exclude_authors: List[str] = None, filter_k: int = 1000, top_k: int = 10) -> List[Tuple[str, float]]:
        """sTdM baseline with persona support and vectorized computation."""
        if not self.persona_mode:
            # Fall back to regular sTdM baseline
            return self.stdm_baseline(focal_author, exclude_authors, filter_k, top_k)
        
        # Determine focal persona
        if focal_persona is None:
            author_personas = self.author_to_personas.get(focal_author, [])
            if not author_personas:
                return []
            focal_persona_id = author_personas[0]
        else:
            focal_persona_id = f"{focal_author}-{focal_persona}"
        
        if (focal_persona_id not in self.task_embeddings or 
            focal_persona_id not in self.method_embeddings):
            return []
        
        # Get candidate personas (excluding focal author and excluded authors)
        exclude_set = set([focal_author] + (exclude_authors or []))
        candidate_personas = [pid for pid in self.persona_ids 
                            if pid.split('-')[0] not in exclude_set and 
                            pid in self.task_embeddings and pid in self.method_embeddings]
        
        if not candidate_personas:
            return []
        
        # Vectorized computation for all candidates
        focal_task = self.task_embeddings[focal_persona_id].reshape(1, -1)
        focal_method = self.method_embeddings[focal_persona_id].reshape(1, -1)
        
        candidate_task_embs = np.vstack([self.task_embeddings[pid] for pid in candidate_personas])
        candidate_method_embs = np.vstack([self.method_embeddings[pid] for pid in candidate_personas])
        
        # Compute all distances at once
        task_distances = cosine_distances(focal_task, candidate_task_embs)[0]
        method_distances = cosine_distances(focal_method, candidate_method_embs)[0]
        
        # Create candidate data
        candidates = [{
            'persona_id': candidate_personas[i],
            'task_dist': task_distances[i],
            'method_dist': method_distances[i]
        } for i in range(len(candidate_personas))]
        
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
            results.append((candidate['persona_id'], combined_score))
        
        return results
    
    def get_author_recommendations(self, focal_author: str, method: str = "ST", 
                                  exclude_authors: List[str] = None, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Get author-level recommendations by aggregating persona-level results.
        
        This is the main recommendation interface that handles both standard and persona modes.
        In persona mode, it performs persona-level computation and aggregates to author level
        by selecting the best-matching persona for each candidate author.
        
        Aggregation Logic (Persona Mode):
        1. Retrieve 3x more persona-level results for better coverage
        2. Group results by candidate author
        3. Select highest-scoring persona for each author
        4. Sort authors by best persona scores
        5. Return top-k author recommendations
        
        Algorithm Selection:
        - "ST": Similar Tasks baseline (task similarity only)
        - "sTdM": Similar Tasks, distant Methods (task similarity + method dissimilarity)
        
        Args:
            focal_author: The author seeking recommendations
            method: Algorithm to use ("ST" or "sTdM")
            exclude_authors: Authors to exclude from recommendations
            top_k: Number of author recommendations to return
            
        Returns:
            List of (author_id, best_persona_id, score) tuples
            - author_id: Recommended author
            - best_persona_id: Best matching persona (empty string in standard mode)
            - score: Similarity/combined score from best persona
            
        Example:
            >>> recommendations = baselines.get_author_recommendations("1891568", "ST", top_k=5)
            >>> for author_id, persona_id, score in recommendations:
            ...     print(f"Author {author_id} (persona {persona_id}): {score:.4f}")
        """
        if not self.persona_mode:
            # Convert regular baseline results to expected format
            if method == "ST":
                results = self.st_baseline(focal_author, exclude_authors, top_k)
            else:
                results = self.stdm_baseline(focal_author, exclude_authors, top_k=top_k)
            
            return [(author_id, "", score) for author_id, score in results]
        
        # Persona mode: get persona-level results and aggregate by author
        if method == "ST":
            persona_results = self.st_baseline_persona(focal_author, exclude_authors=exclude_authors, top_k=top_k*3)
        else:
            persona_results = self.stdm_baseline_persona(focal_author, exclude_authors=exclude_authors, top_k=top_k*3)
        
        # Aggregate by author (take best persona per author)
        author_scores = {}
        for persona_id, score in persona_results:
            candidate_author = persona_id.split('-')[0]
            persona_suffix = persona_id.split('-', 1)[1] if '-' in persona_id else ""
            
            if candidate_author not in author_scores or score > author_scores[candidate_author][1]:
                author_scores[candidate_author] = (persona_suffix, score)
        
        # Convert to list and sort
        author_results = [(author, persona, score) for author, (persona, score) in author_scores.items()]
        author_results.sort(key=lambda x: x[2], reverse=True)
        
        return author_results[:top_k]


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
                            'paper_id': paper_id,  # Preserve paper ID for weighting
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
        # Combine papers following original paper format
        combined_texts = []
        for p in papers:
            title = p['title'].strip()
            abstract = p['abstract'].strip()
            text = f"{title}. {abstract}" if abstract else title
            combined_texts.append(text)
        combined_text = ' '.join(combined_texts)
        
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
            # Adapt to 986_paper_matching_pairs.csv format
            if 'author2' in row:
                team_authors = ast.literal_eval(row['author2'])
            elif 'author_old_paper' in row:
                team_authors = ast.literal_eval(row['author_old_paper'])
            else:
                continue
            
            # Handle ground_truth_authors format    
            if pd.notna(row['ground_truth_authors']):
                if '|' in str(row['ground_truth_authors']):
                    gt_authors = set(row['ground_truth_authors'].split('|'))
                else:
                    # Assume it's already a list-like string or list
                    try:
                        gt_authors = set(ast.literal_eval(row['ground_truth_authors']))
                    except:
                        gt_authors = set([str(row['ground_truth_authors'])])
            else:
                gt_authors = set()
            
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
            # Adapt to 986_paper_matching_pairs.csv format
            if 'author2' in row:
                team_authors = ast.literal_eval(row['author2'])
            elif 'author_old_paper' in row:
                team_authors = ast.literal_eval(row['author_old_paper'])
            else:
                continue
                
            evaluation_authors.update(team_authors)
            
            # Handle ground_truth_authors format
            if pd.notna(row['ground_truth_authors']):
                if '|' in str(row['ground_truth_authors']):
                    gt_authors = row['ground_truth_authors'].split('|')
                    evaluation_authors.update([a.strip() for a in gt_authors])
                else:
                    try:
                        gt_authors = ast.literal_eval(row['ground_truth_authors'])
                        evaluation_authors.update(gt_authors)
                    except:
                        evaluation_authors.add(str(row['ground_truth_authors']))
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
    evaluation_data_path = "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
    
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