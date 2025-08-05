#!/usr/bin/env python3
"""
Enhanced Bridger implementation with Persona support
Following the original paper's approach of clustering papers into personas
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaBridgerEmbeddingManager:
    """Enhanced version with persona clustering support"""
    
    def __init__(self, storage_dir: str = "./bridger_embeddings"):
        self.storage_dir = storage_dir
        # Load SPECTER2 for both paper clustering and term embedding
        self.specter_model = SentenceTransformer('allenai/specter2_base')
        
    def create_author_personas(self, 
                              author_papers: Dict[str, List[Dict]], 
                              min_papers_per_persona: int = 4,
                              distance_threshold: float = 88.0) -> Dict[str, List[Dict]]:
        """
        Create author personas by clustering papers based on SPECTER embeddings
        
        Returns:
            Dict mapping author_id to list of personas, where each persona contains paper subsets
        """
        author_personas = {}
        
        for author_id, papers in author_papers.items():
            if len(papers) < min_papers_per_persona:
                # Not enough papers for persona clustering, treat as single persona
                author_personas[author_id] = [{"papers": papers, "persona_id": "A"}]
                continue
            
            logger.info(f"Creating personas for author {author_id} with {len(papers)} papers...")
            
            # Step 1: Generate SPECTER embeddings for all papers
            paper_texts = []
            for paper in papers:
                title = paper.get('title', '').strip()
                abstract = paper.get('abstract', '').strip()
                text = f"{title}. {abstract}".strip() if abstract else title.strip()
                paper_texts.append(text)
            
            if not paper_texts:
                continue
                
            try:
                # Generate paper embeddings using SPECTER2
                paper_embeddings = self.specter_model.encode(paper_texts)
                
                # Step 2: Cluster papers using hierarchical clustering
                clusterer = AgglomerativeClustering(
                    linkage="ward",
                    affinity="euclidean", 
                    n_clusters=None,
                    distance_threshold=distance_threshold
                )
                
                cluster_labels = clusterer.fit_predict(paper_embeddings)
                
                # Step 3: Group papers by cluster and filter by minimum size
                cluster_groups = {}
                for paper_idx, cluster_id in enumerate(cluster_labels):
                    if cluster_id not in cluster_groups:
                        cluster_groups[cluster_id] = []
                    cluster_groups[cluster_id].append(papers[paper_idx])
                
                # Filter clusters with enough papers
                valid_personas = []
                persona_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                
                for i, (cluster_id, cluster_papers) in enumerate(cluster_groups.items()):
                    if len(cluster_papers) >= min_papers_per_persona:
                        persona_data = {
                            "papers": cluster_papers,
                            "persona_id": persona_letters[i % len(persona_letters)],
                            "cluster_id": cluster_id,
                            "paper_count": len(cluster_papers)
                        }
                        valid_personas.append(persona_data)
                
                if not valid_personas:
                    # No valid personas found, use all papers as single persona
                    valid_personas = [{"papers": papers, "persona_id": "A"}]
                
                # Sort personas by paper count (largest first)
                valid_personas.sort(key=lambda x: x.get("paper_count", 0), reverse=True)
                
                author_personas[author_id] = valid_personas
                logger.info(f"Created {len(valid_personas)} personas for author {author_id}")
                
            except Exception as e:
                logger.warning(f"Failed to create personas for author {author_id}: {e}")
                # Fallback to single persona
                author_personas[author_id] = [{"papers": papers, "persona_id": "A"}]
        
        return author_personas
    
    def generate_persona_embeddings(self, 
                                   author_personas: Dict[str, List[Dict]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate task and method embeddings for each author persona
        
        Returns:
            (task_embeddings, method_embeddings) where keys are "author_id-persona_id"
        """
        task_embeddings = {}
        method_embeddings = {}
        
        total_personas = sum(len(personas) for personas in author_personas.values())
        logger.info(f"Generating embeddings for {total_personas} author personas...")
        
        processed_count = 0
        
        for author_id, personas in author_personas.items():
            for persona in personas:
                persona_id = f"{author_id}-{persona['persona_id']}"
                persona_papers = persona["papers"]
                
                processed_count += 1
                if processed_count % 50 == 0:
                    logger.info(f"Processing persona {processed_count}/{total_personas}")
                
                # Extract terms using DyGIE++ (simplified - you can integrate your DyGIE++ extractor)
                persona_task_terms, persona_method_terms = self._extract_terms_from_papers(persona_papers)
                
                # Generate embeddings using SPECTER2
                if persona_task_terms:
                    try:
                        task_embs = self.specter_model.encode(persona_task_terms)
                        if len(task_embs.shape) == 1:
                            task_embs = task_embs.reshape(1, -1)
                        
                        task_emb = np.mean(task_embs, axis=0)
                        task_embeddings[persona_id] = task_emb / np.linalg.norm(task_emb)
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate task embedding for persona {persona_id}: {e}")
                
                if persona_method_terms:
                    try:
                        method_embs = self.specter_model.encode(persona_method_terms)
                        if len(method_embs.shape) == 1:
                            method_embs = method_embs.reshape(1, -1)
                        
                        method_emb = np.mean(method_embs, axis=0)
                        method_embeddings[persona_id] = method_emb / np.linalg.norm(method_emb)
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate method embedding for persona {persona_id}: {e}")
        
        logger.info(f"Generated embeddings: {len(task_embeddings)} task, {len(method_embeddings)} method")
        return task_embeddings, method_embeddings
    
    def _extract_terms_from_papers(self, papers: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Extract terms using DyGIE++ model (integrated with proper term extractor)
        """
        if not papers:
            return [], []
        
        # Import DyGIE++ extractor from the main module
        try:
            import sys
            from pathlib import Path
            
            # Add scripts directory to Python path
            scripts_dir = Path(__file__).parent / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.append(str(scripts_dir))
            
            from embedding_generator import DyGIETermExtractor
            
            # Initialize term extractor if not already done
            if not hasattr(self, '_term_extractor'):
                dygie_path = Path(self.storage_dir).parent / "dygiepp" / "pretrained_models" / "scierc"
                self._term_extractor = DyGIETermExtractor(str(dygie_path))
            
            # Create temporary author data structure for DyGIE++ processing
            temp_author_id = "temp_persona"
            temp_author_papers = {temp_author_id: papers}
            
            # Format and run DyGIE++ extraction
            input_file = self._term_extractor.format_for_dygie(temp_author_papers)
            predictions_file = self._term_extractor.run_dygie_prediction(input_file)
            author_terms = self._term_extractor.parse_dygie_output(predictions_file)
            
            # Extract terms for our temporary author
            if temp_author_id in author_terms:
                task_terms = author_terms[temp_author_id]['task']
                method_terms = author_terms[temp_author_id]['method']
                return task_terms, method_terms
            else:
                logger.warning(f"No terms extracted for persona papers")
                return [], []
                
        except Exception as e:
            logger.warning(f"Failed to use DyGIE++ extraction, falling back to simple method: {e}")
            # Fallback to simple keyword extraction
            return self._simple_keyword_extraction(papers)
    
    def _simple_keyword_extraction(self, papers: List[Dict]) -> Tuple[List[str], List[str]]:
        """Fallback simple keyword extraction method"""
        task_terms = []
        method_terms = []
        
        # Simple keyword-based extraction (fallback only)
        task_keywords = [
            "classification", "detection", "recognition", "analysis", "prediction",
            "segmentation", "generation", "translation", "parsing", "extraction"
        ]
        
        method_keywords = [
            "neural network", "deep learning", "machine learning", "transformer",
            "cnn", "lstm", "bert", "attention", "reinforcement learning", "svm"
        ]
        
        for paper in papers:
            title = paper.get('title', '').strip()
            abstract = paper.get('abstract', '').strip()
            text = (f"{title}. {abstract}" if abstract else title).lower()
            
            for keyword in task_keywords:
                if keyword in text:
                    task_terms.append(keyword)
            
            for keyword in method_keywords:
                if keyword in text:
                    method_terms.append(keyword)
        
        # Remove duplicates
        task_terms = list(set(task_terms))
        method_terms = list(set(method_terms))
        
        return task_terms, method_terms


class PersonaBridgerBaselines:
    """Enhanced Bridger baselines that work with personas"""
    
    def __init__(self, task_embeddings: Dict[str, np.ndarray], method_embeddings: Dict[str, np.ndarray]):
        self.task_embeddings = task_embeddings
        self.method_embeddings = method_embeddings
        self.persona_ids = sorted(list(set(task_embeddings.keys()) | set(method_embeddings.keys())))
        
        # Create author to personas mapping
        self.author_to_personas = {}
        for persona_id in self.persona_ids:
            author_id = persona_id.split('-')[0]
            if author_id not in self.author_to_personas:
                self.author_to_personas[author_id] = []
            self.author_to_personas[author_id].append(persona_id)
    
    def st_baseline_persona(self, focal_author: str, focal_persona: str = None, 
                           exclude_authors: List[str] = None, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        ST baseline with persona support
        
        Args:
            focal_author: Author ID
            focal_persona: Specific persona ID (if None, use best persona)
            exclude_authors: Authors to exclude
            top_k: Number of recommendations
        """
        if focal_persona is None:
            # Find the best persona for this author (e.g., the one with most papers or highest quality)
            author_personas = self.author_to_personas.get(focal_author, [])
            if not author_personas:
                return []
            focal_persona = author_personas[0]  # Use first persona as default
        
        if focal_persona not in self.task_embeddings:
            return []
        
        focal_emb = self.task_embeddings[focal_persona].reshape(1, -1)
        results = []
        
        for persona_id in self.persona_ids:
            candidate_author = persona_id.split('-')[0]
            
            if (candidate_author == focal_author or 
                (exclude_authors and candidate_author in exclude_authors) or
                persona_id not in self.task_embeddings):
                continue
            
            persona_emb = self.task_embeddings[persona_id].reshape(1, -1)
            
            from sklearn.metrics.pairwise import cosine_distances
            distance = cosine_distances(focal_emb, persona_emb)[0][0]
            similarity = 1 - distance
            
            # Include both author and persona info in results
            results.append((persona_id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_author_recommendations(self, focal_author: str, method: str = "ST", 
                                  exclude_authors: List[str] = None, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Get author-level recommendations by aggregating persona-level results
        
        Returns:
            List of (author_id, best_persona_id, score) tuples
        """
        if method == "ST":
            persona_results = self.st_baseline_persona(focal_author, exclude_authors=exclude_authors, top_k=top_k*3)
        else:
            # Implement sTdM persona version here
            persona_results = []
        
        # Aggregate by author (take best persona per author)
        author_scores = {}
        for persona_id, score in persona_results:
            candidate_author = persona_id.split('-')[0]
            if candidate_author not in author_scores or score > author_scores[candidate_author][1]:
                author_scores[candidate_author] = (persona_id, score)
        
        # Convert to list and sort
        author_results = [(author, persona, score) for author, (persona, score) in author_scores.items()]
        author_results.sort(key=lambda x: x[2], reverse=True)
        
        return author_results[:top_k]


# Usage example
def test_persona_implementation():
    """Test the persona-based implementation"""
    
    # Mock data
    mock_author_papers = {
        "author_001": [
            {"title": "Deep Learning for NLP", "abstract": "We propose transformer models for text classification..."},
            {"title": "BERT for Sentiment Analysis", "abstract": "Using BERT for sentiment analysis tasks..."},
            {"title": "CNN for Image Recognition", "abstract": "Convolutional neural networks for computer vision..."},
            {"title": "Object Detection with YOLO", "abstract": "Real-time object detection using YOLO architecture..."},
        ]
    }
    
    # Create personas
    persona_manager = PersonaBridgerEmbeddingManager()
    author_personas = persona_manager.create_author_personas(mock_author_papers)
    
    print("Created personas:")
    for author_id, personas in author_personas.items():
        print(f"  {author_id}: {len(personas)} personas")
        for persona in personas:
            print(f"    Persona {persona['persona_id']}: {len(persona['papers'])} papers")
    
    # Generate embeddings
    task_embs, method_embs = persona_manager.generate_persona_embeddings(author_personas)
    
    # Test recommendations
    baselines = PersonaBridgerBaselines(task_embs, method_embs)
    recommendations = baselines.get_author_recommendations("author_001")
    
    print("\nRecommendations:")
    for author, persona, score in recommendations:
        print(f"  {author} (persona {persona}): {score:.4f}")


if __name__ == "__main__":
    test_persona_implementation()