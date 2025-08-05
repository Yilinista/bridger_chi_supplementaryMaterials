#!/usr/bin/env python3
"""
Clean embedding generator using spaCy for term extraction and SPECTER2 for embeddings
"""

import os
import json
import pickle
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpaCyTermExtractor:
    """Extract scientific terms using spaCy (simplified wrapper)"""
    
    def __init__(self):
        """Initialize spaCy term extractor"""
        from spacy_term_extractor import SpaCyTermExtractor
        self.spacy_extractor = SpaCyTermExtractor()
        logger.info("Initialized spaCy term extractor")
    
    def extract_terms_from_papers(self, author_papers: Dict[str, List[Dict]]) -> Dict[str, Dict[str, List[str]]]:
        """Extract terms using spaCy"""
        return self.spacy_extractor.extract_terms_from_papers(author_papers)


class SPECTER2EmbeddingGenerator:
    """Generate embeddings using SPECTER2 model"""
    
    def __init__(self, model_name: str = "allenai/specter2_base"):
        self.model_name = model_name
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0.5, 1.0))
        self.scaler_fitted = False
    
    def _load_model(self):
        """Load SPECTER2 model"""
        if self.model is None:
            logger.info(f"Loading SPECTER2 model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("SPECTER2 model loaded successfully")
    
    def fit_citation_scaler(self, citation_counts: List[int]):
        """Fit the citation count scaler"""
        if not self.scaler_fitted and citation_counts:
            citation_array = np.array(citation_counts).reshape(-1, 1)
            self.scaler.fit(citation_array)
            self.scaler_fitted = True
            logger.info(f"Citation scaler fitted on {len(citation_counts)} papers")
    
    def _calculate_paper_importance_weight(self, cited_count: int) -> float:
        """Calculate paper importance weight based on citations"""
        if not self.scaler_fitted:
            # Fallback to simple calculation if scaler not fitted
            return min(0.5 + (cited_count / 100) * 0.5, 1.0)
        
        try:
            weight = self.scaler.transform([[cited_count]])[0][0]
            return max(0.5, min(1.0, weight))  # Ensure within bounds
        except Exception:
            return 0.75  # Default weight
    
    def _calculate_author_position_weight(self, authors: List[str], target_author: str) -> float:
        """Calculate author position weight (first/last = 1.0, middle = 0.75)"""
        if not authors or target_author not in authors:
            return 0.75
        
        author_idx = authors.index(target_author)
        if author_idx == 0 or author_idx == len(authors) - 1:
            return 1.0
        else:
            return 0.75
    
    def generate_embeddings(self, terms: List[str]) -> np.ndarray:
        """Generate embeddings for a list of terms"""
        self._load_model()
        
        if not terms:
            return np.zeros((1, 768))  # SPECTER2 dimension
        
        # Create text from terms
        text = ". ".join(terms) if terms else ""
        if not text.strip():
            return np.zeros((1, 768))
        
        try:
            embeddings = self.model.encode([text], convert_to_numpy=True)
            return embeddings[0]  # Return single embedding vector
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")
            return np.zeros((1, 768))


class BridgerEmbeddingManager:
    """Main class for managing Bridger embedding generation"""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.term_extractor = SpaCyTermExtractor()
        self.embedding_generator = SPECTER2EmbeddingGenerator()
        
        logger.info(f"Initialized Bridger embedding manager with storage: {storage_dir}")
    
    def _collect_citation_counts(self, author_papers: Dict[str, List[Dict]]) -> List[int]:
        """Collect all citation counts for scaler fitting"""
        citation_counts = []
        for papers in author_papers.values():
            for paper in papers:
                cited_count = paper.get('cited_count', 0)
                if isinstance(cited_count, (int, float)):
                    citation_counts.append(int(cited_count))
        return citation_counts
    
    def _extract_terms(self, author_papers: Dict[str, List[Dict]]) -> Dict[str, Dict[str, List[str]]]:
        """Extract terms using spaCy"""
        logger.info("Extracting terms with spaCy...")
        return self.term_extractor.extract_terms_from_papers(author_papers)
    
    def _generate_weighted_embeddings(self, 
                                    author_papers: Dict[str, List[Dict]], 
                                    author_terms: Dict[str, Dict[str, List[str]]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate weighted embeddings for all authors"""
        
        # Fit citation scaler
        citation_counts = self._collect_citation_counts(author_papers)
        self.embedding_generator.fit_citation_scaler(citation_counts)
        
        task_embeddings = {}
        method_embeddings = {}
        
        total_authors = len(author_terms)
        processed = 0
        
        for author_id, terms in author_terms.items():
            if processed % 1000 == 0:
                logger.info(f"Processing embeddings: {processed}/{total_authors}")
            
            papers = author_papers.get(author_id, [])
            if not papers:
                processed += 1
                continue
            
            # Collect weighted terms
            weighted_task_terms = []
            weighted_method_terms = []
            
            author_task_terms = terms.get('task', [])
            author_method_terms = terms.get('method', [])
            
            for paper in papers:
                # Get paper metadata
                cited_count = paper.get('cited_count', 0)
                authors = paper.get('authors', [])
                
                # Calculate weights
                position_weight = self.embedding_generator._calculate_author_position_weight(authors, author_id)
                importance_weight = self.embedding_generator._calculate_paper_importance_weight(cited_count)
                combined_weight = position_weight * importance_weight
                
                # Apply weights (simple repetition based on weight)
                weight_factor = max(1, int(combined_weight * 2))  # Scale to 1-2x repetition
                
                weighted_task_terms.extend(author_task_terms * weight_factor)
                weighted_method_terms.extend(author_method_terms * weight_factor)
            
            # Generate embeddings
            if weighted_task_terms:
                task_embeddings[author_id] = self.embedding_generator.generate_embeddings(weighted_task_terms)
            
            if weighted_method_terms:
                method_embeddings[author_id] = self.embedding_generator.generate_embeddings(weighted_method_terms)
            
            processed += 1
        
        logger.info(f"Generated embeddings for {len(task_embeddings)} authors (tasks), {len(method_embeddings)} authors (methods)")
        return task_embeddings, method_embeddings
    
    def generate_embeddings(self, author_papers: Dict[str, List[Dict]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Main method to generate embeddings"""
        logger.info(f"Starting embedding generation for {len(author_papers)} authors")
        
        # Extract terms
        author_terms = self._extract_terms(author_papers)
        
        # Generate embeddings
        task_embeddings, method_embeddings = self._generate_weighted_embeddings(author_papers, author_terms)
        
        # Save results
        self._save_embeddings(task_embeddings, method_embeddings, author_terms)
        
        return task_embeddings, method_embeddings
    
    def _save_embeddings(self, 
                        task_embeddings: Dict[str, np.ndarray], 
                        method_embeddings: Dict[str, np.ndarray],
                        author_terms: Dict[str, Dict[str, List[str]]]):
        """Save embeddings to disk"""
        
        # Save embeddings
        with open(self.storage_dir / "task_embeddings.pkl", 'wb') as f:
            pickle.dump(task_embeddings, f)
        
        with open(self.storage_dir / "method_embeddings.pkl", 'wb') as f:
            pickle.dump(method_embeddings, f)
        
        with open(self.storage_dir / "author_terms.pkl", 'wb') as f:
            pickle.dump(author_terms, f)
        
        # Create metadata
        metadata = {
            "embedding_model": self.embedding_generator.model_name,
            "term_extractor": "spaCy",
            "creation_time": datetime.now().isoformat(),
            "total_authors": len(set(task_embeddings.keys()) | set(method_embeddings.keys())),
            "task_authors": len(task_embeddings),
            "method_authors": len(method_embeddings),
            "total_task_embeddings": sum(1 for emb in task_embeddings.values() if emb is not None),
            "total_method_embeddings": sum(1 for emb in method_embeddings.values() if emb is not None)
        }
        
        with open(self.storage_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Embeddings saved to {self.storage_dir}")
        logger.info(f"Task embeddings: {len(task_embeddings)}, Method embeddings: {len(method_embeddings)}")


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Bridger embeddings using spaCy + SPECTER2")
    parser.add_argument("--storage-dir", default="./embeddings_output", help="Directory to save embeddings")
    args = parser.parse_args()
    
    # Create sample data for testing
    sample_data = {
        "author_123": [
            {
                "title": "Deep Learning for Natural Language Processing",
                "abstract": "We propose transformer-based methods for text classification.",
                "authors": ["author_123", "author_456"],
                "cited_count": 50
            }
        ]
    }
    
    manager = BridgerEmbeddingManager(args.storage_dir)
    task_emb, method_emb = manager.generate_embeddings(sample_data)
    
    print(f"Generated embeddings: {len(task_emb)} task, {len(method_emb)} method")


if __name__ == "__main__":
    main()