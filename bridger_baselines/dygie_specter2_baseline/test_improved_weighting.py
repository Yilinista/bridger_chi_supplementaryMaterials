#!/usr/bin/env python3
"""
Test script for improved weighting implementation following original paper
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Add script path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from embedding_generator import SPECTER2EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_citation_scaler():
    """Test the MinMaxScaler for citation counts"""
    logger.info("Testing citation count MinMaxScaler...")
    
    # Create test data
    author_terms = {
        "author_1": {"task": ["classification", "detection"], "method": ["neural network"]},
        "author_2": {"task": ["generation"], "method": ["transformer", "attention"]}
    }
    
    author_papers = {
        "author_1": [
            {"title": "Paper 1", "abstract": "Abstract 1", "cited_count": 50, "author_sequence": 1, "total_authors": 3},
            {"title": "Paper 2", "abstract": "Abstract 2", "cited_count": 200, "author_sequence": 3, "total_authors": 3}
        ],
        "author_2": [
            {"title": "Paper 3", "abstract": "Abstract 3", "cited_count": 5, "author_sequence": 1, "total_authors": 2},
            {"title": "Paper 4", "abstract": "Abstract 4", "cited_count": 100, "author_sequence": 2, "total_authors": 2}
        ]
    }
    
    # Initialize generator
    generator = SPECTER2EmbeddingGenerator()
    
    # Test citation extraction
    for author_id, papers in author_papers.items():
        for paper in papers:
            cited_count = generator._extract_citation_count(paper)
            logger.info(f"Author {author_id}, Paper: {paper['title']}, Citations: {cited_count}")
    
    # Test position weight calculation
    for author_id, papers in author_papers.items():
        for paper in papers:
            pos_weight = generator._get_author_position_weight_improved(author_id, paper)
            logger.info(f"Author {author_id}, Paper: {paper['title']}, Position weight: {pos_weight}")
    
    logger.info("Citation scaler test completed")


def test_matrix_based_weighting():
    """Test the matrix-based weighting approach"""
    logger.info("Testing matrix-based term weighting...")
    
    # Create test generator
    generator = SPECTER2EmbeddingGenerator()
    
    # Mock citation scaler (simulate fitted scaler)
    from sklearn.preprocessing import MinMaxScaler
    generator.citation_scaler = MinMaxScaler(feature_range=(0.5, 1.0))
    generator.citation_scaler.fit([[0], [50], [100], [200]])  # Mock fit
    
    # Test data
    papers = [
        {"title": "High impact paper", "cited_count": 150, "author_sequence": 1, "total_authors": 2},
        {"title": "Medium impact paper", "cited_count": 30, "author_sequence": 2, "total_authors": 3},
        {"title": "Low impact paper", "cited_count": 5, "author_sequence": 1, "total_authors": 1}
    ]
    
    terms = ["classification", "machine learning", "deep learning"]
    
    # Test weight computation
    weights = generator._compute_term_weights_matrix_based("test_author", papers, terms, "task")
    
    logger.info(f"Computed term weights: {weights}")
    logger.info(f"Weight range: {np.min(weights):.3f} - {np.max(weights):.3f}")
    
    logger.info("Matrix-based weighting test completed")


def test_persona_mode():
    """Test persona mode with weighting"""
    logger.info("Testing persona mode with improved weighting...")
    
    # Mock data for persona mode
    author_personas = {
        "author_123": [
            {
                "persona_id": "A",
                "papers": [
                    {"title": "NLP Paper 1", "abstract": "Natural language processing", 
                     "cited_count": 80, "author_sequence": 1, "total_authors": 2},
                    {"title": "NLP Paper 2", "abstract": "Text classification", 
                     "cited_count": 45, "author_sequence": 2, "total_authors": 3}
                ],
                "paper_count": 2
            },
            {
                "persona_id": "B", 
                "papers": [
                    {"title": "Vision Paper 1", "abstract": "Computer vision", 
                     "cited_count": 120, "author_sequence": 1, "total_authors": 1},
                    {"title": "Vision Paper 2", "abstract": "Image recognition",
                     "cited_count": 30, "author_sequence": 3, "total_authors": 4}
                ],
                "paper_count": 2
            }
        ]
    }
    
    persona_terms = {
        "author_123-A": {"task": ["classification", "language"], "method": ["lstm", "bert"]},
        "author_123-B": {"task": ["recognition", "detection"], "method": ["cnn", "resnet"]}
    }
    
    # Test persona paper reconstruction
    persona_papers = {}
    for author_id, personas in author_personas.items():
        for persona in personas:
            persona_id = f"{author_id}-{persona['persona_id']}"
            if persona_id in persona_terms:
                persona_papers_with_author = []
                for paper in persona["papers"]:
                    paper_with_author = paper.copy()
                    paper_with_author['focal_author_id'] = author_id
                    persona_papers_with_author.append(paper_with_author)
                persona_papers[persona_id] = persona_papers_with_author
    
    logger.info(f"Created persona papers for {len(persona_papers)} personas")
    
    for persona_id, papers in persona_papers.items():
        logger.info(f"Persona {persona_id}: {len(papers)} papers")
        for paper in papers:
            logger.info(f"  - {paper['title']} (citations: {paper['cited_count']})")
    
    logger.info("Persona mode test completed")


def main():
    """Run all tests"""
    logger.info("Starting improved weighting implementation tests...")
    
    try:
        test_citation_scaler()
        print()
        test_matrix_based_weighting() 
        print()
        test_persona_mode()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()