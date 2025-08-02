#!/usr/bin/env python3
"""
Simple recommendation interface for Bridger baselines

Usage:
    python recommend.py --author-id <author_id> --method ST --top-k 10
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

# Add path to import modules
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.embedding_generator import BridgerEmbeddingManager
from bridger_baselines import BridgerBaselines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BridgerRecommender:
    """Simple interface for getting Bridger recommendations"""
    
    def __init__(self, embedding_storage_dir: str = "./bridger_embeddings"):
        """Initialize recommender with precomputed embeddings"""
        self.embedding_manager = BridgerEmbeddingManager(embedding_storage_dir)
        
        try:
            # Load precomputed embeddings
            logger.info("Loading precomputed embeddings...")
            task_embeddings, method_embeddings = self.embedding_manager.load_embeddings()
            
            # Check if persona mode
            persona_mode = any('-' in k for k in task_embeddings.keys())
            author_personas = {}
            
            if persona_mode:
                try:
                    import pickle
                    persona_pickle_path = self.embedding_manager.storage_dir / "persona_embeddings.pkl"
                    if persona_pickle_path.exists():
                        with open(persona_pickle_path, 'rb') as f:
                            persona_data = pickle.load(f)
                        if "author_personas" in persona_data:
                            author_personas = persona_data["author_personas"]
                except:
                    pass
            
            # Initialize baselines
            self.baselines = BridgerBaselines(
                task_embeddings, 
                method_embeddings,
                persona_mode=persona_mode,
                author_personas=author_personas
            )
            
            self.available_authors = sorted(list(set(
                [k.split('-')[0] if '-' in k else k for k in task_embeddings.keys()]
            )))
            
            logger.info(f"Loaded embeddings for {len(self.available_authors)} authors")
            logger.info(f"Persona mode: {'enabled' if persona_mode else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            logger.error("Please generate embeddings first using bridger_baselines_improved.py")
            raise
    
    def get_recommendations(self, 
                          author_id: str, 
                          method: str = "ST", 
                          top_k: int = 10,
                          exclude_authors: List[str] = None) -> List[Tuple[str, str, float]]:
        """
        Get recommendations for an author
        
        Args:
            author_id: Target author ID
            method: Recommendation method ("ST" or "sTdM")
            top_k: Number of recommendations
            exclude_authors: Authors to exclude from recommendations
            
        Returns:
            List of (author_id, persona_info, score) tuples
        """
        if author_id not in self.available_authors:
            available_sample = self.available_authors[:5]
            raise ValueError(f"Author {author_id} not found. Available authors (sample): {available_sample}")
        
        logger.info(f"Getting {method} recommendations for author {author_id}")
        
        recommendations = self.baselines.get_author_recommendations(
            focal_author=author_id,
            method=method,
            exclude_authors=exclude_authors,
            top_k=top_k
        )
        
        return recommendations
    
    def print_recommendations(self, recommendations: List[Tuple[str, str, float]], author_id: str, method: str):
        """Pretty print recommendations"""
        print(f"\n{method} Recommendations for Author {author_id}:")
        print("=" * 50)
        
        if not recommendations:
            print("No recommendations found.")
            return
        
        for i, (rec_author, persona_info, score) in enumerate(recommendations, 1):
            persona_str = f" (persona: {persona_info})" if persona_info else ""
            print(f"{i:2d}. {rec_author}{persona_str} - Score: {score:.4f}")
    
    def list_available_authors(self, limit: int = 20) -> List[str]:
        """List available authors"""
        return self.available_authors[:limit]


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Get Bridger baseline recommendations")
    
    parser.add_argument(
        "--author-id",
        required=True,
        help="Author ID to get recommendations for"
    )
    parser.add_argument(
        "--method",
        choices=["ST", "sTdM"],
        default="ST",
        help="Recommendation method (default: ST)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations (default: 10)"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Author IDs to exclude from recommendations"
    )
    parser.add_argument(
        "--embedding-dir",
        default="./bridger_embeddings",
        help="Directory with precomputed embeddings"
    )
    parser.add_argument(
        "--list-authors",
        action="store_true",
        help="List available authors"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize recommender
        recommender = BridgerRecommender(args.embedding_dir)
        
        if args.list_authors:
            authors = recommender.list_available_authors(50)
            print(f"Available authors (first 50):")
            for i, author in enumerate(authors, 1):
                print(f"{i:2d}. {author}")
            return
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            author_id=args.author_id,
            method=args.method,
            top_k=args.top_k,
            exclude_authors=args.exclude or []
        )
        
        # Print results
        recommender.print_recommendations(recommendations, args.author_id, args.method)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()