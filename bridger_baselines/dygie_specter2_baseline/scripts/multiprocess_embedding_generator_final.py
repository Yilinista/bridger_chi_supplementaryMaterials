#!/usr/bin/env python3
"""
Final multiprocess embedding generator using spaCy + SPECTER2
"""

import json
import pickle
import logging
import multiprocessing as mp
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd

from embedding_generator import BridgerEmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_paper_chunk(args):
    """Load a chunk of papers - runs in subprocess"""
    papers_chunk, author_kg = args
    chunk_author_papers = {}
    
    for paper_id, paper_data in papers_chunk:
        try:
            features = paper_data.get('features', {})
            neighbors = paper_data.get('neighbors', {})
            
            # Check if paper has title (required)
            title = str(features.get('Title', '')).strip()
            if not title or title == 'nan' or len(title) < 5:
                continue
                
            abstract = str(features.get('Abstract', '')).strip()
            if abstract == 'nan':
                abstract = ""
            
            # Get citation count
            cited_count = features.get('Cited', 0)
            if isinstance(cited_count, str):
                try:
                    cited_count = int(cited_count)
                except:
                    cited_count = 0
            
            # Get authors from neighbors
            paper_authors = neighbors.get('author', [])
            if isinstance(paper_authors, str):
                paper_authors = [paper_authors]
            
            paper_info = {
                'title': title,
                'abstract': abstract,
                'authors': paper_authors,
                'cited_count': cited_count
            }
            
            # Assign paper to each author
            for author_id in paper_authors:
                if author_id in author_kg:  # Only process authors in knowledge graph
                    if author_id not in chunk_author_papers:
                        chunk_author_papers[author_id] = []
                    chunk_author_papers[author_id].append(paper_info)
                    
        except Exception as e:
            continue  # Skip problematic papers
    
    return chunk_author_papers


def load_author_paper_data_mp(paper_nodes_path: str, author_kg_path: str, evaluation_authors: set) -> Dict[str, List[Dict]]:
    """Load author-paper data using multiprocessing"""
    logger.info("Loading author-paper data with MULTIPROCESS approach...")
    
    # Load papers
    with open(paper_nodes_path, 'r') as f:
        papers = json.load(f)
    logger.info(f"Loaded {len(papers)} papers from paper nodes")
    
    # Load author knowledge graph
    with open(author_kg_path, 'r') as f:
        author_kg = json.load(f)
    
    # Split papers into chunks for multiprocessing
    paper_items = list(papers.items())
    num_cores = 32  # Use 32 processes
    chunk_size = len(paper_items) // num_cores + 1
    
    logger.info(f"Using {num_cores} processes to process {len(paper_items)} papers")
    
    paper_chunks = []
    for i in range(0, len(paper_items), chunk_size):
        chunk = paper_items[i:i + chunk_size]
        paper_chunks.append((chunk, author_kg))
    
    # Process chunks in parallel
    with mp.Pool(processes=num_cores) as pool:
        chunk_results = pool.map(load_paper_chunk, paper_chunks)
    
    # Merge results
    logger.info("Merging results from all processes...")
    author_papers = {}
    for chunk_result in chunk_results:
        for author_id, papers_list in chunk_result.items():
            if author_id not in author_papers:
                author_papers[author_id] = []
            author_papers[author_id].extend(papers_list)
    
    logger.info(f"Successfully loaded data for {len(author_papers)} authors using MULTIPROCESS approach")
    logger.info(f"Coverage: {len(author_papers) / len(evaluation_authors) * 100:.1f}%")
    
    return author_papers


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final multiprocess embedding generator")
    parser.add_argument("--evaluation-data", required=True)
    parser.add_argument("--paper-nodes", 
                       default="/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json")
    parser.add_argument("--author-kg",
                       default="/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json")
    parser.add_argument("--storage-dir", default="./global_embeddings_final")
    parser.add_argument("--force-regenerate", action="store_true")
    
    args = parser.parse_args()
    
    # Determine authors to process
    if args.evaluation_data == 'dummy_eval.csv':
        logger.info("Processing ALL authors in knowledge graph")
        with open(args.author_kg, 'r') as f:
            author_kg = json.load(f)
        evaluation_authors = set(author_kg.keys())
        logger.info(f"Found {len(evaluation_authors)} total authors in knowledge graph")
    else:
        logger.info(f"Loading evaluation data from {args.evaluation_data}")
        df = pd.read_csv(args.evaluation_data)
        evaluation_authors = set()
        # Parse authors from evaluation data (adjust based on specific format)
    
    # Load author-paper data
    author_papers = load_author_paper_data_mp(
        args.paper_nodes, 
        args.author_kg, 
        evaluation_authors
    )
    
    # Create storage directory
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(exist_ok=True)
    
    # Generate embeddings
    embedding_manager = BridgerEmbeddingManager(str(storage_dir))
    task_embeddings, method_embeddings = embedding_manager.generate_embeddings(author_papers)
    
    logger.info("Embedding generation completed successfully!")
    logger.info(f"Results saved to: {storage_dir}")


if __name__ == "__main__":
    main()