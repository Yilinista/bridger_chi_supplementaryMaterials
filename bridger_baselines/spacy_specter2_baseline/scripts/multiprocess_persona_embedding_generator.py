#!/usr/bin/env python3
"""
Multiprocess Persona-based Embedding Generator for Bridger Algorithm
Combines persona clustering with parallel processing for scalability
"""

import json
import logging
import numpy as np
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import spacy
import torch
from tqdm import tqdm

# Set multiprocessing start method to spawn for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PersonaConfig:
    """Configuration for persona clustering"""
    min_papers_per_persona: int = 4
    distance_threshold: float = 88.0
    min_papers_for_clustering: int = 4

def init_worker():
    """Initialize worker process with models"""
    global nlp, specter_model, term_config, stopwords
    
    # Enable CUDA with better memory management
    import torch
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.15)  # Limit each process to 15% GPU memory
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model successfully")
    
    # Load SPECTER2 model with device allocation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    specter_model = SentenceTransformer('allenai/specter2_base', device=device)
    logger.info(f"Loaded SPECTER2 model on {device}: allenai/specter2_base")
    
    # Load term classification config
    config_dir = Path("/data/jx4237data/fair_teaming_yy/bridger_chi_supplementaryMaterials/bridger_baselines/spacy_specter2_baseline/config")
    
    with open(config_dir / "term_classification.json", 'r') as f:
        term_config = json.load(f)
    
    with open(config_dir / "stopwords.json", 'r') as f:
        stopwords = set(json.load(f))

def create_author_personas(author_id: str, papers: List[Dict], config: PersonaConfig) -> List[Dict]:
    """
    Create personas for an author by clustering their papers
    Returns list of personas with paper subsets
    """
    if len(papers) < config.min_papers_for_clustering:
        # Not enough papers for clustering
        return [{
            "persona_id": f"{author_id}_A",
            "papers": papers,
            "num_papers": len(papers)
        }]
    
    # Generate SPECTER embeddings for papers
    paper_texts = []
    valid_papers = []
    
    for paper in papers:
        title = paper.get('title', '').strip()
        abstract = paper.get('abstract', '').strip()
        
        if title:
            text = f"{title}. {abstract}" if abstract else title
            paper_texts.append(text)
            valid_papers.append(paper)
    
    if len(valid_papers) < config.min_papers_for_clustering:
        return [{
            "persona_id": f"{author_id}_A",
            "papers": papers,
            "num_papers": len(papers)
        }]
    
    # Generate embeddings
    paper_embeddings = specter_model.encode(paper_texts, show_progress_bar=False)
    
    # Cluster papers
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=config.distance_threshold,
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(paper_embeddings)
    
    # Group papers by cluster
    personas = []
    for cluster_id in np.unique(cluster_labels):
        cluster_papers = [p for i, p in enumerate(valid_papers) if cluster_labels[i] == cluster_id]
        
        if len(cluster_papers) >= config.min_papers_per_persona:
            persona_id = f"{author_id}_{chr(65 + len(personas))}"  # A, B, C...
            personas.append({
                "persona_id": persona_id,
                "papers": cluster_papers,
                "num_papers": len(cluster_papers)
            })
    
    # If no valid personas, return all papers as single persona
    if not personas:
        personas.append({
            "persona_id": f"{author_id}_A",
            "papers": papers,
            "num_papers": len(papers)
        })
    
    return personas

def extract_terms_from_papers(papers: List[Dict]) -> Tuple[List[str], List[str]]:
    """Extract task and method terms from papers using spaCy"""
    task_terms = []
    method_terms = []
    
    for paper in papers:
        title = paper.get('title', '').strip()
        abstract = paper.get('abstract', '').strip()
        
        if not title:
            continue
            
        text = f"{title}. {abstract}" if abstract else title
        
        # Process with spaCy
        doc = nlp(text)
        
        # Extract noun phrases and entities
        terms = set()
        
        # Noun phrases
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4:
                terms.add(chunk.text.lower())
        
        # Named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:
                terms.add(ent.text.lower())
        
        # Single nouns and compounds
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                terms.add(token.lemma_.lower())
                
                # Compound terms
                if token.dep_ == 'compound':
                    compound = f"{token.text} {token.head.text}".lower()
                    terms.add(compound)
        
        # Classify terms
        for term in terms:
            term_clean = term.strip()
            if not term_clean or term_clean in stopwords:
                continue
                
            # Check against predefined categories
            if any(task_kw in term_clean for task_kw in term_config.get('task_keywords', [])):
                task_terms.append(term_clean)
            elif any(method_kw in term_clean for method_kw in term_config.get('method_keywords', [])):
                method_terms.append(term_clean)
            else:
                # Default classification based on patterns
                if any(pattern in term_clean for pattern in ['method', 'algorithm', 'model', 'framework', 'approach']):
                    method_terms.append(term_clean)
                else:
                    task_terms.append(term_clean)
    
    return task_terms, method_terms

def process_author_chunk(args):
    """Process a chunk of authors - runs in subprocess"""
    chunk_data, citation_scaler, persona_config = args
    
    # Initialize models in worker
    init_worker()
    
    results = []
    
    for author_id, papers in chunk_data:
        try:
            # Create personas
            personas = create_author_personas(author_id, papers, persona_config)
            
            # Process each persona
            for persona in personas:
                persona_papers = persona['papers']
                
                # Extract terms
                task_terms, method_terms = extract_terms_from_papers(persona_papers)
                
                if not task_terms and not method_terms:
                    continue
                
                # Generate embeddings for unique terms
                unique_tasks = list(set(task_terms))
                unique_methods = list(set(method_terms))
                
                task_embeddings = specter_model.encode(unique_tasks, show_progress_bar=False) if unique_tasks else np.array([])
                method_embeddings = specter_model.encode(unique_methods, show_progress_bar=False) if unique_methods else np.array([])
                
                # Calculate weights for papers
                weights = []
                for paper in persona_papers:
                    # Author position weight
                    author_position = paper.get('author_position', 1)
                    if author_position == 1 or author_position == paper.get('total_authors', 1):
                        position_weight = 1.0
                    else:
                        position_weight = 0.75
                    
                    # Citation weight
                    cited_count = paper.get('cited_count', 0)
                    if citation_scaler and isinstance(cited_count, (int, float)):
                        citation_weight = citation_scaler.transform([[cited_count]])[0][0]
                    else:
                        citation_weight = 0.5
                    
                    weights.append(position_weight * citation_weight)
                
                # Weighted average of embeddings
                if task_embeddings.size > 0:
                    # Create term-level weights (equal weight per unique term)
                    task_weights = [1.0] * len(unique_tasks)
                    task_embedding = np.average(task_embeddings, axis=0, weights=task_weights)
                else:
                    task_embedding = np.zeros(768)
                    
                if method_embeddings.size > 0:
                    # Create term-level weights (equal weight per unique term)  
                    method_weights = [1.0] * len(unique_methods)
                    method_embedding = np.average(method_embeddings, axis=0, weights=method_weights)
                else:
                    method_embedding = np.zeros(768)
                
                # Normalize
                task_norm = np.linalg.norm(task_embedding)
                if task_norm > 0:
                    task_embedding = task_embedding / task_norm
                    
                method_norm = np.linalg.norm(method_embedding)
                if method_norm > 0:
                    method_embedding = method_embedding / method_norm
                
                results.append({
                    'author_id': author_id,
                    'persona_id': persona['persona_id'],
                    'num_papers': persona['num_papers'],
                    'task_embedding': task_embedding,
                    'method_embedding': method_embedding,
                    'num_task_terms': len(unique_tasks),
                    'num_method_terms': len(unique_methods)
                })
                
                # Clear CUDA cache periodically to prevent memory buildup
                import torch
                if torch.cuda.is_available() and len(results) % 10 == 0:
                    torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing author {author_id}: {e}")
            continue
    
    return results

class MultiprocessPersonaEmbeddingGenerator:
    """Manager for multiprocess persona-based embedding generation"""
    
    def __init__(self, output_dir: str, num_processes: int = None, chunk_size: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Limit processes to avoid CUDA memory issues - use fewer processes for CUDA
        self.num_processes = num_processes or min(16, mp.cpu_count())  # Max 16 processes
        self.chunk_size = chunk_size
        self.persona_config = PersonaConfig()
        
        logger.info(f"Initialized MultiprocessPersonaEmbeddingGenerator")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Processes: {self.num_processes}, Chunk size: {self.chunk_size}")
    
    def generate_embeddings(self, author_papers: Dict[str, List[Dict]]) -> Tuple[Dict, Dict]:
        """Generate persona-based embeddings for all authors"""
        logger.info(f"Starting persona-based embedding generation for {len(author_papers)} authors")
        
        # First pass: collect citation counts
        logger.info("Collecting citation counts for scaling...")
        all_citations = []
        for papers in author_papers.values():
            for paper in papers:
                cited_count = paper.get('cited_count', 0)
                if isinstance(cited_count, (int, float)):
                    all_citations.append(int(cited_count))
        
        # Fit citation scaler
        citation_scaler = MinMaxScaler(feature_range=(0.5, 1.0))
        if all_citations:
            citation_scaler.fit(np.array(all_citations).reshape(-1, 1))
            logger.info(f"Citation scaler fitted on {len(all_citations)} values")
        else:
            citation_scaler = None
        
        # Create chunks for parallel processing
        author_items = list(author_papers.items())
        chunks = []
        for i in range(0, len(author_items), self.chunk_size):
            chunk = author_items[i:i + self.chunk_size]
            chunks.append((chunk, citation_scaler, self.persona_config))
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Process chunks in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [executor.submit(process_author_chunk, chunk) for chunk in chunks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error in chunk processing: {e}")
        
        # Organize results by persona
        task_embeddings = {}
        method_embeddings = {}
        persona_metadata = {}
        
        for result in all_results:
            persona_id = result['persona_id']
            task_embeddings[persona_id] = result['task_embedding']
            method_embeddings[persona_id] = result['method_embedding']
            persona_metadata[persona_id] = {
                'author_id': result['author_id'],
                'num_papers': result['num_papers'],
                'num_task_terms': result['num_task_terms'],
                'num_method_terms': result['num_method_terms']
            }
        
        # Save embeddings
        logger.info(f"Saving {len(task_embeddings)} persona embeddings...")
        
        with open(self.output_dir / "task_embeddings_persona.pkl", 'wb') as f:
            pickle.dump(task_embeddings, f)
            
        with open(self.output_dir / "method_embeddings_persona.pkl", 'wb') as f:
            pickle.dump(method_embeddings, f)
            
        with open(self.output_dir / "persona_metadata.json", 'w') as f:
            json.dump(persona_metadata, f, indent=2)
        
        logger.info(f"Embeddings saved to {self.output_dir}")
        
        # Print statistics
        num_authors = len(set(m['author_id'] for m in persona_metadata.values()))
        logger.info(f"Generated embeddings for {num_authors} authors with {len(persona_metadata)} personas")
        
        return task_embeddings, method_embeddings

def main():
    """Main function"""
    # Load data
    data_dir = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data"
    paper_nodes_path = f"{data_dir}/papernodes_remove0/paper_nodes_2024dec.json"
    
    logger.info("Loading author-paper data...")
    
    # Load updated author nodes (contains direct paper mappings)
    updated_author_path = f"{data_dir}/updated_author_nodes_with_papers.json"
    with open(updated_author_path, 'r') as f:
        author_nodes = json.load(f)
    
    # Load paper nodes
    with open(paper_nodes_path, 'r') as f:
        paper_nodes = json.load(f)
    
    logger.info(f"Loaded {len(author_nodes)} authors and {len(paper_nodes)} papers")
    
    # Build author-paper mapping using direct Author Node approach (O(n+k))
    logger.info("Building author-paper mapping from Updated Author Nodes...")
    
    author_papers = {}
    
    # Direct mapping from Updated Author Nodes to Paper Nodes
    for author_id, author_data in tqdm(author_nodes.items(), desc="Processing authors from Updated Nodes"):
        author_papers[author_id] = []
        
        # Get paper IDs from author's neighbors
        neighbors = author_data.get('neighbors', {})
        paper_ids = neighbors.get('papers', [])
        
        # Process each paper for this author
        for paper_id in paper_ids:
            if paper_id in paper_nodes:
                paper_data = paper_nodes[paper_id]
                
                if not isinstance(paper_data, dict):
                    continue
                    
                features = paper_data.get('features', {})
                neighbors = paper_data.get('neighbors', {})
                
                # Extract paper info
                title = str(features.get('Title', '')).strip()
                if not title or title == 'nan':
                    continue
                    
                abstract = str(features.get('Abstract', '')).strip()
                if abstract == 'nan':
                    abstract = ""
                
                cited_count = features.get('CitedCount', 0)
                if isinstance(cited_count, str):
                    try:
                        cited_count = int(cited_count)
                    except:
                        cited_count = 0
                
                # Get actual author count from AuthorNum field
                author_num = features.get('AuthorNum', 0)
                if isinstance(author_num, str):
                    try:
                        author_num = int(author_num)
                    except:
                        author_num = 0
                
                # Get author position from paper's author dict
                authors_dict = neighbors.get('author', {})
                if isinstance(authors_dict, dict) and author_id in authors_dict:
                    author_position = authors_dict[author_id][0]
                    total_authors = author_num if author_num > 0 else len(authors_dict)
                else:
                    # Fallback if position info not available
                    author_position = 1
                    total_authors = author_num if author_num > 0 else 1
                
                author_papers[author_id].append({
                    'paper_id': paper_id,
                    'title': title,
                    'abstract': abstract,
                    'cited_count': cited_count,
                    'author_position': author_position,
                    'total_authors': total_authors
                })
    
    # Filter authors with papers
    author_papers = {k: v for k, v in author_papers.items() if v}
    logger.info(f"Found {len(author_papers)} authors with papers")
    
    # Generate embeddings
    output_dir = "/data/jx4237data/fair_teaming_yy/bridger_chi_supplementaryMaterials/bridger_baselines/spacy_specter2_baseline/persona_embeddings"
    generator = MultiprocessPersonaEmbeddingGenerator(output_dir)
    
    task_embeddings, method_embeddings = generator.generate_embeddings(author_papers)
    
    logger.info("Persona-based embedding generation completed!")

if __name__ == "__main__":
    main()