#!/usr/bin/env python3
"""
DyGIE++ + SPECTER2 Embedding Generator

This module generates and stores author embeddings using DyGIE++ for term extraction
and SPECTER2 for semantic embeddings, following the original Bridger paper methodology.
"""

import json
import pickle
import numpy as np
import pandas as pd
import tempfile
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DyGIETermExtractor:
    """Extract scientific terms using DyGIE++ model"""
    
    def __init__(self, dygie_model_path: str = "pretrained_models/scierc"):
        self.dygie_model_path = dygie_model_path
        self.nlp = None
        
    def _load_spacy(self):
        """Load spaCy model for text processing"""
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_sci_sm")
                logger.info("Loaded scientific spaCy model")
            except OSError:
                try:
                    import spacy
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.warning("Scientific spaCy model not found, using default English model")
                except OSError:
                    raise RuntimeError("No spaCy model found. Please install: python -m spacy download en_core_web_sm")
    
    def format_for_dygie(self, author_papers: Dict[str, List[Dict]]) -> str:
        """Format author papers for DyGIE++ input"""
        self._load_spacy()
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        doc_count = 0
        for author_id, papers in author_papers.items():
            for i, paper in enumerate(papers):
                # Combine title and abstract
                title = paper.get('title', '').strip()
                abstract = paper.get('abstract', '').strip()
                text = f"{title} {abstract}".strip()
                
                if not text:
                    continue
                    
                doc_id = f"{author_id}_{i}"
                
                # Process with spaCy
                spacy_doc = self.nlp(text)
                sentences = []
                for sent in spacy_doc.sents:
                    tokens = [tok.text for tok in sent if not tok.is_space]
                    if tokens:
                        sentences.append(tokens)
                
                if sentences:
                    doc_data = {
                        "doc_key": doc_id,
                        "sentences": sentences,
                        "dataset": "scierc"
                    }
                    temp_file.write(json.dumps(doc_data) + '\n')
                    doc_count += 1
        
        temp_file.close()
        logger.info(f"Formatted {doc_count} documents for DyGIE++")
        return temp_file.name
    
    def run_dygie_prediction(self, input_file: str) -> str:
        """Run DyGIE++ prediction on formatted data"""
        output_file = input_file.replace('.jsonl', '_predictions.jsonl')
        
        # Check if DyGIE++ model exists
        if not Path(self.dygie_model_path).exists():
            raise FileNotFoundError(
                f"DyGIE++ model not found at {self.dygie_model_path}. "
                f"Please download the SciERC model first."
            )
        
        cmd = [
            'python', 'predict.py',
            self.dygie_model_path,
            input_file,
            '--output-file', output_file,
            '--include-relation'
        ]
        
        logger.info("Running DyGIE++ predictions (this may take a while)...")
        
        try:
            # Change to dygiepp directory if it exists
            cwd = 'dygiepp' if Path('dygiepp').exists() else '.'
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=cwd,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"DyGIE++ stderr: {result.stderr}")
                raise RuntimeError(f"DyGIE++ prediction failed: {result.stderr}")
            
            logger.info("DyGIE++ predictions completed successfully")
            return output_file
            
        except subprocess.TimeoutExpired:
            logger.error("DyGIE++ prediction timed out (2 hours)")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "DyGIE++ predict.py not found. Please ensure DyGIE++ is properly installed "
                "and the working directory is correct."
            )
    
    def parse_dygie_output(self, predictions_file: str) -> Dict[str, Dict[str, List[str]]]:
        """Parse DyGIE++ predictions to extract terms"""
        author_terms = {}
        
        with open(predictions_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line)
                    doc_key = doc['doc_key']
                    author_id = doc_key.split('_')[0]
                    
                    if author_id not in author_terms:
                        author_terms[author_id] = {'task': [], 'method': []}
                    
                    # Parse NER entities
                    sentences = doc['sentences']
                    for sent_idx, entities in enumerate(doc.get('ner', [])):
                        for entity in entities:
                            if len(entity) >= 3:
                                start_idx, end_idx, label = entity[:3]
                                
                                # Extract term text
                                if sent_idx < len(sentences):
                                    sentence_tokens = sentences[sent_idx]
                                    if start_idx < len(sentence_tokens) and end_idx < len(sentence_tokens):
                                        term = ' '.join(sentence_tokens[start_idx:end_idx+1])
                                        
                                        # Classify terms based on DyGIE++ labels
                                        if label in ['Task', 'OtherScientificTerm']:
                                            author_terms[author_id]['task'].append(term)
                                        elif label in ['Method', 'Material', 'Metric']:
                                            author_terms[author_id]['method'].append(term)
                
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {line_num} in predictions file")
                    continue
                except (KeyError, IndexError) as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
        
        # Remove duplicates and clean terms
        for author_id in author_terms:
            author_terms[author_id]['task'] = list(set(author_terms[author_id]['task']))
            author_terms[author_id]['method'] = list(set(author_terms[author_id]['method']))
        
        logger.info(f"Extracted terms for {len(author_terms)} authors")
        return author_terms


class SPECTER2EmbeddingGenerator:
    """Generate embeddings using SPECTER2 model"""
    
    def __init__(self, model_name: str = 'allenai/specter2_base'):
        self.model_name = model_name
        self.model = None
        
    def _load_model(self):
        """Lazy load SPECTER2 model"""
        if self.model is None:
            logger.info(f"Loading SPECTER2 model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("SPECTER2 model loaded successfully")
    
    def generate_author_embeddings(self, author_terms: Dict[str, Dict[str, List[str]]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate embeddings for author terms"""
        self._load_model()
        
        task_embeddings = {}
        method_embeddings = {}
        
        total_authors = len(author_terms)
        logger.info(f"Generating embeddings for {total_authors} authors...")
        
        for i, (author_id, terms) in enumerate(author_terms.items()):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing author {i+1}/{total_authors}")
            
            # Generate task embeddings
            if terms['task']:
                try:
                    task_embs = self.model.encode(terms['task'])
                    if len(task_embs.shape) == 1:
                        task_embs = task_embs.reshape(1, -1)
                    
                    # Average embeddings and normalize
                    task_emb = np.mean(task_embs, axis=0)
                    task_embeddings[author_id] = task_emb / np.linalg.norm(task_emb)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate task embedding for author {author_id}: {e}")
            
            # Generate method embeddings
            if terms['method']:
                try:
                    method_embs = self.model.encode(terms['method'])
                    if len(method_embs.shape) == 1:
                        method_embs = method_embs.reshape(1, -1)
                    
                    # Average embeddings and normalize
                    method_emb = np.mean(method_embs, axis=0)
                    method_embeddings[author_id] = method_emb / np.linalg.norm(method_emb)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate method embedding for author {author_id}: {e}")
        
        logger.info(f"Generated embeddings: {len(task_embeddings)} task, {len(method_embeddings)} method")
        return task_embeddings, method_embeddings


class BridgerEmbeddingManager:
    """Manage generation, storage, and loading of Bridger embeddings"""
    
    def __init__(self, storage_dir: str = "./bridger_embeddings"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.term_extractor = DyGIETermExtractor()
        self.embedding_generator = SPECTER2EmbeddingGenerator()
    
    def generate_and_store_embeddings(self, 
                                    author_papers: Dict[str, List[Dict]], 
                                    force_regenerate: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate and store all author embeddings"""
        
        # Check if embeddings already exist
        if self._embeddings_exist() and not force_regenerate:
            logger.info("Embeddings already exist. Use force_regenerate=True to regenerate.")
            return self.load_embeddings()
        
        logger.info("Starting DyGIE++ + SPECTER2 embedding generation pipeline...")
        
        # Step 1: Extract terms with DyGIE++
        logger.info("Step 1: Extracting scientific terms with DyGIE++...")
        author_terms = self._extract_terms(author_papers)
        
        # Save extracted terms for analysis
        self._save_extracted_terms(author_terms)
        
        # Step 2: Generate embeddings with SPECTER2
        logger.info("Step 2: Generating embeddings with SPECTER2...")
        task_embeddings, method_embeddings = self.embedding_generator.generate_author_embeddings(author_terms)
        
        # Step 3: Store embeddings
        logger.info("Step 3: Storing embeddings to disk...")
        self._save_embeddings(task_embeddings, method_embeddings, author_terms)
        
        logger.info("✅ Embedding generation and storage completed successfully!")
        return task_embeddings, method_embeddings
    
    def _extract_terms(self, author_papers: Dict[str, List[Dict]]) -> Dict[str, Dict[str, List[str]]]:
        """Extract terms using DyGIE++"""
        # Format data for DyGIE++
        input_file = self.term_extractor.format_for_dygie(author_papers)
        
        try:
            # Run DyGIE++ prediction
            predictions_file = self.term_extractor.run_dygie_prediction(input_file)
            
            # Parse results
            author_terms = self.term_extractor.parse_dygie_output(predictions_file)
            
            return author_terms
            
        finally:
            # Clean up temporary files
            for file_path in [input_file, predictions_file]:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    logger.debug(f"Cleaned up temporary file: {file_path}")
    
    def _save_embeddings(self, 
                        task_embeddings: Dict[str, np.ndarray], 
                        method_embeddings: Dict[str, np.ndarray],
                        author_terms: Dict[str, Dict[str, List[str]]]):
        """Save embeddings to disk"""
        
        # Create metadata
        metadata = {
            "dygie_model": self.term_extractor.dygie_model_path,
            "embedding_model": self.embedding_generator.model_name,
            "creation_time": datetime.now().isoformat(),
            "total_authors": len(set(task_embeddings.keys()) | set(method_embeddings.keys())),
            "task_authors": len(task_embeddings),
            "method_authors": len(method_embeddings),
            "embedding_dim": 768,
            "overlap_authors": len(set(task_embeddings.keys()) & set(method_embeddings.keys()))
        }
        
        # Create complete data structure
        embedding_data = {
            "metadata": metadata,
            "task_embeddings": task_embeddings,
            "method_embeddings": method_embeddings
        }
        
        # Save as pickle file (recommended for preserving numpy arrays)
        pickle_path = self.storage_dir / "author_embeddings.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        # Save metadata as JSON for easy inspection
        metadata_path = self.storage_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Optional: Save as compressed numpy arrays
        self._save_as_npz(task_embeddings, method_embeddings)
        
        logger.info(f"Embeddings saved to {pickle_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def _save_extracted_terms(self, author_terms: Dict[str, Dict[str, List[str]]]):
        """Save extracted terms for analysis"""
        terms_path = self.storage_dir / "extracted_terms.json"
        
        # Calculate statistics
        stats = {
            "total_authors": len(author_terms),
            "authors_with_tasks": sum(1 for terms in author_terms.values() if terms['task']),
            "authors_with_methods": sum(1 for terms in author_terms.values() if terms['method']),
            "total_task_terms": sum(len(terms['task']) for terms in author_terms.values()),
            "total_method_terms": sum(len(terms['method']) for terms in author_terms.values()),
            "unique_task_terms": len(set(term for terms in author_terms.values() for term in terms['task'])),
            "unique_method_terms": len(set(term for terms in author_terms.values() for term in terms['method']))
        }
        
        # Save terms with statistics
        terms_data = {
            "statistics": stats,
            "author_terms": author_terms
        }
        
        with open(terms_path, 'w') as f:
            json.dump(terms_data, f, indent=2)
        
        logger.info(f"Extracted terms saved to {terms_path}")
        logger.info(f"Term extraction statistics: {stats}")
    
    def _save_as_npz(self, task_embeddings: Dict[str, np.ndarray], method_embeddings: Dict[str, np.ndarray]):
        """Save embeddings as compressed numpy arrays"""
        
        # Save task embeddings
        if task_embeddings:
            task_ids = list(task_embeddings.keys())
            task_vectors = np.array([task_embeddings[aid] for aid in task_ids])
            
            np.savez_compressed(
                self.storage_dir / "task_embeddings.npz",
                ids=task_ids,
                vectors=task_vectors
            )
        
        # Save method embeddings
        if method_embeddings:
            method_ids = list(method_embeddings.keys())
            method_vectors = np.array([method_embeddings[aid] for aid in method_ids])
            
            np.savez_compressed(
                self.storage_dir / "method_embeddings.npz",
                ids=method_ids,
                vectors=method_vectors
            )
    
    def load_embeddings(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load stored embeddings"""
        pickle_path = self.storage_dir / "author_embeddings.pkl"
        
        if not pickle_path.exists():
            raise FileNotFoundError(f"No embeddings found at {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data['metadata']
        logger.info(f"Loaded embeddings for {metadata['total_authors']} authors")
        logger.info(f"Task authors: {metadata['task_authors']}, Method authors: {metadata['method_authors']}")
        logger.info(f"Created: {metadata['creation_time']}")
        
        return data['task_embeddings'], data['method_embeddings']
    
    def _embeddings_exist(self) -> bool:
        """Check if embeddings file exists"""
        return (self.storage_dir / "author_embeddings.pkl").exists()
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about stored embeddings"""
        if not self._embeddings_exist():
            return {"status": "No embeddings found"}
        
        # Load metadata
        metadata_path = self.storage_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Fallback: load from pickle file
            task_embs, method_embs = self.load_embeddings()
            metadata = {
                "total_authors": len(set(task_embs.keys()) | set(method_embs.keys())),
                "task_authors": len(task_embs),
                "method_authors": len(method_embs)
            }
        
        # Calculate storage size
        storage_size_mb = sum(f.stat().st_size for f in self.storage_dir.glob("*")) / (1024*1024)
        
        stats = {
            "status": "Embeddings available",
            "storage_size_mb": round(storage_size_mb, 2),
            **metadata
        }
        
        return stats


def load_author_paper_data(paper_nodes_path: str, author_kg_path: str, evaluation_authors: set) -> Dict[str, List[Dict]]:
    """Load author-paper mappings from Graph-CoT data (same as original function)"""
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
                    
                    if title:  # Only include papers with titles
                        papers_data.append({
                            'title': title,
                            'abstract': abstract if abstract else ''
                        })
            
            if papers_data:
                author_papers[author_id] = papers_data
    
    logger.info(f"Loaded data for {len(author_papers)} authors")
    return author_papers


def main():
    """Main function to generate embeddings"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate DyGIE++ + SPECTER2 embeddings for Bridger baseline")
    
    parser.add_argument(
        "--evaluation-data", 
        required=True,
        help="Path to evaluation CSV file"
    )
    parser.add_argument(
        "--paper-nodes",
        default="/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json",
        help="Path to paper nodes JSON file"
    )
    parser.add_argument(
        "--author-kg",
        default="/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json", 
        help="Path to author knowledge graph JSON file"
    )
    parser.add_argument(
        "--storage-dir",
        default="./bridger_embeddings",
        help="Directory to store embeddings"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration even if embeddings exist"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true", 
        help="Only show embedding statistics, don't generate"
    )
    
    args = parser.parse_args()
    
    # Initialize embedding manager
    embedding_manager = BridgerEmbeddingManager(args.storage_dir)
    
    # Show stats if requested
    if args.stats_only:
        stats = embedding_manager.get_embedding_stats()
        print("Embedding Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Load evaluation authors from CSV
    logger.info(f"Loading evaluation data from {args.evaluation_data}")
    df = pd.read_csv(args.evaluation_data)
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
    
    logger.info(f"Found {len(evaluation_authors)} unique evaluation authors")
    
    # Check if embeddings already exist
    if embedding_manager._embeddings_exist() and not args.force_regenerate:
        logger.info("Embeddings already exist. Use --force-regenerate to regenerate.")
        
        # Load and show stats
        task_embs, method_embs = embedding_manager.load_embeddings()
        
        # Check coverage
        missing_authors = evaluation_authors - set(task_embs.keys()) - set(method_embs.keys())
        if missing_authors:
            logger.warning(f"Missing embeddings for {len(missing_authors)} evaluation authors")
            logger.info("Consider using --force-regenerate to include all authors")
        else:
            logger.info("✅ All evaluation authors have embeddings")
        
        return
    
    # Load author-paper data
    author_papers = load_author_paper_data(args.paper_nodes, args.author_kg, evaluation_authors)
    
    # Generate embeddings
    logger.info("Starting embedding generation...")
    start_time = datetime.now()
    
    task_embeddings, method_embeddings = embedding_manager.generate_and_store_embeddings(
        author_papers, 
        force_regenerate=args.force_regenerate
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Show final statistics
    stats = embedding_manager.get_embedding_stats()
    logger.info("\n" + "="*60)
    logger.info("EMBEDDING GENERATION COMPLETED")
    logger.info("="*60)
    logger.info(f"Duration: {duration}")
    logger.info(f"Task embeddings: {len(task_embeddings)} authors")
    logger.info(f"Method embeddings: {len(method_embeddings)} authors")
    logger.info(f"Storage size: {stats['storage_size_mb']} MB")
    logger.info(f"Storage location: {embedding_manager.storage_dir}")


if __name__ == "__main__":
    main()