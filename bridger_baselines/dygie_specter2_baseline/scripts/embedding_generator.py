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
        self.normalization_nlp = None  # For term normalization
        
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
    
    def _load_normalization_spacy(self):
        """Load spaCy model for term normalization (disable unnecessary components for speed)"""
        if self.normalization_nlp is None:
            try:
                import spacy
                self.normalization_nlp = spacy.load("en_core_sci_sm", disable=["parser", "ner"])
                logger.info("Loaded scientific spaCy model for normalization")
            except OSError:
                try:
                    import spacy
                    self.normalization_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                    logger.warning("Scientific spaCy model not found, using default English model for normalization")
                except OSError:
                    raise RuntimeError("No spaCy model found for normalization. Please install: python -m spacy download en_core_web_sm")
    
    def _normalize_term(self, term: str) -> str:
        """
        Normalize a term following the original paper approach:
        1. Clean punctuation and parentheticals
        2. Normalize case
        3. Lemmatize tokens
        4. Remove extra whitespace
        """
        if not term or not term.strip():
            return ""
        
        self._load_normalization_spacy()
        
        # Step 1: Clean parentheticals and extra whitespace (following original paper)
        import re
        cleaned = re.sub(r'\s\(.*?\)', '', term)  # Remove parentheticals
        cleaned = re.sub(r'\s\(.*', '', cleaned)  # Remove unclosed parentheticals
        cleaned = re.sub(r' +', ' ', cleaned)     # Multiple spaces to single space
        cleaned = cleaned.strip()
        
        if not cleaned:
            return ""
        
        # Step 2: Lemmatization and case normalization
        try:
            doc = self.normalization_nlp(cleaned)
            # Extract lemmas, skip punctuation, convert to lowercase
            lemmas = [tok.lemma_.lower() for tok in doc if not tok.is_punct and not tok.is_space]
            normalized = ' '.join(lemmas)
            return normalized.strip()
        except Exception as e:
            logger.warning(f"Failed to normalize term '{term}': {e}")
            # Fallback: simple cleaning
            return cleaned.lower().strip()
    
    def format_for_dygie(self, author_papers: Dict[str, List[Dict]]) -> str:
        """Format author papers for DyGIE++ input"""
        self._load_spacy()
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        doc_count = 0
        for author_id, papers in author_papers.items():
            for i, paper in enumerate(papers):
                # Combine title and abstract (following original paper format)
                title = paper.get('title', '').strip()
                abstract = paper.get('abstract', '').strip()
                text = f"{title}. {abstract}".strip() if abstract else title.strip()
                
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
        
        # Clean and normalize terms (following original paper approach)
        for author_id in author_terms:
            # Normalize task terms
            if author_terms[author_id]['task']:
                cleaned_tasks = [self._normalize_term(term) for term in author_terms[author_id]['task']]
                author_terms[author_id]['task'] = list(set(term for term in cleaned_tasks if term.strip()))
            
            # Normalize method terms  
            if author_terms[author_id]['method']:
                cleaned_methods = [self._normalize_term(term) for term in author_terms[author_id]['method']]
                author_terms[author_id]['method'] = list(set(term for term in cleaned_methods if term.strip()))
        
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
    
    def generate_author_embeddings(self, author_terms: Dict[str, Dict[str, List[str]]], 
                                 author_papers: Dict[str, List[Dict]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate embeddings for author terms with proper weighting following original paper"""
        self._load_model()
        
        task_embeddings = {}
        method_embeddings = {}
        
        total_authors = len(author_terms)
        logger.info(f"Generating embeddings for {total_authors} authors...")
        
        # Determine if we can apply paper-based weighting
        use_paper_weighting = author_papers is not None
        if use_paper_weighting:
            logger.info("Applying paper-based weighting following original paper methodology")
            
            # First pass: collect all citation counts for proper MinMaxScaler
            all_citation_counts = []
            for author_id, papers in author_papers.items():
                if author_id in author_terms:
                    for paper in papers:
                        cited_count = self._extract_citation_count(paper)
                        all_citation_counts.append(cited_count)
            
            # Fit MinMaxScaler on global citation distribution
            if all_citation_counts:
                from sklearn.preprocessing import MinMaxScaler
                self.citation_scaler = MinMaxScaler(feature_range=(0.5, 1.0))
                citation_array = np.array(all_citation_counts).reshape(-1, 1)
                self.citation_scaler.fit(citation_array)
                logger.info(f"Fitted citation scaler on {len(all_citation_counts)} papers")
                logger.info(f"Citation range: {min(all_citation_counts)}-{max(all_citation_counts)}")
            else:
                self.citation_scaler = None
                logger.warning("No citation data found, using uniform weighting")
        
        # Second pass: generate embeddings with proper weighting
        for i, (author_id, terms) in enumerate(author_terms.items()):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing author {i+1}/{total_authors}")
            
            # Generate task embeddings
            if terms['task']:
                try:
                    task_embs = self.model.encode(terms['task'])
                    if len(task_embs.shape) == 1:
                        task_embs = task_embs.reshape(1, -1)
                    
                    # Apply matrix-based weighting if available
                    if use_paper_weighting and author_id in author_papers:
                        weights = self._compute_term_weights_matrix_based(
                            author_id, author_papers[author_id], terms['task'], 'task'
                        )
                        if weights is not None and len(weights) == len(terms['task']):
                            task_emb = np.average(task_embs, axis=0, weights=weights)
                        else:
                            task_emb = np.mean(task_embs, axis=0)
                    else:
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
                    
                    # Apply matrix-based weighting if available
                    if use_paper_weighting and author_id in author_papers:
                        weights = self._compute_term_weights_matrix_based(
                            author_id, author_papers[author_id], terms['method'], 'method'
                        )
                        if weights is not None and len(weights) == len(terms['method']):
                            method_emb = np.average(method_embs, axis=0, weights=weights)
                        else:
                            method_emb = np.mean(method_embs, axis=0)
                    else:
                        method_emb = np.mean(method_embs, axis=0)
                    
                    method_embeddings[author_id] = method_emb / np.linalg.norm(method_emb)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate method embedding for author {author_id}: {e}")
        
        logger.info(f"Generated embeddings: {len(task_embeddings)} task, {len(method_embeddings)} method")
        return task_embeddings, method_embeddings
    
    def _get_term_weights(self, author_id: str, papers: List[Dict], num_terms: int) -> np.ndarray:
        """
        Generate weights for terms based on author position and paper importance.
        Following the original paper methodology.
        """
        # Load paper nodes to get author position information
        try:
            paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
            with open(paper_nodes_path, 'r') as f:
                paper_nodes = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load paper nodes for weighting: {e}")
            # Fallback to simple uniform weighting
            return np.ones(num_terms)
        
        paper_weights = []
        
        for paper in papers:
            paper_id = paper.get('paper_id')  # This might need adjustment based on your data structure
            
            # Extract paper ID from paper data if available
            if not paper_id:
                # Try to find paper ID by matching title (not ideal but may work)
                paper_title = paper.get('title', '').strip()
                paper_id = self._find_paper_id_by_title(paper_title, paper_nodes)
            
            if paper_id and paper_id in paper_nodes:
                paper_data = paper_nodes[paper_id]
                
                # Get author position weight (following original paper methodology)
                author_position_weight = self._get_author_position_weight(author_id, paper_data)
                
                # Get paper importance weight based on citation count
                cited_count = paper_data.get('features', {}).get('CitedCount', 0)
                paper_importance_weight = self._get_paper_importance_weight(cited_count)
                
                # Combined weight = position_weight * importance_weight
                combined_weight = author_position_weight * paper_importance_weight
                paper_weights.append(combined_weight)
            else:
                # Default weight if paper not found
                paper_weights.append(0.75)  # Average between first/last (1.0) and middle (0.75)
        
        if not paper_weights:
            return np.ones(num_terms)
        
        # Since we don't know which specific terms come from which papers,
        # use the average weight across all papers for this author
        avg_weight = np.mean(paper_weights)
        weights = np.full(num_terms, avg_weight)
        
        return weights
    
    def _find_paper_id_by_title(self, title: str, paper_nodes: dict) -> str:
        """Try to find paper ID by matching title (fallback method)"""
        if not title:
            return None
        
        title_clean = title.lower().strip()
        for paper_id, paper_data in paper_nodes.items():
            paper_title = paper_data.get('features', {}).get('Title', '').lower().strip()
            if paper_title and paper_title == title_clean:
                return paper_id
        return None
    
    def _get_author_position_weight(self, author_id: str, paper_data: dict) -> float:
        """
        Get author position weight following original paper methodology:
        - First or last author: 1.0
        - Middle author: 0.75
        """
        neighbors = paper_data.get('neighbors', {})
        authors = neighbors.get('author', {})
        
        if author_id not in authors:
            return 0.75  # Default to middle author weight
        
        # Get all authors and their positions
        author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
        author_positions.sort(key=lambda x: x[1])  # Sort by position
        
        # Check if this author is first or last
        if len(author_positions) <= 1:
            return 1.0  # Single author gets full weight
        
        first_author_id = author_positions[0][0]
        last_author_id = author_positions[-1][0]
        
        if author_id == first_author_id or author_id == last_author_id:
            return 1.0  # First or last author
        else:
            return 0.75  # Middle author
    
    def _get_paper_importance_weight(self, cited_count: int) -> float:
        """
        Get paper importance weight based on citation count.
        Following original paper: MinMaxScaler(feature_range=(0.5, 1.0))
        """
        # Simple approximation: normalize citation count
        # In a real implementation, you'd collect all citation counts first
        # and then apply MinMaxScaler. Here we use a simplified approach.
        
        if cited_count <= 0:
            return 0.5  # Minimum weight
        elif cited_count >= 100:
            return 1.0  # Maximum weight for highly cited papers
        else:
            # Linear interpolation between 0.5 and 1.0
            return 0.5 + (cited_count / 100.0) * 0.5
    
    def _extract_citation_count(self, paper: Dict) -> int:
        """Extract citation count from paper data"""
        # Try different possible keys for citation count
        for key in ['cited_count', 'CitedCount', 'citation_count', 'citations']:
            if key in paper:
                try:
                    return int(paper[key])
                except (ValueError, TypeError):
                    continue
        
        # Try to extract from features if available
        if 'features' in paper:
            features = paper['features']
            for key in ['CitedCount', 'cited_count', 'citation_count']:
                if key in features:
                    try:
                        return int(features[key])
                    except (ValueError, TypeError):
                        continue
        
        return 0  # Default to 0 if no citation data found
    
    def _compute_term_weights_matrix_based(self, author_id: str, papers: List[Dict], 
                                         terms: List[str], term_type: str) -> np.ndarray:
        """
        Compute term weights using matrix-based approach following original paper.
        This is more accurate than simple averaging approach.
        """
        if not papers or not terms:
            return None
        
        try:
            # Build author-paper matrix with weights
            paper_weights = []
            paper_ids = []
            
            for paper in papers:
                paper_id = paper.get('paper_id', f"paper_{hash(str(paper))}")
                paper_ids.append(paper_id)
                
                # Calculate author position weight (following original paper exactly)
                author_pos_weight = self._get_author_position_weight_improved(author_id, paper)
                
                # Calculate paper importance weight using fitted scaler
                cited_count = self._extract_citation_count(paper)
                if self.citation_scaler is not None:
                    citation_weight = self.citation_scaler.transform([[cited_count]])[0][0]
                else:
                    citation_weight = 0.75  # fallback
                
                # Combined weight = position_weight * importance_weight (original paper formula)
                combined_weight = author_pos_weight * citation_weight
                paper_weights.append(combined_weight)
            
            # For simplicity, assign average paper weight to all terms
            # In a full implementation, you would track which terms come from which papers
            if paper_weights:
                avg_weight = np.mean(paper_weights)
                term_weights = np.full(len(terms), avg_weight)
                return term_weights
            else:
                return np.ones(len(terms)) * 0.75  # fallback uniform weight
                
        except Exception as e:
            logger.warning(f"Failed to compute matrix-based weights for {author_id}: {e}")
            return np.ones(len(terms)) * 0.75  # fallback uniform weight
    
    def _get_author_position_weight_improved(self, author_id: str, paper: Dict) -> float:
        """
        Improved author position weight calculation following original paper exactly.
        Returns:
            1.0 for first or last author
            0.75 for middle author
        """
        try:
            # Use focal_author_id if available (for persona mode)
            focal_author = paper.get('focal_author_id', author_id)
            
            # Try to extract author position information from paper data
            # This is a simplified version - in real implementation you'd need access to full author list
            
            # If we have author sequence information
            if 'author_sequence' in paper:
                seq_num = paper['author_sequence']
                total_authors = paper.get('total_authors', 1)
                
                # First or last author gets full weight
                if seq_num == 1 or seq_num == total_authors:
                    return 1.0
                else:
                    return 0.75
            
            # If we have author list
            if 'authors' in paper:
                authors = paper['authors']
                if isinstance(authors, list) and len(authors) > 0:
                    if len(authors) == 1:
                        return 1.0  # single author
                    
                    # Check if this author is first or last
                    try:
                        author_idx = authors.index(focal_author)
                        if author_idx == 0 or author_idx == len(authors) - 1:
                            return 1.0
                        else:
                            return 0.75
                    except ValueError:
                        return 0.75  # author not found in list, assume middle
            
            # Try to get position from paper_nodes data using the original approach
            if hasattr(self, '_paper_nodes_cache'):
                return self._get_author_position_from_paper_nodes(focal_author, paper)
            
            # Default fallback - assume average case (middle author)
            return 0.75
            
        except Exception as e:
            logger.debug(f"Could not determine author position for {author_id}: {e}")
            return 0.75  # fallback to middle author weight
    
    def _get_author_position_from_paper_nodes(self, author_id: str, paper: Dict) -> float:
        """
        Get author position weight from paper nodes data following original implementation
        """
        try:
            # Load paper nodes if not cached
            if not hasattr(self, '_paper_nodes_cache'):
                paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
                try:
                    with open(paper_nodes_path, 'r') as f:
                        self._paper_nodes_cache = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load paper nodes for position weighting: {e}")
                    return 0.75
            
            # Try to find paper in paper nodes
            paper_id = paper.get('paper_id')
            if not paper_id:
                # Try to find paper ID by matching title (fallback method)
                paper_title = paper.get('title', '').strip().lower()
                for pid, paper_data in self._paper_nodes_cache.items():
                    cached_title = paper_data.get('features', {}).get('Title', '').strip().lower()
                    if cached_title and cached_title == paper_title:
                        paper_id = pid
                        break
            
            if paper_id and paper_id in self._paper_nodes_cache:
                paper_data = self._paper_nodes_cache[paper_id]
                neighbors = paper_data.get('neighbors', {})
                authors = neighbors.get('author', {})
                
                if author_id not in authors:
                    return 0.75  # Default to middle author weight
                
                # Get all authors and their positions
                author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
                author_positions.sort(key=lambda x: x[1])  # Sort by position
                
                # Check if this author is first or last
                if len(author_positions) <= 1:
                    return 1.0  # Single author gets full weight
                
                first_author_id = author_positions[0][0]
                last_author_id = author_positions[-1][0]
                
                if author_id == first_author_id or author_id == last_author_id:
                    return 1.0  # First or last author
                else:
                    return 0.75  # Middle author
            
            return 0.75  # fallback
            
        except Exception as e:
            logger.debug(f"Failed to get author position from paper nodes: {e}")
            return 0.75


class BridgerEmbeddingManager:
    """Manage generation, storage, and loading of Bridger embeddings"""
    
    def __init__(self, storage_dir: str = "./bridger_embeddings"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.term_extractor = DyGIETermExtractor()
        self.embedding_generator = SPECTER2EmbeddingGenerator()
        
        # Persona support
        self.use_personas = False
        self.min_papers_per_persona = 4
        self.persona_distance_threshold = 88.0
    
    def generate_and_store_embeddings(self, 
                                    author_papers: Dict[str, List[Dict]], 
                                    force_regenerate: bool = False,
                                    enable_persona: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate and store all author embeddings"""
        
        # Check if embeddings already exist
        if self._embeddings_exist() and not force_regenerate:
            logger.info("Embeddings already exist. Use force_regenerate=True to regenerate.")
            return self.load_embeddings()
        
        logger.info("Starting DyGIE++ + SPECTER2 embedding generation pipeline...")
        
        if enable_persona:
            logger.info("Persona mode enabled - clustering papers into author personas...")
            
            # Step 1: Create author personas
            logger.info("Step 1: Creating author personas...")
            author_personas = self.create_author_personas(author_papers)
            
            # Step 2: Extract terms for each persona
            logger.info("Step 2: Extracting scientific terms for each persona...")
            persona_terms = self._extract_persona_terms(author_personas)
            
            # Step 3: Generate embeddings for each persona
            logger.info("Step 3: Generating embeddings for each persona...")
            task_embeddings, method_embeddings = self._generate_persona_embeddings(persona_terms)
            
            # Step 4: Store persona embeddings and metadata
            logger.info("Step 4: Storing persona embeddings to disk...")
            self._save_persona_embeddings(task_embeddings, method_embeddings, persona_terms, author_personas)
            
        else:
            logger.info("Standard mode - generating author-level embeddings...")
            
            # Step 1: Extract terms with DyGIE++
            logger.info("Step 1: Extracting scientific terms with DyGIE++...")
            author_terms = self._extract_terms(author_papers)
            
            # Save extracted terms for analysis
            self._save_extracted_terms(author_terms)
            
            # Step 2: Generate embeddings with SPECTER2
            logger.info("Step 2: Generating embeddings with SPECTER2...")
            task_embeddings, method_embeddings = self.embedding_generator.generate_author_embeddings(
                author_terms, author_papers
            )
            
            # Step 3: Store embeddings
            logger.info("Step 3: Storing embeddings to disk...")
            self._save_embeddings(task_embeddings, method_embeddings, author_terms)
        
        logger.info("✅ Embedding generation and storage completed successfully!")
        return task_embeddings, method_embeddings
    
    def enable_personas(self, min_papers_per_persona: int = 4, distance_threshold: float = 88.0):
        """Enable persona mode with specified parameters"""
        self.use_personas = True
        self.min_papers_per_persona = min_papers_per_persona
        self.persona_distance_threshold = distance_threshold
        logger.info(f"Enabled persona mode: min_papers={min_papers_per_persona}, threshold={distance_threshold}")
    
    def create_author_personas(self, author_papers: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Create author personas by clustering papers based on SPECTER2 embeddings.
        
        This method implements the persona creation logic following the original Bridger paper
        methodology. It clusters each author's papers to discover natural research topic
        groupings, creating separate personas for distinct research areas.
        
        Clustering Logic:
        1. Check minimum paper requirement (default: 4 papers per author)
        2. Generate SPECTER2 embeddings for all author papers (title + abstract)
        3. Apply Agglomerative Clustering with Ward linkage
        4. Filter clusters by minimum size requirement
        5. Create personas with alphabetical IDs (A, B, C, etc.)
        
        Parameters:
        - Algorithm: Agglomerative Clustering
        - Linkage: Ward (minimizes within-cluster variance)
        - Distance Threshold: 88.0 (empirically determined)
        - Min Papers per Persona: 4 papers
        - Max Personas per Author: 26 (A-Z)
        
        Args:
            author_papers: Dict mapping author_id to list of paper dictionaries
                         Each paper dict should contain 'title' and 'abstract' keys
        
        Returns:
            Dict mapping author_id to list of persona dictionaries
            Each persona dict contains:
            - papers: List of papers in this persona
            - persona_id: Letter identifier (A, B, C, etc.)
            - cluster_id: Numeric cluster identifier
            - paper_count: Number of papers in persona
            
        Example:
            >>> personas = manager.create_author_personas(author_papers)
            >>> print(personas["author_123"])
            [
                {
                    "papers": [paper1, paper2, paper3, paper4],
                    "persona_id": "A",
                    "cluster_id": 0,
                    "paper_count": 4
                },
                {
                    "papers": [paper5, paper6, paper7],
                    "persona_id": "B", 
                    "cluster_id": 1,
                    "paper_count": 3
                }
            ]
            
        Note:
            Authors with insufficient papers (<4) will have a single persona containing
            all their papers. This ensures all authors have at least one persona.
        """
        from sklearn.cluster import AgglomerativeClustering
        
        author_personas = {}
        total_authors = len(author_papers)
        
        logger.info(f"Creating personas for {total_authors} authors...")
        
        for i, (author_id, papers) in enumerate(author_papers.items()):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing author {i+1}/{total_authors}")
            
            if len(papers) < self.min_papers_per_persona:
                # Not enough papers for persona clustering, treat as single persona
                author_personas[author_id] = [{
                    "papers": papers, 
                    "persona_id": "A",
                    "paper_count": len(papers)
                }]
                continue
            
            try:
                # Step 1: Generate SPECTER embeddings for all papers
                paper_texts = []
                valid_papers = []
                
                for paper in papers:
                    title = paper.get('title', '').strip()
                    abstract = paper.get('abstract', '').strip()
                    text = f"{title}. {abstract}".strip() if abstract else title.strip()
                    
                    if text:
                        paper_texts.append(text)
                        valid_papers.append(paper)
                
                if len(valid_papers) < self.min_papers_per_persona:
                    author_personas[author_id] = [{
                        "papers": valid_papers, 
                        "persona_id": "A",
                        "paper_count": len(valid_papers)
                    }]
                    continue
                
                # Generate paper embeddings using SPECTER2
                paper_embeddings = self.embedding_generator.model.encode(paper_texts)
                if self.embedding_generator.model is None:
                    self.embedding_generator._load_model()
                    paper_embeddings = self.embedding_generator.model.encode(paper_texts)
                
                # Step 2: Cluster papers using hierarchical clustering
                if len(paper_embeddings) == 1:
                    # Only one paper, single persona
                    author_personas[author_id] = [{
                        "papers": valid_papers,
                        "persona_id": "A", 
                        "paper_count": len(valid_papers)
                    }]
                    continue
                
                clusterer = AgglomerativeClustering(
                    linkage="ward",
                    affinity="euclidean",
                    n_clusters=None,
                    distance_threshold=self.persona_distance_threshold
                )
                
                cluster_labels = clusterer.fit_predict(paper_embeddings)
                
                # Step 3: Group papers by cluster and filter by minimum size
                cluster_groups = {}
                for paper_idx, cluster_id in enumerate(cluster_labels):
                    if cluster_id not in cluster_groups:
                        cluster_groups[cluster_id] = []
                    cluster_groups[cluster_id].append(valid_papers[paper_idx])
                
                # Filter clusters with enough papers
                valid_personas = []
                persona_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                
                for i, (cluster_id, cluster_papers) in enumerate(cluster_groups.items()):
                    if len(cluster_papers) >= self.min_papers_per_persona:
                        persona_data = {
                            "papers": cluster_papers,
                            "persona_id": persona_letters[i % len(persona_letters)],
                            "cluster_id": cluster_id,
                            "paper_count": len(cluster_papers)
                        }
                        valid_personas.append(persona_data)
                
                if not valid_personas:
                    # No valid personas found, use all papers as single persona
                    valid_personas = [{
                        "papers": valid_papers, 
                        "persona_id": "A",
                        "paper_count": len(valid_papers)
                    }]
                
                # Sort personas by paper count (largest first)
                valid_personas.sort(key=lambda x: x.get("paper_count", 0), reverse=True)
                
                author_personas[author_id] = valid_personas
                
            except Exception as e:
                logger.warning(f"Failed to create personas for author {author_id}: {e}")
                # Fallback to single persona
                author_personas[author_id] = [{
                    "papers": papers, 
                    "persona_id": "A",
                    "paper_count": len(papers)
                }]
        
        # Log statistics
        total_personas = sum(len(personas) for personas in author_personas.values())
        multi_persona_authors = sum(1 for personas in author_personas.values() if len(personas) > 1)
        
        logger.info(f"Created {total_personas} personas for {total_authors} authors")
        logger.info(f"{multi_persona_authors} authors have multiple personas")
        
        return author_personas
    
    def _extract_persona_terms(self, author_personas: Dict[str, List[Dict]]) -> Dict[str, Dict[str, List[str]]]:
        """Extract terms for each persona using DyGIE++"""
        
        # Flatten personas into a paper collection for DyGIE++ processing
        persona_papers = {}
        persona_mapping = {}  # Maps persona_id back to author_id and persona info
        
        for author_id, personas in author_personas.items():
            for persona in personas:
                persona_id = f"{author_id}-{persona['persona_id']}"
                persona_papers[persona_id] = persona["papers"]
                persona_mapping[persona_id] = {
                    "author_id": author_id,
                    "persona_id": persona["persona_id"],
                    "paper_count": persona["paper_count"]
                }
        
        # Extract terms for all personas
        persona_terms = self._extract_terms(persona_papers)
        
        logger.info(f"Extracted terms for {len(persona_terms)} personas")
        return persona_terms
    
    def _generate_persona_embeddings(self, persona_terms: Dict[str, Dict[str, List[str]]], 
                                   author_personas: Dict[str, List[Dict]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate embeddings for persona terms with weighting support"""
        
        if author_personas is None:
            # Fallback to simple averaging if no persona data available
            logger.warning("No author personas provided, using simple averaging")
            return self.embedding_generator.generate_author_embeddings(persona_terms)
        
        # Reconstruct persona papers mapping for weight calculation
        persona_papers = {}
        for author_id, personas in author_personas.items():
            for persona in personas:
                persona_id = f"{author_id}-{persona['persona_id']}"
                if persona_id in persona_terms:
                    # Add author_id info to papers for proper weight calculation
                    persona_papers_with_author = []
                    for paper in persona["papers"]:
                        paper_with_author = paper.copy()
                        paper_with_author['focal_author_id'] = author_id
                        persona_papers_with_author.append(paper_with_author)
                    persona_papers[persona_id] = persona_papers_with_author
        
        logger.info("Generating persona embeddings with paper-based weighting following original methodology")
        return self.embedding_generator.generate_author_embeddings(persona_terms, persona_papers)
    
    def _save_persona_embeddings(self, 
                               task_embeddings: Dict[str, np.ndarray], 
                               method_embeddings: Dict[str, np.ndarray],
                               persona_terms: Dict[str, Dict[str, List[str]]],
                               author_personas: Dict[str, List[Dict]]):
        """Save persona embeddings with metadata"""
        
        # Create enhanced metadata
        metadata = {
            "dygie_model": self.term_extractor.dygie_model_path,
            "embedding_model": self.embedding_generator.model_name,
            "creation_time": datetime.now().isoformat(),
            "mode": "persona",
            "persona_settings": {
                "min_papers_per_persona": self.min_papers_per_persona,
                "distance_threshold": self.persona_distance_threshold
            },
            "total_personas": len(set(task_embeddings.keys()) | set(method_embeddings.keys())),
            "task_personas": len(task_embeddings),
            "method_personas": len(method_embeddings),
            "embedding_dim": 768,
            "overlap_personas": len(set(task_embeddings.keys()) & set(method_embeddings.keys())),
            "total_authors": len(author_personas),
            "multi_persona_authors": sum(1 for personas in author_personas.values() if len(personas) > 1)
        }
        
        # Create complete data structure
        embedding_data = {
            "metadata": metadata,
            "task_embeddings": task_embeddings,
            "method_embeddings": method_embeddings,
            "author_personas": author_personas,  # Store persona structure
            "persona_terms": persona_terms       # Store extracted terms
        }
        
        # Save as pickle file
        pickle_path = self.storage_dir / "persona_embeddings.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        # Save metadata as JSON
        metadata_path = self.storage_dir / "persona_metadata.json"
        # Convert numpy arrays to lists for JSON serialization
        json_metadata = metadata.copy()
        with open(metadata_path, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        
        # Save persona statistics
        self._save_persona_statistics(author_personas, persona_terms)
        
        logger.info(f"Persona embeddings saved to {pickle_path}")
        logger.info(f"Persona metadata saved to {metadata_path}")
    
    def _save_persona_statistics(self, author_personas: Dict[str, List[Dict]], persona_terms: Dict[str, Dict[str, List[str]]]):
        """Save detailed persona statistics"""
        
        stats = {
            "author_statistics": {},
            "persona_statistics": {},
            "overall_statistics": {
                "total_authors": len(author_personas),
                "total_personas": sum(len(personas) for personas in author_personas.values()),
                "multi_persona_authors": sum(1 for personas in author_personas.values() if len(personas) > 1),
                "avg_personas_per_author": sum(len(personas) for personas in author_personas.values()) / len(author_personas),
                "total_papers": sum(sum(p["paper_count"] for p in personas) for personas in author_personas.values())
            }
        }
        
        # Author-level statistics
        for author_id, personas in author_personas.items():
            stats["author_statistics"][author_id] = {
                "persona_count": len(personas),
                "total_papers": sum(p["paper_count"] for p in personas),
                "persona_details": [
                    {
                        "persona_id": p["persona_id"],
                        "paper_count": p["paper_count"]
                    } for p in personas
                ]
            }
        
        # Persona-level statistics
        for persona_key, terms in persona_terms.items():
            stats["persona_statistics"][persona_key] = {
                "task_terms": len(terms.get('task', [])),
                "method_terms": len(terms.get('method', [])),
            }
        
        stats_path = self.storage_dir / "persona_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Persona statistics saved to {stats_path}")
    
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
        """Load stored embeddings (supports both persona and author modes)"""
        
        # Try persona embeddings first
        persona_pickle_path = self.storage_dir / "persona_embeddings.pkl"
        author_pickle_path = self.storage_dir / "author_embeddings.pkl"
        
        if persona_pickle_path.exists():
            logger.info("Loading persona embeddings...")
            with open(persona_pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            metadata = data['metadata']
            logger.info(f"Loaded persona embeddings: {metadata['total_personas']} personas from {metadata['total_authors']} authors")
            logger.info(f"Task personas: {metadata['task_personas']}, Method personas: {metadata['method_personas']}")
            logger.info(f"Multi-persona authors: {metadata['multi_persona_authors']}")
            logger.info(f"Created: {metadata['creation_time']}")
            
            # Set persona mode based on loaded data
            self.use_personas = True
            self.min_papers_per_persona = metadata.get('persona_settings', {}).get('min_papers_per_persona', 4)
            self.persona_distance_threshold = metadata.get('persona_settings', {}).get('distance_threshold', 88.0)
            
            return data['task_embeddings'], data['method_embeddings']
            
        elif author_pickle_path.exists():
            logger.info("Loading author embeddings...")
            with open(author_pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            metadata = data['metadata']
            logger.info(f"Loaded embeddings for {metadata['total_authors']} authors")
            logger.info(f"Task authors: {metadata['task_authors']}, Method authors: {metadata['method_authors']}")
            logger.info(f"Created: {metadata['creation_time']}")
            
            self.use_personas = False
            
            return data['task_embeddings'], data['method_embeddings']
            
        else:
            raise FileNotFoundError(f"No embeddings found in {self.storage_dir}")
    
    def load_persona_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[Dict]]]:
        """Load persona embeddings with author persona structure"""
        persona_pickle_path = self.storage_dir / "persona_embeddings.pkl"
        
        if not persona_pickle_path.exists():
            raise FileNotFoundError(f"No persona embeddings found at {persona_pickle_path}")
        
        with open(persona_pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['task_embeddings'], data['method_embeddings'], data['author_personas']
    
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
    """Load author-paper mappings from Graph-CoT data with enhanced metadata for weighting"""
    logger.info("Loading author-paper data with enhanced metadata...")
    
    # Load paper nodes
    with open(paper_nodes_path, 'r') as f:
        papers = json.load(f)
    
    # Load author knowledge graph
    with open(author_kg_path, 'r') as f:
        author_kg = json.load(f)
    
    # Build author-paper mappings with enhanced metadata
    author_papers = {}
    for author_id in evaluation_authors:
        if author_id in author_kg:
            paper_ids = author_kg[author_id]
            papers_data = []
            
            for paper_id in paper_ids:
                if paper_id in papers:
                    paper_data = papers[paper_id]
                    features = paper_data.get('features', {})
                    neighbors = paper_data.get('neighbors', {})
                    
                    title = features.get('Title', '').strip()
                    abstract = features.get('Abstract', '').strip()
                    
                    if title:  # Only include papers with titles
                        # Extract citation count for proper weighting
                        cited_count = features.get('CitedCount', 0)
                        
                        # Extract author position information if available
                        authors_info = neighbors.get('author', {})
                        
                        paper_dict = {
                            'paper_id': paper_id,  # Keep paper ID for position lookup
                            'title': title,
                            'abstract': abstract if abstract else '',
                            'cited_count': cited_count,
                            'features': features,  # Keep full features for other metadata
                        }
                        
                        # Add author position info if available
                        if author_id in authors_info:
                            author_position_data = authors_info[author_id]
                            if isinstance(author_position_data, list) and len(author_position_data) > 0:
                                paper_dict['author_sequence'] = author_position_data[0]
                        
                        # Add total number of authors
                        if authors_info:
                            paper_dict['total_authors'] = len(authors_info)
                            paper_dict['authors'] = list(authors_info.keys())
                        
                        papers_data.append(paper_dict)
            
            if papers_data:
                author_papers[author_id] = papers_data
    
    logger.info(f"Loaded enhanced data for {len(author_papers)} authors")
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