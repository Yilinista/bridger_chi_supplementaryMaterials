#!/usr/bin/env python3
"""
spaCy-based term extractor with configurable keywords and stopwords
"""

import spacy
import re
import json
import logging
from typing import Dict, List, Set
from pathlib import Path

logger = logging.getLogger(__name__)

class SpaCyTermExtractor:
    """Extract scientific terms using spaCy NER and rule-based patterns"""
    
    def __init__(self, config_dir: str = None):
        self.nlp = None
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self._load_config()
        self._load_spacy()
    
    def _load_spacy(self):
        """Load spaCy model for text processing"""
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_sci_sm")
                logger.info("Loaded scientific spaCy model for term extraction")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Using default English spaCy model for term extraction")
                except OSError:
                    raise RuntimeError("No spaCy model found. Please install: python -m spacy download en_core_web_sm")
    
    def _load_config(self):
        """Load configuration from external files"""
        config_dir = Path(self.config_dir)
        
        # Load term classification config
        term_config_path = config_dir / "term_classification.json"
        try:
            with open(term_config_path, 'r') as f:
                term_config = json.load(f)
            
            self.method_keywords = set(term_config["method_keywords"])
            self.task_keywords = set(term_config["task_keywords"])
            self.weights = term_config["classification_weights"]
            self.scientific_indicators = term_config["scientific_indicators"]
            
            logger.info(f"Loaded term classification config: {len(self.method_keywords)} method keywords, {len(self.task_keywords)} task keywords")
            
        except Exception as e:
            logger.warning(f"Failed to load term classification config: {e}. Using defaults.")
            self._setup_default_patterns()
        
        # Load stopwords config
        stopwords_config_path = config_dir / "stopwords.json"
        try:
            with open(stopwords_config_path, 'r') as f:
                stopwords_config = json.load(f)
            
            # Combine all stopword categories
            self.stopwords = set()
            self.stopwords.update(stopwords_config["english_stopwords"])
            self.stopwords.update(stopwords_config["academic_stopwords"]) 
            self.stopwords.update(stopwords_config["scientific_noise_words"])
            
            logger.info(f"Loaded {len(self.stopwords)} stopwords from config")
            
        except Exception as e:
            logger.warning(f"Failed to load stopwords config: {e}. Using minimal defaults.")
            self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def _setup_default_patterns(self):
        """Fallback method to setup default patterns if config loading fails"""
        # Minimal default keywords
        self.method_keywords = {
            'algorithm', 'method', 'approach', 'technique', 'framework', 'model', 'system',
            'neural', 'network', 'deep', 'learning', 'machine', 'statistical'
        }
        
        self.task_keywords = {
            'task', 'problem', 'application', 'domain', 'field', 'area', 'research',
            'recognition', 'detection', 'identification', 'classification', 'prediction'
        }
        
        # Default weights
        self.weights = {
            "method_weight_in_term": 1.0,
            "method_weight_in_context": 0.5,
            "task_weight_in_term": 1.0,
            "task_weight_in_context": 0.5,
            "tie_breaker_word_threshold": 2
        }
        
        # Default scientific indicators
        self.scientific_indicators = {
            "technical_suffixes": ["-based", "-driven", "-aware", "-free", "tion", "sion", "ness", "ment"],
            "technical_prefixes": ["multi-", "semi-", "auto-", "self-", "cross-", "meta-"],
            "abbreviation_pattern": "\\d|[A-Z]{2,}"
        }
    
    def _normalize_term(self, term: str) -> str:
        """Normalize a term by cleaning and lemmatizing"""
        if not term or not term.strip():
            return ""
        
        # Clean parentheticals and extra whitespace
        cleaned = re.sub(r'\s\(.*?\)', '', term)
        cleaned = re.sub(r'\s\(.*', '', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)
        cleaned = cleaned.strip().lower()
        
        if not cleaned or len(cleaned) < 3:
            return ""
        
        return cleaned
    
    def _classify_term(self, term: str, context: str = "") -> str:
        """Classify term as task or method based on keywords and context using configurable weights"""
        term_lower = term.lower()
        context_lower = context.lower()
        
        # Check for method indicators using configurable weights
        method_score = sum(self.weights["method_weight_in_term"] for kw in self.method_keywords if kw in term_lower)
        method_score += sum(self.weights["method_weight_in_context"] for kw in self.method_keywords if kw in context_lower)
        
        # Check for task indicators using configurable weights
        task_score = sum(self.weights["task_weight_in_term"] for kw in self.task_keywords if kw in term_lower)
        task_score += sum(self.weights["task_weight_in_context"] for kw in self.task_keywords if kw in context_lower)
        
        # Default classification based on scores
        if method_score > task_score:
            return 'method'
        elif task_score > method_score:
            return 'task'
        else:
            # Tie-breaker: longer phrases tend to be tasks, shorter tend to be methods
            threshold = self.weights["tie_breaker_word_threshold"]
            return 'task' if len(term.split()) > threshold else 'method'
    
    def extract_terms_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract terms from text using spaCy NER and patterns"""
        if not text.strip():
            return {'task': [], 'method': []}
        
        doc = self.nlp(text)
        
        task_terms = []
        method_terms = []
        
        # Extract named entities
        for ent in doc.ents:
            entity_text = self._normalize_term(ent.text)
            if not entity_text:
                continue
                
            # Use entity context for classification
            sent_text = ent.sent.text if ent.sent else text
            classification = self._classify_term(entity_text, sent_text)
            
            if classification == 'task':
                task_terms.append(entity_text)
            else:
                method_terms.append(entity_text)
        
        # Extract noun phrases as potential scientific terms
        for chunk in doc.noun_chunks:
            chunk_text = self._normalize_term(chunk.text)
            if not chunk_text or len(chunk_text.split()) > 4:
                continue
                
            # Skip if already found as entity
            if chunk_text in task_terms or chunk_text in method_terms:
                continue
            
            # Filter for scientific-looking terms
            if self._is_scientific_term(chunk_text):
                sent_text = chunk.sent.text if chunk.sent else text
                classification = self._classify_term(chunk_text, sent_text)
                
                if classification == 'task':
                    task_terms.append(chunk_text)
                else:
                    method_terms.append(chunk_text)
        
        return {
            'task': list(set(task_terms)),
            'method': list(set(method_terms))
        }
    
    def _is_scientific_term(self, term: str) -> bool:
        """Check if a term looks like a scientific term using configurable stopwords"""
        # Skip stopwords from configuration
        if term in self.stopwords:
            return False
        
        # Check for scientific indicators using configuration
        scientific_indicators = [
            # Has scientific keywords
            any(kw in term for kw in self.method_keywords | self.task_keywords),
            # Contains technical suffixes from config
            any(term.endswith(suffix) for suffix in self.scientific_indicators["technical_suffixes"]),
            # Contains technical prefixes from config
            any(term.startswith(prefix) for prefix in self.scientific_indicators["technical_prefixes"]),
            # Contains numbers or abbreviations
            bool(re.search(self.scientific_indicators["abbreviation_pattern"], term)),
            # Multiple words with at least one technical word
            len(term.split()) > 1 and any(word in self.method_keywords | self.task_keywords for word in term.split())
        ]
        
        return any(scientific_indicators)
    
    def extract_terms_from_papers(self, author_papers: Dict[str, List[Dict]]) -> Dict[str, Dict[str, List[str]]]:
        """Extract terms for multiple authors from their papers"""
        author_terms = {}
        
        total_authors = len(author_papers)
        logger.info(f"Extracting terms for {total_authors} authors using spaCy...")
        
        for i, (author_id, papers) in enumerate(author_papers.items()):
            if (i + 1) % 1000 == 0:
                logger.info(f"Processing author {i+1}/{total_authors}")
            
            combined_task_terms = []
            combined_method_terms = []
            
            for paper in papers:
                title = paper.get('title', '').strip()
                abstract = paper.get('abstract', '').strip()
                
                # Combine title and abstract
                text = f"{title}. {abstract}".strip() if abstract else title.strip()
                
                if text:
                    terms = self.extract_terms_from_text(text)
                    combined_task_terms.extend(terms['task'])
                    combined_method_terms.extend(terms['method'])
            
            # Remove duplicates and filter
            author_terms[author_id] = {
                'task': list(set(t for t in combined_task_terms if t)),
                'method': list(set(t for t in combined_method_terms if t))
            }
        
        # Log statistics
        total_task_terms = sum(len(terms['task']) for terms in author_terms.values())
        total_method_terms = sum(len(terms['method']) for terms in author_terms.values())
        authors_with_tasks = sum(1 for terms in author_terms.values() if terms['task'])
        authors_with_methods = sum(1 for terms in author_terms.values() if terms['method'])
        
        logger.info(f"spaCy term extraction completed:")
        logger.info(f"  Authors with task terms: {authors_with_tasks}/{total_authors}")
        logger.info(f"  Authors with method terms: {authors_with_methods}/{total_authors}")
        logger.info(f"  Total task terms: {total_task_terms}")
        logger.info(f"  Total method terms: {total_method_terms}")
        
        return author_terms


def test_spacy_extractor():
    """Test the spaCy extractor"""
    extractor = SpaCyTermExtractor()
    
    test_text = """
    Deep learning neural networks for natural language processing tasks.
    We propose a transformer-based approach for sentiment analysis and text classification.
    The model uses attention mechanisms and convolutional layers for feature extraction.
    Our method achieves state-of-the-art performance on benchmark datasets.
    """
    
    terms = extractor.extract_terms_from_text(test_text)
    print(f"Task terms: {terms['task']}")
    print(f"Method terms: {terms['method']}")


if __name__ == "__main__":
    test_spacy_extractor()