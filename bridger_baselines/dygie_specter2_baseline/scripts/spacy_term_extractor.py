#!/usr/bin/env python3
"""
spaCy-based term extractor as fallback when DyGIE++ is not available
"""

import spacy
import re
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

class SpaCyTermExtractor:
    """Extract scientific terms using spaCy NER and rule-based patterns"""
    
    def __init__(self):
        self.nlp = None
        self._load_spacy()
        self._setup_patterns()
    
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
    
    def _setup_patterns(self):
        """Setup patterns for identifying scientific terms"""
        # Method keywords - things that describe HOW research is done
        self.method_keywords = {
            'algorithm', 'method', 'approach', 'technique', 'framework', 'model', 'system',
            'neural', 'network', 'deep', 'learning', 'machine', 'statistical', 'regression',
            'classification', 'clustering', 'optimization', 'search', 'genetic', 'evolutionary',
            'bayesian', 'probabilistic', 'stochastic', 'deterministic', 'heuristic',
            'supervised', 'unsupervised', 'reinforcement', 'semi-supervised',
            'convolutional', 'recurrent', 'transformer', 'attention', 'encoder', 'decoder',
            'feature', 'extraction', 'selection', 'engineering', 'preprocessing',
            'training', 'testing', 'validation', 'cross-validation', 'evaluation',
            'metric', 'measure', 'score', 'accuracy', 'precision', 'recall', 'f1',
            'loss', 'function', 'objective', 'cost', 'penalty', 'regularization',
            'gradient', 'descent', 'backpropagation', 'forward', 'backward',
            'linear', 'nonlinear', 'kernel', 'support', 'vector', 'decision', 'tree',
            'random', 'forest', 'ensemble', 'boosting', 'bagging', 'voting'
        }
        
        # Task keywords - things that describe WHAT is being solved
        self.task_keywords = {
            'task', 'problem', 'application', 'domain', 'field', 'area', 'research',
            'recognition', 'detection', 'identification', 'classification', 'prediction',
            'estimation', 'generation', 'synthesis', 'analysis', 'processing', 'understanding',
            'parsing', 'segmentation', 'clustering', 'matching', 'alignment', 'retrieval',
            'recommendation', 'ranking', 'scoring', 'filtering', 'selection',
            'language', 'speech', 'vision', 'image', 'video', 'text', 'document',
            'sentiment', 'emotion', 'opinion', 'stance', 'aspect', 'topic', 'theme',
            'semantic', 'syntactic', 'morphological', 'phonetic', 'lexical',
            'translation', 'summarization', 'simplification', 'generation', 'completion',
            'question', 'answering', 'dialogue', 'conversation', 'chatbot', 'assistant',
            'knowledge', 'graph', 'extraction', 'construction', 'completion', 'reasoning',
            'inference', 'entailment', 'contradiction', 'paraphrase', 'similarity'
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
        """Classify term as task or method based on keywords and context"""
        term_lower = term.lower()
        context_lower = context.lower()
        
        # Check for method indicators
        method_score = sum(1 for kw in self.method_keywords if kw in term_lower)
        method_score += sum(0.5 for kw in self.method_keywords if kw in context_lower)
        
        # Check for task indicators  
        task_score = sum(1 for kw in self.task_keywords if kw in term_lower)
        task_score += sum(0.5 for kw in self.task_keywords if kw in context_lower)
        
        # Default classification based on scores
        if method_score > task_score:
            return 'method'
        elif task_score > method_score:
            return 'task'
        else:
            # Tie-breaker: longer phrases tend to be tasks, shorter tend to be methods
            return 'task' if len(term.split()) > 2 else 'method'
    
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
        """Check if a term looks like a scientific term"""
        # Skip very common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        if term in common_words:
            return False
        
        # Check for scientific indicators
        scientific_indicators = [
            # Has scientific keywords
            any(kw in term for kw in self.method_keywords | self.task_keywords),
            # Contains technical suffixes
            any(term.endswith(suffix) for suffix in ['-based', '-driven', '-aware', '-free', 'tion', 'sion', 'ness', 'ment']),
            # Contains technical prefixes  
            any(term.startswith(prefix) for prefix in ['multi-', 'semi-', 'auto-', 'self-', 'cross-', 'meta-']),
            # Contains numbers or abbreviations
            bool(re.search(r'\d|[A-Z]{2,}', term)),
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