#!/usr/bin/env python3
"""
Test term normalization functionality
"""

import sys
from pathlib import Path

# Add the scripts directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from embedding_generator import DyGIETermExtractor

def test_term_normalization():
    """Test term normalization functionality"""
    
    print("Testing term normalization functionality...")
    
    # Initialize extractor
    extractor = DyGIETermExtractor()
    
    # Test cases: original term -> expected normalized result
    test_cases = [
        ("Deep Learning", "deep learning"),
        ("machine learning algorithms", "machine learning algorithm"),
        ("Natural Language Processing (NLP)", "natural language processing"),
        ("neural networks,", "neural network"),
        ("Convolutional Neural Networks", "convolutional neural network"),
        ("deep   learning   methods", "deep learning method"),
        ("BERT (Bidirectional Encoder)", "bert"),
        ("data mining techniques", "data mining technique"),
        ("Support Vector Machines", "support vector machine"),
        ("reinforcement learning.", "reinforcement learning"),
    ]
    
    print("\nTerm normalization test results:")
    print("=" * 60)
    
    success_count = 0
    total_count = len(test_cases)
    
    for original, expected in test_cases:
        try:
            normalized = extractor._normalize_term(original)
            success = normalized == expected
            status = "PASS" if success else "FAIL"
            
            print(f"{status}: '{original}'")
            print(f"   Expected: '{expected}'")
            print(f"   Actual:   '{normalized}'")
            
            if success:
                success_count += 1
            print()
            
        except Exception as e:
            print(f"FAIL: '{original}' - Error: {e}")
            print()
    
    print("=" * 60)
    print(f"Test results: {success_count}/{total_count} passed")
    
    # Demonstrate normalization features
    print("\nNormalization functionality demo:")
    print("-" * 40)
    
    demo_terms = [
        "Deep Learning",
        "MACHINE LEARNING ALGORITHMS", 
        "Natural Language Processing (NLP)",
        "neural networks,",
        "deep   learning   methods",
    ]
    
    for term in demo_terms:
        normalized = extractor._normalize_term(term)
        print(f"'{term}' -> '{normalized}'")

if __name__ == "__main__":
    test_term_normalization()