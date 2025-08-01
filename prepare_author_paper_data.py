#!/usr/bin/env python3
"""
Data Preparation for Faithful Bridger Implementation

This script helps prepare the author-paper dataset needed for faithful Bridger replication.
It analyzes your current data and shows what additional data is needed.
"""

import pandas as pd
import ast
from collections import defaultdict
from typing import Dict, List, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_current_data_structure(data_path: str):
    """Analyze the current BetterTeaming dataset structure."""
    logger.info(f"Analyzing data structure: {data_path}")
    
    df = pd.read_csv(data_path)
    
    print("="*60)
    print("CURRENT DATA STRUCTURE ANALYSIS")
    print("="*60)
    
    # Basic info
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Extract all unique authors
    all_authors = set()
    for idx, row in df.iterrows():
        try:
            authors = ast.literal_eval(row['author2'])
            all_authors.update(authors)
        except:
            pass
        
        if pd.notna(row['ground_truth_authors']):
            gt_authors = row['ground_truth_authors'].split('|')
            all_authors.update([a.strip() for a in gt_authors])
    
    print(f"Unique authors found: {len(all_authors)}")
    print()
    
    # Check text availability
    print("TEXT DATA AVAILABILITY:")
    print(f"  paper titles (title2): {df['title2'].notna().sum()}/{len(df)} available")
    print(f"  paper abstracts (abstract2): {df['abstract2'].notna().sum()}/{len(df)} available")
    print(f"  reference titles (titles1): {df['titles1'].notna().sum()}/{len(df)} available")
    print()
    
    # Show what's missing for faithful replication
    print("MISSING FOR FAITHFUL BRIDGER REPLICATION:")
    print("❌ Individual author publication histories")
    print("❌ Paper abstracts for all authors")
    print("❌ Full corpus of papers per author")
    print("❌ DyGIE++ NER model for term extraction")
    print()
    
    return all_authors, df


def suggest_data_requirements():
    """Suggest what data is needed for faithful Bridger replication."""
    
    print("="*60)
    print("DATA REQUIREMENTS FOR FAITHFUL BRIDGER REPLICATION")
    print("="*60)
    
    print("You need a dataset with the following structure:")
    print()
    
    print("OPTION 1: Author-Paper Table")
    print("CSV with columns:")
    print("  - author_id: string")
    print("  - paper_id: string") 
    print("  - title: string")
    print("  - abstract: string")
    print("  - year: int (optional)")
    print()
    
    print("OPTION 2: MAG-style Dataset")
    print("Multiple files:")
    print("  - authors.csv: author_id, name")
    print("  - papers.csv: paper_id, title, abstract, year")
    print("  - paper_author_affiliations.csv: paper_id, author_id")
    print()
    
    print("OPTION 3: JSON Format")
    print("Structure:")
    print("  {")
    print("    'author_123': [")
    print("      {'paper_id': 'p1', 'title': '...', 'abstract': '...'},")
    print("      {'paper_id': 'p2', 'title': '...', 'abstract': '...'},")
    print("      ...")
    print("    ],")
    print("    ...")
    print("  }")
    print()


def create_sample_author_paper_data(authors_sample: List[str], output_path: str):
    """Create a sample author-paper dataset for testing."""
    
    logger.info(f"Creating sample data for {len(authors_sample)} authors...")
    
    sample_data = []
    
    # Create fake papers for each author
    for author_id in authors_sample[:10]:  # Limit to first 10 for demonstration
        
        # Generate 3-5 fake papers per author
        import random
        random.seed(hash(author_id) % 10000)
        
        num_papers = random.randint(3, 5)
        
        for i in range(num_papers):
            paper_id = f"paper_{author_id}_{i}"
            
            # Generate fake but realistic-looking academic titles
            tasks = ['classification', 'detection', 'analysis', 'prediction', 'optimization']
            domains = ['neural networks', 'computer vision', 'natural language', 'machine learning', 'data mining']
            methods = ['deep learning', 'reinforcement learning', 'transformer', 'CNN', 'statistical']
            
            task = random.choice(tasks)
            domain = random.choice(domains)
            method = random.choice(methods)
            
            title = f"Improved {task} in {domain} using {method} approaches"
            abstract = f"This paper presents a novel approach to {task} in the field of {domain}. " \
                      f"We propose a {method}-based method that outperforms existing techniques. " \
                      f"Our approach leverages advanced {method} architectures to achieve better " \
                      f"performance in {task} tasks. Experimental results demonstrate significant " \
                      f"improvements over baseline methods in {domain} applications."
            
            sample_data.append({
                'author_id': author_id,
                'paper_id': paper_id,
                'title': title,
                'abstract': abstract,
                'year': random.randint(2015, 2023)
            })
    
    # Save sample data
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv(output_path, index=False)
    
    logger.info(f"Sample data saved to {output_path}")
    print(f"\nSample data created: {len(df_sample)} paper records for {len(set(df_sample['author_id']))} authors")
    print(f"Saved to: {output_path}")
    
    return df_sample


def test_faithful_implementation_with_sample_data(sample_data_path: str):
    """Test the faithful implementation with sample data."""
    
    from faithful_bridger_implementation import FaithfulBridgerImplementation
    
    logger.info("Testing faithful implementation with sample data...")
    
    # Load sample data
    df = pd.read_csv(sample_data_path)
    
    # Convert to required format
    author_papers = defaultdict(list)
    for _, row in df.iterrows():
        author_papers[row['author_id']].append({
            'paper_id': row['paper_id'],
            'title': row['title'],
            'abstract': row['abstract']
        })
    
    # Initialize and run faithful implementation
    bridger = FaithfulBridgerImplementation(dict(author_papers))
    bridger.process_all_authors()
    
    # Show results
    summary = bridger.get_embeddings_summary()
    
    print("\n" + "="*60)
    print("FAITHFUL IMPLEMENTATION TEST RESULTS")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Show some extracted terms
    print("\nSample extracted terms:")
    for author_id in list(bridger.author_task_terms.keys())[:3]:
        task_terms = bridger.author_task_terms[author_id]
        method_terms = bridger.author_method_terms[author_id]
        print(f"  Author {author_id}:")
        print(f"    Tasks: {task_terms}")
        print(f"    Methods: {method_terms}")


def main():
    """Main analysis and preparation workflow."""
    
    # Path to your BetterTeaming dataset
    data_path = "/data/jx4237data/Graph-CoT/Pipeline/step1_process/strict_0.88_remove_case1_year2-5/paper_levels_0.88_year2-5.csv"
    
    # Analyze current data
    all_authors, df = analyze_current_data_structure(data_path)
    
    # Show requirements
    suggest_data_requirements()
    
    # Create sample data for testing
    sample_authors = sorted(list(all_authors))
    sample_data_path = "./sample_author_paper_data.csv"
    create_sample_author_paper_data(sample_authors, sample_data_path)
    
    # Test faithful implementation
    test_faithful_implementation_with_sample_data(sample_data_path)
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR FAITHFUL BRIDGER REPLICATION")
    print("="*60)
    print("1. Obtain author publication histories with abstracts")
    print("2. Set up DyGIE++ NER model for term extraction")
    print("3. Replace rule-based extraction with proper NER")
    print("4. Use real sentence transformer embeddings")
    print("5. Run faithful pipeline on full dataset")
    print("6. Compare results with MATRIX using same evaluation framework")


if __name__ == "__main__":
    main()