#!/usr/bin/env python3
"""
Check completeness of author position information in Paper Nodes
Verify if author positions are complete and accurate
"""

import json
import pandas as pd
from collections import defaultdict, Counter
import numpy as np

def check_author_position_completeness():
    """Check completeness of author position information"""
    
    print("=" * 80)
    print("AUTHOR POSITION INFORMATION COMPLETENESS CHECK")
    print("=" * 80)
    
    # Load paper nodes
    paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
    
    print(f"Loading paper nodes from: {paper_nodes_path}")
    with open(paper_nodes_path, 'r') as f:
        paper_nodes = json.load(f)
    
    print(f"Total papers: {len(paper_nodes)}")
    
    # Statistics
    papers_with_authors = 0
    papers_with_position_info = 0
    position_completeness_stats = []
    position_gap_examples = []
    
    # Sample some papers for detailed analysis
    sample_papers = list(paper_nodes.items())[:1000]  # First 1000 papers
    
    print(f"\nAnalyzing first {len(sample_papers)} papers...")
    
    for paper_id, paper_data in sample_papers:
        neighbors = paper_data.get('neighbors', {})
        authors = neighbors.get('author', {})
        
        if not authors:
            continue
            
        papers_with_authors += 1
        
        # Extract author positions
        author_positions = []
        authors_with_positions = 0
        
        for author_id, author_info in authors.items():
            if isinstance(author_info, list) and len(author_info) > 0:
                position = author_info[0]
                if isinstance(position, (int, float)) and position > 0:
                    author_positions.append(int(position))
                    authors_with_positions += 1
                else:
                    # Invalid position
                    pass
            else:
                # No position info
                pass
        
        if author_positions:
            papers_with_position_info += 1
            
            # Check position completeness
            author_positions.sort()
            expected_positions = list(range(1, len(authors) + 1))
            actual_positions = sorted(author_positions)
            
            # Calculate completeness
            if len(actual_positions) == len(authors):
                completeness = len(set(actual_positions) & set(expected_positions)) / len(expected_positions)
            else:
                completeness = len(actual_positions) / len(authors)
            
            position_completeness_stats.append(completeness)
            
            # Check for gaps or issues
            if completeness < 1.0:
                gap_info = {
                    'paper_id': paper_id,
                    'total_authors': len(authors),
                    'authors_with_positions': authors_with_positions,
                    'expected_positions': expected_positions,
                    'actual_positions': actual_positions,
                    'completeness': completeness
                }
                position_gap_examples.append(gap_info)
    
    print(f"\n{'='*50}")
    print("ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"Papers with authors: {papers_with_authors}")
    print(f"Papers with position info: {papers_with_position_info}")
    print(f"Position info coverage: {papers_with_position_info/papers_with_authors*100:.1f}%")
    
    if position_completeness_stats:
        completeness_array = np.array(position_completeness_stats)
        print(f"\nPosition completeness statistics:")
        print(f"  Mean completeness: {completeness_array.mean():.3f}")
        print(f"  Median completeness: {np.median(completeness_array):.3f}")
        print(f"  Papers with 100% completeness: {np.sum(completeness_array == 1.0)/len(completeness_array)*100:.1f}%")
        print(f"  Papers with <90% completeness: {np.sum(completeness_array < 0.9)/len(completeness_array)*100:.1f}%")
    
    # Show examples of problematic papers
    if position_gap_examples:
        print(f"\n{'='*50}")
        print("EXAMPLES OF INCOMPLETE POSITION INFO")
        print("=" * 50)
        
        # Sort by worst completeness
        position_gap_examples.sort(key=lambda x: x['completeness'])
        
        for i, example in enumerate(position_gap_examples[:5]):  # Show top 5 worst cases
            print(f"\nExample {i+1}: Paper {example['paper_id']}")
            print(f"  Total authors: {example['total_authors']}")
            print(f"  Authors with positions: {example['authors_with_positions']}")
            print(f"  Expected positions: {example['expected_positions']}")
            print(f"  Actual positions: {example['actual_positions']}")
            print(f"  Completeness: {example['completeness']:.1%}")
    
    return papers_with_position_info > 0

def check_specific_papers_from_evaluation():
    """Check author position information for specific papers in evaluation data"""
    
    print(f"\n{'='*80}")
    print("CHECKING AUTHOR POSITIONS IN EVALUATION PAPERS")
    print("=" * 80)
    
    # Load evaluation data
    eval_path = "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
    df = pd.read_csv(eval_path)
    
    # Load paper nodes
    paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
    with open(paper_nodes_path, 'r') as f:
        paper_nodes = json.load(f)
    
    # Check some specific papers from evaluation data
    sample_paper_ids = []
    for _, row in df.head(5).iterrows():
        sample_paper_ids.append(str(row['old_paper_id']))
        sample_paper_ids.append(str(row['New_paper_id']))
    
    print(f"Checking position info for evaluation papers: {sample_paper_ids}")
    
    for paper_id in sample_paper_ids:
        if paper_id in paper_nodes:
            paper_data = paper_nodes[paper_id]
            print(f"\n--- Paper {paper_id} ---")
            
            # Get paper title for context
            title = paper_data.get('features', {}).get('Title', 'No title')[:60] + "..."
            print(f"Title: {title}")
            
            neighbors = paper_data.get('neighbors', {})
            authors = neighbors.get('author', {})
            
            if authors:
                print(f"Authors ({len(authors)} total):")
                
                # Sort authors by position if available
                author_list = []
                for author_id, author_info in authors.items():
                    if isinstance(author_info, list) and len(author_info) > 0:
                        position = author_info[0] if isinstance(author_info[0], (int, float)) else None
                        author_list.append((author_id, position, author_info))
                    else:
                        author_list.append((author_id, None, author_info))
                
                # Sort by position (put None at end)
                author_list.sort(key=lambda x: x[1] if x[1] is not None else 999)
                
                for author_id, position, full_info in author_list:
                    print(f"  Position {position}: Author {author_id} - {full_info}")
                
                # Check position sequence
                positions = [pos for _, pos, _ in author_list if pos is not None]
                if positions:
                    expected = list(range(1, len(authors) + 1))
                    actual = sorted(positions)
                    print(f"  Expected positions: {expected}")
                    print(f"  Actual positions: {actual}")
                    
                    if expected == actual:
                        print(f"  Position sequence is complete")
                    else:
                        print(f"  Position sequence has gaps or issues")
            else:
                print(f"  No author information found")
        else:
            print(f"\n--- Paper {paper_id} ---")
            print(f"  Paper not found in paper nodes")

def check_alternative_author_sources():
    """Check if there are other data sources containing complete author position information"""
    
    print(f"\n{'='*80}")
    print("CHECKING ALTERNATIVE AUTHOR DATA SOURCES")
    print("=" * 80)
    
    # Check if there are other potential sources
    base_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data"
    
    print(f"Checking for alternative author data sources in: {base_path}")
    
    import os
    from pathlib import Path
    
    base_dir = Path(base_path)
    if base_dir.exists():
        # Look for any files that might contain author information
        potential_files = []
        for item in base_dir.rglob("*author*"):
            if item.is_file() and item.suffix in ['.json', '.csv']:
                size_mb = item.stat().st_size / (1024*1024)
                potential_files.append((str(item), size_mb))
        
        print(f"\nFound potential author data files:")
        for file_path, size_mb in potential_files:
            print(f"  {file_path} ({size_mb:.1f} MB)")
        
        # Check if there's a paper-author mapping file
        for file_path, size_mb in potential_files:
            if 'paper' in file_path.lower() and 'author' in file_path.lower():
                print(f"\nExamining: {file_path}")
                try:
                    if file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            sample_data = json.load(f)
                        if isinstance(sample_data, dict):
                            sample_keys = list(sample_data.keys())[:3]
                            print(f"  Sample keys: {sample_keys}")
                            if sample_keys:
                                sample_value = sample_data[sample_keys[0]]
                                print(f"  Sample value structure: {type(sample_value)} - {sample_value}")
                    elif file_path.endswith('.csv'):
                        sample_df = pd.read_csv(file_path, nrows=5)
                        print(f"  Columns: {list(sample_df.columns)}")
                        print(f"  Shape: {sample_df.shape}")
                except Exception as e:
                    print(f"  Error reading file: {e}")
    
    print(f"\nRECOMMENDATIONS:")
    print(f"1. If position info is incomplete in paper_nodes, look for paper-author CSV files")
    print(f"2. Check if MAG (Microsoft Academic Graph) original data has better position info")
    print(f"3. Consider using approximate position weighting based on available data")

if __name__ == "__main__":
    # Check overall completeness
    has_position_info = check_author_position_completeness()
    
    # Check specific evaluation papers
    check_specific_papers_from_evaluation()
    
    # Look for alternative sources
    check_alternative_author_sources()
    
    print(f"\n{'='*80}")
    print("CONCLUSION AND RECOMMENDATIONS")
    print("=" * 80)
    
    if has_position_info:
        print("Paper nodes contain author position information")
        print("Check the completeness statistics above")
        print("Current weighting implementation should work, but may have gaps")
    else:
        print("No author position information found in paper nodes")
        print("Current weighting implementation will not work correctly")
        print("Need to find alternative data source or implement fallback weighting")