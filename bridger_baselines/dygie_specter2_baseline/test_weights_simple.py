#!/usr/bin/env python3
"""
Simple test for author position weighting logic (without loading heavy models)
"""

import json
import sys
from pathlib import Path

def test_author_position_logic():
    """Test the core author position weighting logic"""
    
    print("Testing author position weighting logic...")
    
    # Load paper nodes
    paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
    
    try:
        with open(paper_nodes_path, 'r') as f:
            paper_nodes = json.load(f)
        print(f"Loaded {len(paper_nodes)} paper nodes")
    except Exception as e:
        print(f"Error loading paper nodes: {e}")
        return
    
    # Find a sample paper with multiple authors
    sample_papers = []
    for paper_id, paper_data in list(paper_nodes.items())[:100]:  # Check first 100 papers
        authors = paper_data.get('neighbors', {}).get('author', {})
        if len(authors) >= 3:  # Papers with 3+ authors
            sample_papers.append((paper_id, paper_data))
        if len(sample_papers) >= 3:  # Get 3 sample papers
            break
    
    if not sample_papers:
        print("No suitable sample papers found!")
        return
    
    # Test position weighting logic
    def get_author_position_weight(author_id: str, paper_data: dict) -> float:
        """Test implementation of position weighting"""
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
    
    def get_paper_importance_weight(cited_count: int) -> float:
        """Test implementation of paper importance weighting"""
        if cited_count <= 0:
            return 0.5  # Minimum weight
        elif cited_count >= 100:
            return 1.0  # Maximum weight for highly cited papers
        else:
            # Linear interpolation between 0.5 and 1.0
            return 0.5 + (cited_count / 100.0) * 0.5
    
    # Test the weighting on sample papers
    for i, (paper_id, paper_data) in enumerate(sample_papers):
        print(f"\n--- Sample Paper {i+1} (ID: {paper_id}) ---")
        
        # Get paper info
        features = paper_data.get('features', {})
        title = features.get('Title', '')[:60] + "..."
        cited_count = features.get('CitedCount', 0)
        
        print(f"Title: {title}")
        print(f"Citation count: {cited_count}")
        print(f"Paper importance weight: {get_paper_importance_weight(cited_count):.3f}")
        
        # Get authors and their positions
        authors = paper_data.get('neighbors', {}).get('author', {})
        author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
        author_positions.sort(key=lambda x: x[1])  # Sort by position
        
        print(f"Authors ({len(author_positions)} total):")
        
        for j, (author_id, position) in enumerate(author_positions):
            position_weight = get_author_position_weight(author_id, paper_data)
            importance_weight = get_paper_importance_weight(cited_count)
            combined_weight = position_weight * importance_weight
            
            role = "First" if j == 0 else ("Last" if j == len(author_positions)-1 else "Middle")
            
            print(f"  Position {position} ({role}): Author {author_id}")
            print(f"    Position weight: {position_weight}")
            print(f"    Combined weight: {combined_weight:.3f}")
    
    print("\n=== Weight Calculation Summary ===")
    print("Position weights:")
    print("  - First author: 1.0")
    print("  - Last author: 1.0") 
    print("  - Middle authors: 0.75")
    print("Paper importance weights:")
    print("  - 0 citations: 0.5")
    print("  - 100+ citations: 1.0")
    print("  - Linear scale between 0.5-1.0")
    print("Final weight = Position weight × Paper importance weight")

if __name__ == "__main__":
    test_author_position_logic()