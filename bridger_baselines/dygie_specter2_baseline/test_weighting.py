#!/usr/bin/env python3
"""
Test author position weighting functionality
"""

import sys
import json
from pathlib import Path

# Add the scripts directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.embedding_generator import SPECTER2EmbeddingGenerator
from bridger_baselines import load_author_paper_data

def test_author_weighting():
    """Test the author position weighting system"""
    
    print("Testing author position weighting...")
    
    # Load sample data
    evaluation_data_path = "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
    paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
    author_kg_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json"
    
    # Load a small sample of authors for testing
    import pandas as pd
    df = pd.read_csv(evaluation_data_path)
    
    # Get first few authors for testing
    sample_authors = set()
    for _, row in df.head(3).iterrows():
        try:
            import ast
            # Adapt to 986_paper_matching_pairs.csv format
            if 'author2' in row:
                team_authors = ast.literal_eval(row['author2'])
            elif 'author_old_paper' in row:
                team_authors = ast.literal_eval(row['author_old_paper'])
            else:
                continue
            sample_authors.update(team_authors[:2])  # Just first 2 authors per team
        except:
            continue
    
    print(f"Testing with {len(sample_authors)} sample authors")
    
    # Load author-paper data
    author_papers = load_author_paper_data(paper_nodes_path, author_kg_path, sample_authors)
    
    if not author_papers:
        print("No author data loaded!")
        return
    
    # Test the weighting functionality
    embedding_generator = SPECTER2EmbeddingGenerator()
    
    for author_id, papers in list(author_papers.items())[:2]:  # Test first 2 authors
        print(f"\nTesting author {author_id} with {len(papers)} papers:")
        
        # Test weight calculation
        try:
            weights = embedding_generator._get_term_weights(author_id, papers, 5)  # 5 dummy terms
            print(f"  Generated weights: {weights}")
            print(f"  Average weight: {weights.mean():.3f}")
            
            # Test individual paper weights
            for i, paper in enumerate(papers[:3]):  # First 3 papers
                paper_id = paper.get('paper_id')
                title = paper.get('title', '')[:50] + "..."
                print(f"  Paper {i+1} (ID: {paper_id}): {title}")
                
                if paper_id:
                    # Load paper data to check author position
                    try:
                        with open(paper_nodes_path, 'r') as f:
                            paper_nodes = json.load(f)
                        
                        if paper_id in paper_nodes:
                            paper_data = paper_nodes[paper_id]
                            position_weight = embedding_generator._get_author_position_weight(author_id, paper_data)
                            cited_count = paper_data.get('features', {}).get('CitedCount', 0)
                            importance_weight = embedding_generator._get_paper_importance_weight(cited_count)
                            
                            print(f"    Author position weight: {position_weight}")
                            print(f"    Citation count: {cited_count}")
                            print(f"    Paper importance weight: {importance_weight:.3f}")
                            print(f"    Combined weight: {position_weight * importance_weight:.3f}")
                            
                            # Show author positions in this paper
                            authors = paper_data.get('neighbors', {}).get('author', {})
                            if authors:
                                author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
                                author_positions.sort(key=lambda x: x[1])
                                print(f"    Author positions in paper: {len(author_positions)} total")
                                for j, (aid, pos) in enumerate(author_positions):
                                    marker = " <-- THIS AUTHOR" if aid == author_id else ""
                                    print(f"      Position {pos}: {aid}{marker}")
                        else:
                            print(f"    Paper {paper_id} not found in paper nodes")
                    except Exception as e:
                        print(f"    Error processing paper {paper_id}: {e}")
                
        except Exception as e:
            print(f"  Error testing weights for author {author_id}: {e}")
    
    print("\nWeighting test completed!")

if __name__ == "__main__":
    test_author_weighting()