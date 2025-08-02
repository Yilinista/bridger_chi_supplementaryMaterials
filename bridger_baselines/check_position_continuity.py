#!/usr/bin/env python3
"""
Check position continuity of authors in 986 evaluation cases
Analyze whether there are position discontinuities or missing issues
"""

import json
import pandas as pd
import ast
from collections import defaultdict, Counter

def check_evaluation_papers_position_continuity():
    """Check position continuity of authors in evaluation papers"""
    
    print("=" * 80)
    print("EVALUATION PAPERS POSITION CONTINUITY CHECK")
    print("=" * 80)
    
    # Load evaluation data
    eval_path = "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
    df = pd.read_csv(eval_path)
    
    # Load paper nodes
    paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
    with open(paper_nodes_path, 'r') as f:
        paper_nodes = json.load(f)
    
    print(f"Evaluation data: {len(df)} cases")
    print(f"Paper nodes: {len(paper_nodes)} papers")
    
    # Collect all paper IDs from evaluation data
    evaluation_paper_ids = set()
    for _, row in df.iterrows():
        evaluation_paper_ids.add(str(row['old_paper_id']))
        evaluation_paper_ids.add(str(row['New_paper_id']))
    
    print(f"Papers involved in evaluation: {len(evaluation_paper_ids)}")
    
    # Check position continuity for each paper
    continuity_stats = {
        'total_papers': 0,
        'papers_with_authors': 0,
        'continuous_positions': 0,
        'discontinuous_positions': 0,
        'position_gaps': [],
        'discontinuous_examples': []
    }
    
    position_patterns = Counter()
    
    for paper_id in evaluation_paper_ids:
        if paper_id not in paper_nodes:
            continue
            
        paper_data = paper_nodes[paper_id]
        neighbors = paper_data.get('neighbors', {})
        authors = neighbors.get('author', {})
        
        if not authors:
            continue
            
        continuity_stats['total_papers'] += 1
        continuity_stats['papers_with_authors'] += 1
        
        # Extract positions
        positions = []
        for author_id, author_info in authors.items():
            if isinstance(author_info, list) and len(author_info) > 0:
                pos = author_info[0]
                if isinstance(pos, (int, float)) and pos > 0:
                    positions.append(int(pos))
        
        if not positions:
            continue
            
        positions.sort()
        
        # Check continuity
        expected_positions = list(range(1, len(authors) + 1))
        is_continuous = positions == expected_positions
        
        # Alternative check: positions should be 1,2,3,...,n
        starts_from_1 = positions[0] == 1
        no_gaps = all(positions[i] == positions[i-1] + 1 for i in range(1, len(positions)))
        is_perfect_sequence = starts_from_1 and no_gaps and len(positions) == len(authors)
        
        if is_perfect_sequence:
            continuity_stats['continuous_positions'] += 1
        else:
            continuity_stats['discontinuous_positions'] += 1
            
            # Record the pattern
            pattern = f"Positions: {positions}, Authors: {len(authors)}"
            position_patterns[pattern] += 1
            
            # Record details for first few discontinuous cases
            if len(continuity_stats['discontinuous_examples']) < 10:
                title = paper_data.get('features', {}).get('Title', 'No title')[:50] + "..."
                
                gap_info = {
                    'paper_id': paper_id,
                    'title': title,
                    'total_authors': len(authors),
                    'positions_found': len(positions),
                    'actual_positions': positions,
                    'expected_positions': expected_positions,
                    'starts_from_1': starts_from_1,
                    'has_gaps': not no_gaps
                }
                continuity_stats['discontinuous_examples'].append(gap_info)
    
    # Print results
    print(f"\n{'='*50}")
    print("CONTINUITY ANALYSIS RESULTS")
    print("=" * 50)
    
    total = continuity_stats['papers_with_authors']
    continuous = continuity_stats['continuous_positions']
    discontinuous = continuity_stats['discontinuous_positions']
    
    print(f"Total papers checked: {total}")
    print(f"Papers with continuous positions: {continuous} ({continuous/total*100:.1f}%)")
    print(f"Papers with discontinuous positions: {discontinuous} ({discontinuous/total*100:.1f}%)"
    
    if discontinuous > 0:
        print(f"\n{'='*50}")
        print("DISCONTINUOUS POSITION PATTERNS")
        print("=" * 50)
        
        print(f"Found {len(position_patterns)} different discontinuous patterns:")
        for pattern, count in position_patterns.most_common(10):
            print(f"  {count}x: {pattern}")
        
        print(f"\n{'='*50}")
        print("DISCONTINUOUS EXAMPLES")
        print("=" * 50)
        
        for i, example in enumerate(continuity_stats['discontinuous_examples']):
            print(f"\nExample {i+1}: Paper {example['paper_id']}")
            print(f"  Title: {example['title']}")
            print(f"  Total authors: {example['total_authors']}")
            print(f"  Authors with position info: {example['positions_found']}")
            print(f"  Actual positions: {example['actual_positions']}")
            print(f"  Expected positions: {example['expected_positions']}")
            print(f"  Starts from 1: {example['starts_from_1']}")
            print(f"  Has gaps: {example['has_gaps']}"
            
            # Show impact on our logic
            if example['actual_positions']:
                positions = example['actual_positions']
                our_first = positions[0]  # Our logic: min position
                our_last = positions[-1]  # Our logic: max position
                expected_first = 1
                expected_last = example['total_authors']
                
                print(f"  Our logic:")
                print(f"    First author position: {our_first} (expected: {expected_first})")
                print(f"    Last author position: {our_last} (expected: {expected_last})"
                
                if our_first == expected_first and our_last == expected_last:
                    print(f"    Our logic is still correct")
                else:
                    print(f"    Our logic may have bias")
    
    else:
        print("\nAll evaluation papers have continuous author positions!")
    
    return discontinuous == 0

def analyze_specific_discontinuous_cases():
    """Analyze specific discontinuous cases to understand the reasons"""
    
    print(f"\n{'='*80}")
    print("DETAILED DISCONTINUOUS CASE ANALYSIS")
    print("=" * 80)
    
    # Load data
    eval_path = "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
    df = pd.read_csv(eval_path)
    
    paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
    with open(paper_nodes_path, 'r') as f:
        paper_nodes = json.load(f)
    
    # Find papers from evaluation data that have author position issues
    problem_cases = []
    
    evaluation_paper_ids = set()
    for _, row in df.iterrows():
        evaluation_paper_ids.add(str(row['old_paper_id']))
        evaluation_paper_ids.add(str(row['New_paper_id']))
    
    # Sample check more papers to find issues
    for paper_id in list(evaluation_paper_ids)[:100]:  # Check first 100 evaluation papers
        if paper_id not in paper_nodes:
            continue
            
        paper_data = paper_nodes[paper_id]
        neighbors = paper_data.get('neighbors', {})
        authors = neighbors.get('author', {})
        
        if not authors:
            continue
        
        # Get positions
        author_positions = []
        for author_id, author_info in authors.items():
            if isinstance(author_info, list) and len(author_info) > 0:
                pos = author_info[0]
                if isinstance(pos, (int, float)) and pos > 0:
                    author_positions.append((author_id, int(pos)))
        
        if not author_positions:
            continue
            
        author_positions.sort(key=lambda x: x[1])
        positions = [pos for _, pos in author_positions]
        
        # Check for issues
        expected = list(range(1, len(authors) + 1))
        has_issue = positions != expected
        
        if has_issue:
            title = paper_data.get('features', {}).get('Title', 'No title')
            problem_cases.append({
                'paper_id': paper_id,
                'title': title,
                'positions': positions,
                'expected': expected,
                'author_count': len(authors),
                'position_count': len(positions)
            })
    
    if problem_cases:
        print(f"Found {len(problem_cases)} evaluation papers with position issues:")
        
        for case in problem_cases[:5]:  # Show first 5
            print(f"\nPaper {case['paper_id']}:")
            print(f"  Title: {case['title'][:60]}...")
            print(f"  Author count: {case['author_count']}")
            print(f"  Position count: {case['position_count']}")
            print(f"  Actual positions: {case['positions']}")
            print(f"  Expected positions: {case['expected']}"
            
            # Analyze the issue type
            if case['positions'][0] != 1:
                print(f"  Issue: Does not start from 1 (starts from {case['positions'][0]})")
            if len(case['positions']) != case['author_count']:
                print(f"  Issue: Missing position info ({case['position_count']}/{case['author_count']})")
            if any(case['positions'][i] != case['positions'][i-1] + 1 for i in range(1, len(case['positions']))):
                print(f"  Issue: Gaps in positions")
        
        return False
    else:
        print("No position discontinuity issues found in the 100 evaluation papers checked")
        return True

if __name__ == "__main__":
    # Check overall continuity
    all_continuous = check_evaluation_papers_position_continuity()
    
    # Analyze specific cases if issues found
    if not all_continuous:
        sample_continuous = analyze_specific_discontinuous_cases()
    else:
        sample_continuous = True
    
    print(f"\n{'='*80}")
    print("FINAL ASSESSMENT")
    print("=" * 80)
    
    if all_continuous and sample_continuous:
        print("Author position information quality in evaluation data is high")
        print("Our First/Last author identification logic is fully applicable")
        print("No need to worry about position discontinuity issues")
    else:
        print("Found some position discontinuity cases")
        print("Need to evaluate impact on our logic")
        print("Our logic has some fault tolerance and should handle these cases")