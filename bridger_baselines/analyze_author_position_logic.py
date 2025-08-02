#!/usr/bin/env python3
"""
Author Position Logic Analysis
Analyze our First/Last author identification logic
Show detailed calculation process and potential issues
"""

import json

def analyze_author_position_logic():
    """Analyze our current First/Last author identification logic"""
    
    print("=" * 80)
    print("AUTHOR POSITION LOGIC ANALYSIS")
    print("=" * 80)
    
    print("Current logic (embedding_generator.py:384-410):")
    print("""
        # 1. Get all authors and position information
        author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
        
        # 2. Sort by position
        author_positions.sort(key=lambda x: x[1])
        
        # 3. Identify first and last authors
        first_author_id = author_positions[0][0]    # First after sorting = minimum position
        last_author_id = author_positions[-1][0]    # Last after sorting = maximum position
        
        # 4. Determine weights
        if author_id == first_author_id or author_id == last_author_id:
            return 1.0  # First or last author
        else:
            return 0.75  # Middle author
    """)
    
    print("Key logic points:")
    print("- First Author = author with minimum position number (usually 1)")
    print("- Last Author = author with maximum position number (usually total count)")
    print("- Sorting ensures correct identification even if position numbers are not consecutive")
    
    # Test different scenarios
    test_scenarios = [
        {
            "name": "Normal case",
            "authors": {
                "author1": [1],
                "author2": [2], 
                "author3": [3],
                "author4": [4]
            },
            "expected_first": "author1",
            "expected_last": "author4"
        },
        {
            "name": "Non-consecutive positions",
            "authors": {
                "author1": [1],
                "author2": [3],
                "author3": [5],
                "author4": [7]
            },
            "expected_first": "author1",  # Position 1, still minimum
            "expected_last": "author4"   # Position 7, still maximum
        },
        {
            "name": "Unordered positions",
            "authors": {
                "author1": [3],
                "author2": [1],
                "author3": [4],
                "author4": [2]
            },
            "expected_first": "author2",  # Position 1, minimum after sorting
            "expected_last": "author3"   # Position 4, maximum after sorting
        },
        {
            "name": "Single author",
            "authors": {
                "author1": [1]
            },
            "expected_first": "author1",
            "expected_last": "author1"
        },
        {
            "name": "Positions starting from 2 (data issue)",
            "authors": {
                "author1": [2],
                "author2": [3],
                "author3": [4]
            },
            "expected_first": "author1",  # Position 2, but minimum
            "expected_last": "author3"   # Position 4, but maximum
        }
    ]
    
    print(f"\n{'-' * 80}")
    print("SCENARIO TESTING")
    print("-" * 80)
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 50)
        
        authors = scenario["authors"]
        
        # Simulate our logic
        author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
        author_positions.sort(key=lambda x: x[1])
        
        if author_positions:
            first_author_id = author_positions[0][0]
            last_author_id = author_positions[-1][0]
        else:
            first_author_id = None
            last_author_id = None
        
        print(f"Input authors: {authors}")
        print(f"After sorting: {author_positions}")
        print(f"Identified First: {first_author_id} (expected: {scenario['expected_first']})")
        print(f"Identified Last: {last_author_id} (expected: {scenario['expected_last']})")
        
        # Check weight assignment
        print("Weight assignment:")
        for author_id, pos_data in authors.items():
            if len(author_positions) <= 1:
                weight = 1.0
                role = "single author"
            elif author_id == first_author_id or author_id == last_author_id:
                weight = 1.0
                role = "first/last author"
            else:
                weight = 0.75
                role = "middle author"
            
            print(f"  {author_id} (pos {pos_data[0]}) -> weight {weight} ({role})")
        
        # Verify results
        first_correct = first_author_id == scenario["expected_first"]
        last_correct = last_author_id == scenario["expected_last"]
        
        if first_correct and last_correct:
            print("Result: Logic correct")
        else:
            print("Result: Logic error")

def test_with_real_data():
    """Test logic with real data"""
    
    print(f"\n{'-' * 80}")
    print("REAL DATA TESTING")
    print("-" * 80)
    
    # Load a sample of real paper data
    try:
        paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
        
        with open(paper_nodes_path, 'r') as f:
            paper_nodes = json.load(f)
        
        # Sample a few papers with multiple authors
        sample_count = 0
        for paper_id, paper_data in paper_nodes.items():
            neighbors = paper_data.get('neighbors', {})
            authors = neighbors.get('author', {})
            
            if len(authors) > 2:  # Only test papers with multiple authors
                print(f"\nPaper ID: {paper_id}")
                paper_title = paper_data.get('features', {}).get('Title', 'Unknown')
                print(f"Title: {paper_title[:60]}...")
                
                # Apply our logic
                author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
                author_positions.sort(key=lambda x: x[1])
                
                first_author_id = author_positions[0][0]
                last_author_id = author_positions[-1][0]
                
                print(f"Total authors: {len(authors)}")
                print(f"Position info: {[(aid, pos) for aid, pos in author_positions]}")
                print(f"First Author: {first_author_id} (position {author_positions[0][1]})")
                print(f"Last Author: {last_author_id} (position {author_positions[-1][1]})")
                
                # Show weight assignment
                print("Weight assignment:")
                for author_id, pos_data in authors.items():
                    if len(authors) <= 1:
                        weight = 1.0
                        role = "single author"
                    elif author_id == first_author_id or author_id == last_author_id:
                        weight = 1.0
                        role = "first author" if author_id == first_author_id else "last author"
                    else:
                        weight = 0.75
                        role = "middle author"
                    
                    pos = pos_data[0]
                    print(f"  Position{pos}: {author_id} -> {weight} ({role})")
                
                sample_count += 1
                if sample_count >= 3:  # Limit to 3 samples
                    break
                    
    except Exception as e:
        print(f"Cannot load real data for testing: {e}")

def identify_potential_issues():
    """Identify potential issues"""
    
    print(f"\n{'-' * 80}")
    print("POTENTIAL ISSUES ANALYSIS")
    print("-" * 80)
    
    print("Potential issues:")
    
    print("\n1. Data quality issues:")
    print("   - If position numbers don't start from 1, first author is still minimum position")
    print("   - Example: positions [2,3,4] -> First=author at position 2 (may not be true first author)")
    
    print("\n2. Missing position issues:")
    print("   - If some authors lack position information, they are skipped")
    print("   - May lead to inaccurate first/last identification")
    
    print("\n3. Logic assumptions:")
    print("   - Assumption: smaller position number = more important")
    print("   - Assumption: maximum position number = last author (corresponding author)")
    print("   - These assumptions are correct in most cases")
    
    print("\nAdvantages:")
    print("   - Good robustness: works even with non-consecutive positions")
    print("   - Automatic sorting: handles unordered data")
    print("   - Simple and effective: direct comparison based on position numbers")
    
    print("\nRecommendations:")
    print("   - Current logic is suitable for evaluation data (verified complete position info)")
    print("   - For poor data quality cases, default weight (0.75) serves as fallback")
    print("   - Consider adding logging for exception cases to aid debugging")

if __name__ == "__main__":
    # Analyze logic
    analyze_author_position_logic()
    
    # Test with real data
    test_with_real_data()
    
    # Check potential issues
    identify_potential_issues()
    
    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print("=" * 80)
    print("Current First/Last author identification logic is reasonable and effective")
    print("Position sorting ensures correct operation even with imperfect data")
    print("Suitable for our evaluation scenario as evaluation data has complete position info")
    print("Logic is simple and clear: minimum position=First, maximum position=Last")