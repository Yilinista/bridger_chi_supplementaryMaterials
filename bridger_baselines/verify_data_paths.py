#!/usr/bin/env python3
"""
Data Path Verification Script
Verify all data paths exist and are accessible
"""

import os
import pandas as pd
import json
from pathlib import Path

def verify_paths():
    """Verify all critical data paths"""
    
    print("=" * 60)
    print("DATA PATH VERIFICATION")
    print("=" * 60)
    
    # Define all data paths
    paths = {
        "Evaluation Data": "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv",
        "Paper Nodes": "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json",
        "Author Knowledge Graph": "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json",
        "DyGIE++ Model": "./dygie_specter2_baseline/dygiepp/pretrained/scierc.tar.gz"
    }
    
    all_good = True
    
    for name, path in paths.items():
        print(f"\n{name}:")
        print(f"  Path: {path}")
        
        if os.path.exists(path):
            print(f"  Status: File exists")
            
            # Get file size
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")
            
            # Special checks
            if path.endswith('.csv'):
                try:
                    df = pd.read_csv(path)
                    print(f"  CSV shape: {df.shape}")
                    print(f"  Columns: {list(df.columns)}")
                    
                    # Check key columns
                    if 'author2' in df.columns:
                        print(f"  Contains 'author2' column")
                    elif 'author_old_paper' in df.columns:
                        print(f"  Contains 'author_old_paper' column (can map to team_authors)")
                    else:
                        print(f"  Warning: Missing 'author2' or 'author_old_paper' column")
                        all_good = False
                        
                except Exception as e:
                    print(f"  Error: CSV read failed: {e}")
                    all_good = False
                    
            elif path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    print(f"  JSON entries: {len(data)}")
                    
                    # Show sample keys
                    if isinstance(data, dict):
                        sample_keys = list(data.keys())[:3]
                        print(f"  Sample keys: {sample_keys}")
                        
                except Exception as e:
                    print(f"  Error: JSON read failed: {e}")
                    all_good = False
                    
        else:
            print(f"  Error: File does not exist")
            all_good = False
    
    print(f"\n{'=' * 60}")
    if all_good:
        print("All data paths verified successfully!")
        print("System ready to run Bridger baseline evaluation")
    else:
        print("Some data paths have issues")
        print("Please check and fix the above issues before running")
    print("=" * 60)
    
    return all_good

def check_evaluation_data_format():
    """Check evaluation data format in detail"""
    
    eval_path = "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
    
    if not os.path.exists(eval_path):
        print(f"Error: Evaluation data file not found: {eval_path}")
        return False
    
    print(f"\n{'=' * 60}")
    print("EVALUATION DATA FORMAT CHECK")
    print("=" * 60)
    
    try:
        df = pd.read_csv(eval_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show first few rows
        print(f"\nFirst 5 rows preview:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.head())
        
        # Check required columns
        required_columns = ['author2', 'author_old_paper']  # Either one is acceptable
        optional_columns = ['ground_truth_authors']
        
        print(f"\nColumn check:")
        has_required = False
        for col in required_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                print(f"  {col}: {non_null_count}/{len(df)} non-null")
                has_required = True
        
        if not has_required:
            print(f"  Error: Missing required columns: any of {required_columns}")
            return False
        
        for col in optional_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                print(f"  {col}: {non_null_count}/{len(df)} non-null (optional)")
            else:
                print(f"  Warning: Missing optional column: {col}")
        
        # Check author column format
        author_col = 'author2' if 'author2' in df.columns else 'author_old_paper'
        print(f"\n{author_col} column format check:")
        sample_authors = df[author_col].dropna().head(3)
        for i, authors in enumerate(sample_authors):
            print(f"  Sample {i+1}: {authors}")
            try:
                import ast
                parsed = ast.literal_eval(authors)
                print(f"    Parsed result: {len(parsed)} authors")
            except:
                print(f"    Warning: Cannot parse as Python list")
        
        return True
        
    except Exception as e:
        print(f"Error: Failed to read evaluation data: {e}")
        return False

if __name__ == "__main__":
    # Verify all paths
    paths_ok = verify_paths()
    
    # Check evaluation data format in detail
    format_ok = check_evaluation_data_format()
    
    if paths_ok and format_ok:
        print(f"\nAll verification passed! System ready to run.")
        print(f"\nNext steps:")
        print(f"   python bridger_baselines.py")
        print(f"   or")
        print(f"   python dygie_specter2_baseline/scripts/embedding_generator.py --evaluation-data /home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv")
    else:
        print(f"\nPlease fix the above issues first.")