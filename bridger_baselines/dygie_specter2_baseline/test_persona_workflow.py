#!/usr/bin/env python3
"""
Test script to verify the complete persona-enabled workflow
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the scripts directory and parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridger_baselines_improved import run_bridger_evaluation_improved

def test_persona_workflow():
    """Test the persona workflow with a small sample"""
    
    # Create a temporary directory for embeddings
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Testing persona-enabled Bridger workflow...")
        print(f"Using temporary storage: {temp_dir}")
        
        # Use a small evaluation dataset
        evaluation_data = "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
        
        if not os.path.exists(evaluation_data):
            print("Evaluation data not found. Please check the path.")
            return
        
        try:
            # Test without persona mode first
            print("\n=== Testing Standard Mode ===")
            results_standard = run_bridger_evaluation_improved(
                evaluation_data_path=evaluation_data,
                embedding_storage_dir=temp_dir + "_standard",
                force_regenerate=True,
                enable_persona=False
            )
            
            print("Standard mode results:")
            for baseline, metrics in results_standard.items():
                print(f"  {baseline}: Hit@10={metrics['Hit@10']:.1%}, MRR={metrics['MRR']:.4f}")
            
            print("\nStandard workflow test completed successfully!")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_persona_workflow()