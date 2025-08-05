#!/usr/bin/env python3
"""
Improved Bridger Baselines using DyGIE++ + SPECTER2

This is an enhanced version of the original Bridger baselines that uses:
- DyGIE++ for high-quality scientific term extraction
- SPECTER2 for semantic embeddings
- Precomputed embeddings for fast evaluation

Usage:
    python bridger_baselines_improved.py --evaluation-data path/to/data.csv
"""

import json
import pandas as pd
import numpy as np
import logging
import pickle
from typing import Dict, List, Tuple
from pathlib import Path
import argparse

# Import the original baseline classes
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bridger_baselines import BridgerBaselines, evaluate_baselines

# Import the embedding manager
from embedding_generator import BridgerEmbeddingManager, load_author_paper_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_bridger_evaluation_improved(
    evaluation_data_path: str,
    paper_nodes_path: str = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json",
    author_kg_path: str = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json",
    embedding_storage_dir: str = "./bridger_embeddings",
    force_regenerate: bool = False,
    enable_persona: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Run Bridger baseline evaluation using DyGIE++ + SPECTER2 embeddings.
    
    Args:
        evaluation_data_path: Path to BetterTeaming CSV file
        paper_nodes_path: Path to paper nodes JSON
        author_kg_path: Path to author knowledge graph JSON
        embedding_storage_dir: Directory containing precomputed embeddings
        force_regenerate: Whether to force regeneration of embeddings
        enable_persona: Whether to enable persona mode during generation
        
    Returns:
        Evaluation results for ST and sTdM baselines
    """
    
    # Initialize embedding manager
    embedding_manager = BridgerEmbeddingManager(embedding_storage_dir)
    
    # Load evaluation authors from CSV
    logger.info(f"Loading evaluation data from {evaluation_data_path}")
    df = pd.read_csv(evaluation_data_path)
    evaluation_authors = set()
    
    for _, row in df.iterrows():
        try:
            import ast
            team_authors = ast.literal_eval(row['author2'])
            evaluation_authors.update(team_authors)
            
            if pd.notna(row['ground_truth_authors']):
                gt_authors = row['ground_truth_authors'].split('|')
                evaluation_authors.update([a.strip() for a in gt_authors])
        except:
            continue
    
    logger.info(f"Found {len(evaluation_authors)} unique evaluation authors")
    
    # Try to load precomputed embeddings
    try:
        if not force_regenerate:
            logger.info("Attempting to load precomputed embeddings...")
            task_embeddings, method_embeddings = embedding_manager.load_embeddings()
            
            # Check if we have embeddings for all evaluation authors
            available_authors = set(task_embeddings.keys()) | set(method_embeddings.keys())
            missing_authors = evaluation_authors - available_authors
            
            if missing_authors:
                logger.warning(f"Missing embeddings for {len(missing_authors)} authors")
                if len(missing_authors) / len(evaluation_authors) > 0.1:  # More than 10% missing
                    logger.info("Too many missing authors, regenerating embeddings...")
                    raise FileNotFoundError("Incomplete embeddings")
                else:
                    logger.info(f"Proceeding with {len(available_authors)} available authors")
            
        else:
            raise FileNotFoundError("Force regeneration requested")
            
    except FileNotFoundError:
        # Need to generate embeddings
        logger.info("Generating embeddings (this may take a while)...")
        
        # Load author-paper data
        author_papers = load_author_paper_data(paper_nodes_path, author_kg_path, evaluation_authors)
        
        # Generate and store embeddings
        task_embeddings, method_embeddings = embedding_manager.generate_and_store_embeddings(
            author_papers, 
            force_regenerate=force_regenerate,
            enable_persona=enable_persona
        )
    
    # Filter embeddings to evaluation authors only (handle both persona and author keys)
    if enable_persona or any('-' in k for k in task_embeddings.keys()):
        # Persona mode: filter by author prefix
        eval_task_embeddings = {k: v for k, v in task_embeddings.items() if k.split('-')[0] in evaluation_authors}
        eval_method_embeddings = {k: v for k, v in method_embeddings.items() if k.split('-')[0] in evaluation_authors}
    else:
        # Standard mode: filter by author ID
        eval_task_embeddings = {k: v for k, v in task_embeddings.items() if k in evaluation_authors}
        eval_method_embeddings = {k: v for k, v in method_embeddings.items() if k in evaluation_authors}
    
    logger.info(f"Using embeddings: {len(eval_task_embeddings)} task, {len(eval_method_embeddings)} method")
    
    # Check if we're in persona mode
    persona_mode = enable_persona or any('-' in k for k in eval_task_embeddings.keys())
    author_personas = {}
    
    if persona_mode:
        try:
            # Try to load persona data
            persona_pickle_path = embedding_manager.storage_dir / "persona_embeddings.pkl"
            
            if persona_pickle_path.exists():
                with open(persona_pickle_path, 'rb') as f:
                    persona_data = pickle.load(f)
                
                if "author_personas" in persona_data:
                    author_personas = persona_data["author_personas"]
                    logger.info("Loaded persona mode embeddings")
                else:
                    logger.info("Persona mode enabled but no persona data found")
        except Exception as e:
            logger.warning(f"Failed to load persona data: {e}")
    
    logger.info(f"Persona mode: {'enabled' if persona_mode else 'disabled'}")
    
    # Initialize baselines with embeddings
    baselines = BridgerBaselines(eval_task_embeddings, eval_method_embeddings, 
                                persona_mode=persona_mode, author_personas=author_personas)
    
    # Run evaluation
    logger.info("Running baseline evaluation...")
    results = evaluate_baselines(baselines, evaluation_data_path)
    
    return results


def compare_with_original_baseline(evaluation_data_path: str, **kwargs):
    """Compare improved baseline with original random-vector baseline"""
    
    logger.info("Running comparison between original and improved baselines...")
    
    # Run improved baseline
    logger.info("\n=== Running Improved Baseline (DyGIE++ + SPECTER2) ===")
    improved_results = run_bridger_evaluation_improved(evaluation_data_path, **kwargs)
    
    # Run original baseline for comparison
    logger.info("\n=== Running Original Baseline (Random Vectors) ===")
    try:
        from bridger_baselines import run_bridger_evaluation
        original_results = run_bridger_evaluation(evaluation_data_path)
    except Exception as e:
        logger.warning(f"Could not run original baseline: {e}")
        original_results = None
    
    # Display comparison
    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80)
    
    methods = ['ST', 'sTdM']
    metrics = ['Hit@10', 'MRR']
    
    for method in methods:
        print(f"\n{method} Baseline:")
        print("-" * 40)
        
        for metric in metrics:
            if original_results:
                original_val = original_results[method][metric]
                improved_val = improved_results[method][metric]
                improvement = ((improved_val - original_val) / original_val * 100) if original_val > 0 else 0
                
                print(f"  {metric}:")
                print(f"    Original:  {original_val:.4f}")
                print(f"    Improved:  {improved_val:.4f}")
                print(f"    Change:    {improvement:+.1f}%")
            else:
                print(f"  {metric}: {improved_results[method][metric]:.4f}")
        
        print(f"  Queries: {improved_results[method]['Queries']}")
    
    return improved_results, original_results


def batch_evaluation(evaluation_files: List[str], **kwargs):
    """Run evaluation on multiple datasets"""
    
    all_results = {}
    
    for eval_file in evaluation_files:
        logger.info(f"\n=== Evaluating on {eval_file} ===")
        
        try:
            results = run_bridger_evaluation_improved(eval_file, **kwargs)
            all_results[eval_file] = results
            
            # Show results for this file
            print(f"\nResults for {Path(eval_file).name}:")
            for baseline, metrics in results.items():
                print(f"  {baseline}: Hit@10={metrics['Hit@10']:.1%}, MRR={metrics['MRR']:.4f}")
                
        except Exception as e:
            logger.error(f"Failed to evaluate {eval_file}: {e}")
            all_results[eval_file] = None
    
    return all_results


def main():
    """Main function for running improved Bridger baselines"""
    
    parser = argparse.ArgumentParser(description="Run improved Bridger baselines with DyGIE++ + SPECTER2")
    
    parser.add_argument(
        "--evaluation-data",
        required=True,
        help="Path to evaluation CSV file"
    )
    parser.add_argument(
        "--paper-nodes",
        default="/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json",
        help="Path to paper nodes JSON file"
    )
    parser.add_argument(
        "--author-kg", 
        default="/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json",
        help="Path to author knowledge graph JSON file"
    )
    parser.add_argument(
        "--embedding-dir",
        default="./bridger_embeddings",
        help="Directory containing precomputed embeddings"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of embeddings"
    )
    parser.add_argument(
        "--compare-original",
        action="store_true",
        help="Compare with original random-vector baseline"
    )
    parser.add_argument(
        "--batch-mode",
        nargs="+",
        help="Run evaluation on multiple CSV files"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show embedding statistics"
    )
    parser.add_argument(
        "--enable-persona",
        action="store_true",
        help="Enable persona mode during embedding generation"
    )
    
    args = parser.parse_args()
    
    # Show embedding statistics if requested
    if args.stats_only:
        embedding_manager = BridgerEmbeddingManager(args.embedding_dir)
        stats = embedding_manager.get_embedding_stats()
        
        print("Embedding Statistics:")
        print("=" * 40)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Prepare common arguments
    common_args = {
        "paper_nodes_path": args.paper_nodes,
        "author_kg_path": args.author_kg,
        "embedding_storage_dir": args.embedding_dir,
        "force_regenerate": args.force_regenerate,
        "enable_persona": args.enable_persona
    }
    
    # Run evaluation
    if args.batch_mode:
        # Batch mode: evaluate multiple files
        evaluation_files = args.batch_mode
        if args.evaluation_data not in evaluation_files:
            evaluation_files.append(args.evaluation_data)
        
        all_results = batch_evaluation(evaluation_files, **common_args)
        
        # Summary
        print("\n" + "="*80)
        print("BATCH EVALUATION SUMMARY") 
        print("="*80)
        
        for file_path, results in all_results.items():
            if results:
                print(f"\n{Path(file_path).name}:")
                for baseline, metrics in results.items():
                    print(f"  {baseline}: Hit@10={metrics['Hit@10']:.1%}, MRR={metrics['MRR']:.4f}")
            else:
                print(f"\n{Path(file_path).name}: FAILED")
    
    elif args.compare_original:
        # Compare with original baseline
        improved_results, original_results = compare_with_original_baseline(
            args.evaluation_data, **common_args
        )
    
    else:
        # Single evaluation
        results = run_bridger_evaluation_improved(args.evaluation_data, **common_args)
        
        # Display results
        print("\n" + "="*70)
        print("IMPROVED BRIDGER BASELINE RESULTS (DyGIE++ + SPECTER2)")
        print("="*70)
        
        for baseline, metrics in results.items():
            print(f"\n{baseline} Baseline:")
            for metric, value in metrics.items():
                if metric != 'Queries':
                    print(f"  {metric}: {value:.4f}")
            print(f"  Queries: {metrics['Queries']}")
        
        print(f"\nEmbedding source: {args.embedding_dir}")
        
        # Show embedding stats
        embedding_manager = BridgerEmbeddingManager(args.embedding_dir)
        stats = embedding_manager.get_embedding_stats()
        print(f"Embedding info: {stats.get('task_authors', 'N/A')} task authors, "
              f"{stats.get('method_authors', 'N/A')} method authors, "
              f"{stats.get('storage_size_mb', 'N/A')} MB")


if __name__ == "__main__":
    main()