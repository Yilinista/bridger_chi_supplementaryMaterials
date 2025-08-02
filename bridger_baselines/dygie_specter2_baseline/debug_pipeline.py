#!/usr/bin/env python3
"""
Debug script for DyGIE++ → SPECTER2 pipeline
Run each step manually to identify issues
"""

import json
import pandas as pd
import numpy as np
import tempfile
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_step1_data_loading(evaluation_data_path, paper_nodes_path, author_kg_path):
    """Debug: Check data loading"""
    print("=== Step 1: Data Loading Debug ===")
    
    # Load evaluation data
    df = pd.read_csv(evaluation_data_path)
    print(f"Evaluation CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Extract evaluation authors
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
    
    print(f"Total evaluation authors: {len(evaluation_authors)}")
    print(f"Sample authors: {list(evaluation_authors)[:5]}")
    
    # Load Graph-CoT data
    print(f"Loading papers from: {paper_nodes_path}")
    with open(paper_nodes_path, 'r') as f:
        papers = json.load(f)
    print(f"Total papers loaded: {len(papers)}")
    
    print(f"Loading author KG from: {author_kg_path}")
    with open(author_kg_path, 'r') as f:
        author_kg = json.load(f)
    print(f"Total authors in KG: {len(author_kg)}")
    
    # Check overlap
    authors_with_papers = set(evaluation_authors) & set(author_kg.keys())
    print(f"Evaluation authors with papers: {len(authors_with_papers)}")
    
    if len(authors_with_papers) == 0:
        print("ERROR: No evaluation authors found in author KG!")
        return None
    
    # Sample author-paper mapping
    sample_author = list(authors_with_papers)[0]
    paper_ids = author_kg[sample_author]
    print(f"Sample author {sample_author} has {len(paper_ids)} papers")
    
    sample_papers = []
    for paper_id in paper_ids[:3]:  # Check first 3 papers
        if paper_id in papers:
            features = papers[paper_id].get('features', {})
            title = features.get('Title', '').strip()
            abstract = features.get('Abstract', '').strip()
            sample_papers.append({'title': title, 'abstract': abstract})
    
    print(f"Sample papers for {sample_author}:")
    for i, paper in enumerate(sample_papers):
        print(f"  Paper {i+1}: {paper['title'][:100]}...")
    
    return evaluation_authors, papers, author_kg

def debug_step2_dygie_formatting(author_papers, output_file="debug_dygie_input.jsonl"):
    """Debug: Check DyGIE++ input formatting"""
    print("=== Step 2: DyGIE++ Formatting Debug ===")
    
    try:
        import spacy
        nlp = spacy.load("en_core_sci_sm")
        print("Loaded scientific spaCy model")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm") 
            print("Loaded basic English spaCy model")
        except OSError:
            print("ERROR: No spaCy model found!")
            return None
    
    # Format sample data
    sample_author = list(author_papers.keys())[0]
    sample_papers = author_papers[sample_author][:2]  # First 2 papers
    
    formatted_docs = []
    with open(output_file, 'w') as f:
        for i, paper in enumerate(sample_papers):
            title = paper['title'].strip()
            abstract = paper['abstract'].strip()
            text = f"{title}. {abstract}".strip() if abstract else title.strip()
            print(f"Processing paper {i+1}: {text[:100]}...")
            
            if not text:
                print(f"  WARNING: Empty text for paper {i+1}")
                continue
            
            spacy_doc = nlp(text)
            sentences = []
            for sent in spacy_doc.sents:
                tokens = [tok.text for tok in sent if not tok.is_space]
                if tokens:
                    sentences.append(tokens)
            
            if sentences:
                doc_data = {
                    "doc_key": f"{sample_author}_{i}",
                    "sentences": sentences,
                    "dataset": "scierc"
                }
                formatted_docs.append(doc_data)
                f.write(json.dumps(doc_data) + '\n')
                print(f"  Formatted into {len(sentences)} sentences")
    
    print(f"Created {len(formatted_docs)} formatted documents")
    print(f"Output saved to: {output_file}")
    return output_file

def debug_step3_dygie_prediction(input_file, dygie_model_path="models/dygiepp/pretrained_models/scierc"):
    """Debug: Run DyGIE++ prediction"""
    print("=== Step 3: DyGIE++ Prediction Debug ===")
    
    if not Path(dygie_model_path).exists():
        print(f"ERROR: DyGIE++ model not found at {dygie_model_path}")
        print("Please run setup script or download manually")
        return None
    
    output_file = input_file.replace('.jsonl', '_predictions.jsonl')
    
    cmd = [
        'python', 'predict.py',
        dygie_model_path,
        input_file,
        '--output-file', output_file
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        cwd = 'models/dygiepp' if Path('models/dygiepp').exists() else '.'
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=300)
        
        if result.returncode != 0:
            print(f"ERROR: DyGIE++ failed")
            print(f"STDERR: {result.stderr}")
            return None
        
        print("DyGIE++ prediction completed successfully")
        print(f"Output saved to: {output_file}")
        
        # Check output
        with open(output_file, 'r') as f:
            predictions = [json.loads(line) for line in f]
        
        print(f"Generated {len(predictions)} predictions")
        
        # Show sample prediction
        if predictions:
            sample_pred = predictions[0]
            print(f"Sample prediction keys: {sample_pred.keys()}")
            if 'ner' in sample_pred:
                total_entities = sum(len(entities) for entities in sample_pred['ner'])
                print(f"Total entities extracted: {total_entities}")
        
        return output_file
        
    except subprocess.TimeoutExpired:
        print("ERROR: DyGIE++ prediction timed out")
        return None
    except FileNotFoundError:
        print("ERROR: DyGIE++ predict.py not found")
        print("Make sure DyGIE++ is properly installed")
        return None

def debug_step4_term_extraction(predictions_file):
    """Debug: Parse DyGIE++ predictions"""
    print("=== Step 4: Term Extraction Debug ===")
    
    author_terms = {}
    
    with open(predictions_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line)
                doc_key = doc['doc_key']
                author_id = doc_key.split('_')[0]
                
                if author_id not in author_terms:
                    author_terms[author_id] = {'task': [], 'method': []}
                
                sentences = doc['sentences']
                for sent_idx, entities in enumerate(doc.get('ner', [])):
                    for entity in entities:
                        if len(entity) >= 3:
                            start_idx, end_idx, label = entity[:3]
                            
                            if sent_idx < len(sentences) and start_idx < len(sentences[sent_idx]):
                                term = ' '.join(sentences[sent_idx][start_idx:end_idx+1])
                                
                                if label in ['Task', 'OtherScientificTerm']:
                                    author_terms[author_id]['task'].append(term)
                                elif label in ['Method', 'Material', 'Metric']:
                                    author_terms[author_id]['method'].append(term)
                                
                                print(f"Extracted {label}: '{term}'")
            
            except json.JSONDecodeError:
                print(f"WARNING: Failed to parse line {line_num}")
                continue
    
    # Remove duplicates
    for author_id in author_terms:
        author_terms[author_id]['task'] = list(set(author_terms[author_id]['task']))
        author_terms[author_id]['method'] = list(set(author_terms[author_id]['method']))
    
    print(f"Extracted terms for {len(author_terms)} authors:")
    for author_id, terms in author_terms.items():
        print(f"  {author_id}: {len(terms['task'])} tasks, {len(terms['method'])} methods")
        if terms['task']:
            print(f"    Sample tasks: {terms['task'][:3]}")
        if terms['method']:
            print(f"    Sample methods: {terms['method'][:3]}")
    
    return author_terms

def debug_step5_specter2_embeddings(author_terms):
    """Debug: Generate SPECTER2 embeddings"""
    print("=== Step 5: SPECTER2 Embeddings Debug ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('allenai/specter2_base')
        print("SPECTER2 model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load SPECTER2 model: {e}")
        return None
    
    task_embeddings = {}
    method_embeddings = {}
    
    for author_id, terms in author_terms.items():
        print(f"Processing embeddings for author: {author_id}")
        
        # Task embeddings
        if terms['task']:
            try:
                task_embs = model.encode(terms['task'])
                print(f"  Task embeddings shape: {task_embs.shape}")
                
                if len(task_embs.shape) == 1:
                    task_embs = task_embs.reshape(1, -1)
                
                task_emb = np.mean(task_embs, axis=0)
                task_embeddings[author_id] = task_emb / np.linalg.norm(task_emb)
                print(f"  Final task embedding shape: {task_embeddings[author_id].shape}")
                
            except Exception as e:
                print(f"  ERROR generating task embeddings: {e}")
        
        # Method embeddings
        if terms['method']:
            try:
                method_embs = model.encode(terms['method'])
                print(f"  Method embeddings shape: {method_embs.shape}")
                
                if len(method_embs.shape) == 1:
                    method_embs = method_embs.reshape(1, -1)
                
                method_emb = np.mean(method_embs, axis=0)
                method_embeddings[author_id] = method_emb / np.linalg.norm(method_emb)
                print(f"  Final method embedding shape: {method_embeddings[author_id].shape}")
                
            except Exception as e:
                print(f"  ERROR generating method embeddings: {e}")
    
    print(f"Generated embeddings: {len(task_embeddings)} task, {len(method_embeddings)} method")
    return task_embeddings, method_embeddings

def main():
    """Run complete debug pipeline"""
    print("Starting DyGIE++ → SPECTER2 Pipeline Debug")
    print("=" * 50)
    
    # Set paths (modify these for your data)
    evaluation_data_path = "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
    paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
    author_kg_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json"
    
    # Step 1: Data loading
    result = debug_step1_data_loading(evaluation_data_path, paper_nodes_path, author_kg_path)
    if result is None:
        return
    evaluation_authors, papers, author_kg = result
    
    # Build author_papers for sample authors (first 3)
    sample_authors = list(evaluation_authors)[:3]
    author_papers = {}
    
    for author_id in sample_authors:
        if author_id in author_kg:
            paper_ids = author_kg[author_id]
            papers_data = []
            
            for paper_id in paper_ids:
                if paper_id in papers:
                    features = papers[paper_id].get('features', {})
                    title = features.get('Title', '').strip()
                    abstract = features.get('Abstract', '').strip()
                    
                    if title:
                        papers_data.append({
                            'title': title,
                            'abstract': abstract if abstract else ''
                        })
            
            if papers_data:
                author_papers[author_id] = papers_data
    
    print(f"Sample processing {len(author_papers)} authors")
    
    # Step 2: DyGIE++ formatting
    formatted_file = debug_step2_dygie_formatting(author_papers)
    if formatted_file is None:
        return
    
    # Step 3: DyGIE++ prediction
    predictions_file = debug_step3_dygie_prediction(formatted_file)
    if predictions_file is None:
        return
    
    # Step 4: Term extraction
    author_terms = debug_step4_term_extraction(predictions_file)
    if not author_terms:
        return
    
    # Step 5: SPECTER2 embeddings
    embeddings = debug_step5_specter2_embeddings(author_terms)
    if embeddings is None:
        return
    
    task_embeddings, method_embeddings = embeddings
    
    print("\n" + "=" * 50)
    print("DEBUG PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Final results:")
    print(f"  Authors processed: {len(author_papers)}")
    print(f"  Task embeddings: {len(task_embeddings)}")
    print(f"  Method embeddings: {len(method_embeddings)}")
    
    # Clean up
    Path(formatted_file).unlink(missing_ok=True)
    Path(predictions_file).unlink(missing_ok=True)

if __name__ == "__main__":
    main()