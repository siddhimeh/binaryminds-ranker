"""
semantic_similarity.py

This module contains the logic to compute semantic similarity scores between the Job Description (JD)
and candidate profiles (concatenated summary and career history description) using
the sentence-transformer model 'all-MiniLM-L6-v2' running offline on CPU.
It is part of the Redrob x Hack2Skill Intelligent Candidate Discovery & Ranking pipeline.
"""

import os
import sys
import json
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer

# Ensure parent directory is in path so src/ imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def compute_semantic_scores(candidates, jd_text):
    """
    Computes cosine similarity scores between candidate text profiles and the Job Description.
    Runs offline on CPU in batches of 500.
    
    Parameters:
        candidates (list): List of candidate dictionaries.
        jd_text (str): String content of the Job Description.
        
    Returns:
        scores (dict): Dictionary mapping candidate_id to similarity_score (float, 0 to 1).
    """
    if not candidates:
        return {}
        
    # 1. Load sentence-transformer model on CPU
    # When run for the first time, it downloads the weights to the local cache.
    # Subsequent runs run fully offline.
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # 2. Encode Job Description into an embedding vector
    # normalize_embeddings=True makes the embedding unit length (L2 norm = 1.0)
    jd_embedding = model.encode(
        jd_text, 
        convert_to_numpy=True, 
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    candidate_texts = []
    candidate_ids = []
    
    # 3. Concatenate summary + career history descriptions for each candidate
    for candidate in candidates:
        candidate_ids.append(candidate.get("candidate_id"))
        
        profile = candidate.get("profile", {})
        summary = profile.get("summary") or ""
        
        career_history = candidate.get("career_history", [])
        career_desc_list = []
        for role in career_history:
            desc = role.get("description") or ""
            career_desc_list.append(desc)
            
        career_text = "\n".join(career_desc_list)
        combined_text = f"{summary}\n{career_text}".strip()
        candidate_texts.append(combined_text)
        
    # 4. Encode candidate texts in batches of 500 to conserve memory
    scores = {}
    batch_size = 500
    num_candidates = len(candidate_texts)
    
    for i in range(0, num_candidates, batch_size):
        batch_texts = candidate_texts[i : i + batch_size]
        batch_ids = candidate_ids[i : i + batch_size]
        
        # Encode batch with normalized embeddings
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Since embeddings are normalized, dot product is equivalent to cosine similarity
        similarities = np.dot(batch_embeddings, jd_embedding)
        
        # Map values to [0, 1] range by clipping at 0.0 and 1.0
        # Cosine similarity typically falls in [0, 1] for text comparisons, but handles [-1, 0) if any
        for cand_id, sim in zip(batch_ids, similarities):
            scores[cand_id] = float(max(0.0, min(1.0, sim)))
            
    return scores

def read_docx_text(file_path):
    """Helper function to extract text from a .docx file using python-docx."""
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

if __name__ == "__main__":
    print("Testing semantic_similarity.py against sample_candidates.json...")
    
    # Locate paths
    sample_path = os.path.join(
        project_root, 
        "India_runs_data_and_ai_challenge", 
        "sample_candidates.json"
    )
    jd_path = os.path.join(
        project_root,
        "India_runs_data_and_ai_challenge",
        "job_description.docx"
    )
    
    # Fallback to double-nested folder check
    if not os.path.exists(sample_path):
        sample_path = os.path.join(
            project_root,
            "India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "sample_candidates.json"
        )
        jd_path = os.path.join(
            project_root,
            "India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "job_description.docx"
        )
        
    try:
        # Load Job Description text
        print(f"Loading Job Description from {jd_path}...")
        jd_text = read_docx_text(jd_path)
        
        # Load sample candidates
        print(f"Loading candidates from {sample_path}...")
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_candidates = json.load(f)
            
        print("Computing semantic similarity scores for sample candidates...")
        # Run similarity scoring
        scores = compute_semantic_scores(sample_candidates, jd_text)
        
        print("\nSimilarity scores for the first 5 candidates:")
        for cand in sample_candidates[:5]:
            cid = cand.get("candidate_id")
            name = cand.get("profile", {}).get("anonymized_name")
            print(f"Candidate: {cid} ({name}) - Similarity Score: {scores.get(cid, 0.0):.4f}")
            
        print("\nTest passed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
