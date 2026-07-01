"""
ranker.py

This is the main entry point that runs the candidate ranking pipeline end-to-end.
It streams candidates from the dataset, filters out honeypots, scores candidates in memory-efficient batches,
computes semantic similarity on the shortlist, ranks, outputs the final CSV, and runs the validator.
"""

import os
import sys
import json
import time
import csv
import subprocess
from tqdm import tqdm

# Adjust sys.path so all src/ imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from our modules
try:
    from src.honeypot_filter import filter_honeypots
    from src.scorer import score_candidates
    from src.semantic_similarity import compute_semantic_scores, read_docx_text
except ImportError:
    from honeypot_filter import filter_honeypots
    from scorer import score_candidates
    from semantic_similarity import compute_semantic_scores, read_docx_text

# Define the required skills list from the Job Description
JD_SKILLS = [
    "Python", "embeddings", "sentence-transformers", "vector databases", 
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch", "Elasticsearch", 
    "FAISS", "NDCG", "MRR", "MAP", "fine-tuning", "LoRA", "QLoRA", "PEFT", 
    "learning-to-rank", "XGBoost", "distributed systems", "NLP"
]

def find_file(relative_name):
    """Robustly find a file by checking standard locations relative to project root."""
    paths_to_try = [
        os.path.join(project_root, relative_name),
        os.path.join(project_root, "India_runs_data_and_ai_challenge", relative_name),
        os.path.join(project_root, "India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", relative_name)
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            return p
    # Fallback to the relative_name if not found
    return os.path.join(project_root, relative_name)

def main():
    start_time = time.time()
    print("=" * 60)
    print("BinaryMinds — Candidate Ranker End-to-End Pipeline Starting")
    print("=" * 60)
    
    # Locate files
    candidates_file = find_file("candidates.jsonl")
    jd_file = find_file("job_description.docx")
    validator_file = find_file("validate_submission.py")
    output_dir = os.path.join(project_root, "output")
    output_file = os.path.join(output_dir, "submission.csv")
    
    print(f"Candidates dataset: {candidates_file}")
    print(f"Job Description: {jd_file}")
    print(f"Output destination: {output_file}")
    
    if not os.path.exists(candidates_file):
        print(f"Error: Candidates file not found at {candidates_file}")
        sys.exit(1)
        
    # Step 1: Stream candidates in batches of 5000 and keep top 2000
    print("\n--- STEP 1 & 2 & 3: Streaming, Filtering, and Initial Scoring ---")
    batch_size = 5000
    running_top_candidates = [] # Holds dicts of {"scored_info": ..., "candidate_dict": ...}
    
    current_batch = []
    total_processed = 0
    total_honeypots_removed = 0
    
    # Estimate total lines for progress bar if candidates.jsonl size is known
    # Typical candidates.jsonl file has ~100,000 lines
    file_size_mb = os.path.getsize(candidates_file) / (1024 * 1024)
    print(f"Streaming {file_size_mb:.1f} MB candidates file...")
    
    with open(candidates_file, "r", encoding="utf-8") as f:
        # We read line by line to keep RAM usage extremely low
        for line in tqdm(f, desc="Processing candidate streams", unit=" lines"):
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
                current_batch.append(candidate)
            except json.JSONDecodeError:
                continue
                
            if len(current_batch) >= batch_size:
                # Process the batch
                total_processed += len(current_batch)
                
                # Filter honeypots
                clean_candidates, removed = filter_honeypots(current_batch)
                total_honeypots_removed += removed
                
                # Initial score with default semantic score of 0.5
                scored_batch = score_candidates(clean_candidates, JD_SKILLS)
                
                # Map candidate_id -> candidate_dict for clean candidates in this batch
                id_to_cand = {c["candidate_id"]: c for c in clean_candidates}
                
                # Merge into running pool
                combined = list(running_top_candidates)
                for item in scored_batch:
                    cid = item["candidate_id"]
                    combined.append({
                        "scored_info": item,
                        "candidate_dict": id_to_cand[cid]
                    })
                    
                # Sort running pool: final_score descending, candidate_id ascending
                combined.sort(key=lambda x: (-x["scored_info"]["final_score"], x["scored_info"]["candidate_id"]))
                
                # Prune to top 2000 to keep peak RAM well under limits
                running_top_candidates = combined[:2000]
                
                # Reset batch
                current_batch = []
                
        # Process any remaining candidates in the last batch
        if current_batch:
            total_processed += len(current_batch)
            clean_candidates, removed = filter_honeypots(current_batch)
            total_honeypots_removed += removed
            
            scored_batch = score_candidates(clean_candidates, JD_SKILLS)
            id_to_cand = {c["candidate_id"]: c for c in clean_candidates}
            
            combined = list(running_top_candidates)
            for item in scored_batch:
                cid = item["candidate_id"]
                combined.append({
                    "scored_info": item,
                    "candidate_dict": id_to_cand[cid]
                })
                
            combined.sort(key=lambda x: (-x["scored_info"]["final_score"], x["scored_info"]["candidate_id"]))
            running_top_candidates = combined[:2000]
            
    print(f"\nStreaming completed.")
    print(f"Total candidates read from file: {total_processed}")
    print(f"Total honeypots filtered: {total_honeypots_removed}")
    print(f"Shortlisted candidates retained in pool: {len(running_top_candidates)}")
    
    # Step 4: Semantic scoring on shortlist only
    print("\n--- STEP 4: Computing Semantic Similarity for Shortlist ---")
    if not running_top_candidates:
        print("Error: No candidates remained after honeypot filtering.")
        sys.exit(1)
        
    print(f"Loading Job Description text from {jd_file}...")
    jd_text = read_docx_text(jd_file)
    
    shortlist_candidates = [x["candidate_dict"] for x in running_top_candidates]
    print(f"Running sentence-transformer similarity on the top {len(shortlist_candidates)} candidates...")
    
    # Compute similarity scores
    semantic_scores = compute_semantic_scores(shortlist_candidates, jd_text)
    
    # Step 5: Final Ranking
    print("\n--- STEP 5: Final Scoring & Ranking ---")
    # Re-run scorer with computed semantic scores (this replaces the default 0.5)
    final_ranked_results = score_candidates(shortlist_candidates, JD_SKILLS, semantic_scores)
    
    # Take top 100
    top_100 = final_ranked_results[:100]
    print(f"Final ranking computed. Top Candidate ID: {top_100[0]['candidate_id']} (Score: {top_100[0]['final_score']:.4f})")
    
    # Step 6: Output
    print(f"\n--- STEP 6: Writing Output to CSV ---")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write headers as required by validate_submission.py
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, res in enumerate(top_100, 1):
            writer.writerow([
                res["candidate_id"],
                rank,
                round(res["final_score"], 6),
                res["reasoning"]
            ])
            
    print(f"Successfully wrote 100 rows to {output_file}")
    
    # Step 7: Validate
    print("\n--- STEP 7: Validating Submission Format ---")
    if os.path.exists(validator_file):
        print(f"Running validator: {validator_file}...")
        try:
            # We run python validator_file output_file
            res = subprocess.run(
                [sys.executable, validator_file, output_file], 
                capture_output=True, 
                text=True
            )
            print("Validator Output:")
            print(res.stdout)
            if res.stderr:
                print("Validator Error Output:", res.stderr)
                
            if res.returncode == 0:
                print("VALIDATION STATUS: PASSED")
            else:
                print("VALIDATION STATUS: FAILED")
        except Exception as e:
            print(f"Error running validator subprocess: {e}")
    else:
        print(f"Warning: Validator script not found at {validator_file}. Checking locally...")
        
    # Step 8: Timer
    runtime = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"BinaryMinds Pipeline Complete.")
    print(f"Total Runtime: {runtime:.2f} seconds")
    if runtime < 300:
        print(f"CONSTRAINT MET: Runtime under 300 seconds (5 mins)")
    else:
        print(f"WARNING: Runtime exceeded 300 seconds limit!")
    print("=" * 60)

if __name__ == "__main__":
    main()
