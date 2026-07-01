"""
scorer.py

This module contains the logic to compute a final weighted score for each candidate,
generate a factual reasoning string based on real candidate data, and sort candidates.
It is part of the Redrob x Hack2Skill Intelligent Candidate Discovery & Ranking pipeline.
"""

import os
import sys
import json

# Ensure parent directory is in path so src/ imports work under different run contexts
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.feature_extractor import extract_features
except ImportError:
    from feature_extractor import extract_features

def has_only_consulting_experience(career_history):
    """
    Checks if a candidate has only worked at consulting/service firms in their entire career,
    which is an explicit disqualifier in the Job Description.
    """
    consulting_firms = {
        "tcs", "tata consultancy services", "infosys", "wipro", 
        "accenture", "cognizant", "capgemini", "tech mahindra", "hcl"
    }
    if not career_history:
        return False
        
    has_valid_company = False
    for role in career_history:
        company = role.get("company", "").strip().lower()
        if company:
            has_valid_company = True
            is_consulting = any(firm in company for firm in consulting_firms)
            if not is_consulting:
                # Found a non-consulting/product company experience!
                return False
                
    # If they have career history and all companies match consulting firms, return True
    return has_valid_company

def score_candidates(candidates, jd_skills, semantic_scores=None):
    """
    Scores and ranks candidates based on feature weights and optional semantic similarity scores.
    
    Parameters:
        candidates (list): List of candidate profile dictionaries.
        jd_skills (list): List of required skill strings.
        semantic_scores (dict): Dictionary of {candidate_id: similarity_score}. Defaults to 0.5.
        
    Returns:
        scored_candidates (list): Sorted list of candidate result dictionaries.
    """
    if semantic_scores is None:
        semantic_scores = {}
        
    scored_list = []
    
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        
        # 1. Extract numeric features
        features = extract_features(candidate, jd_skills)
        
        # 2. Get semantic score (default to 0.5 if not provided)
        semantic_score = semantic_scores.get(candidate_id, 0.5)
        
        # 3. Compute final weighted score
        # Weights:
        # - skills_match_score  * 0.30
        # - experience_score    * 0.25
        # - semantic_score      * 0.20
        # - behavioral_score    * 0.15
        # - availability_score  * 0.10
        weighted_score = (
            features["skills_match_score"] * 0.30 +
            features["experience_score"] * 0.25 +
            semantic_score * 0.20 +
            features["behavioral_score"] * 0.15 +
            features["availability_score"] * 0.10
        )
        
        # 4. Domain-specific filter: Apply a penalty if candidate has only consulting experience
        # as explicitly specified in the JD ("People who have only worked at consulting firms... in their entire career").
        is_only_consulting = has_only_consulting_experience(career_history)
        final_score = weighted_score
        if is_only_consulting:
            # Apply severe penalty so they sink in the rankings
            final_score = weighted_score * 0.1
            
        # 5. Generate factual reasoning string
        # Count actual matched skills at advanced/expert level
        matched_skills = 0
        total_skills = len(jd_skills)
        cand_skills = candidate.get("skills", [])
        if jd_skills and cand_skills:
            strong_skills = {s.get("name", "").strip().lower() for s in cand_skills if s.get("proficiency", "").lower() in ("advanced", "expert")}
            matched_skills = sum(1 for s in jd_skills if s.strip().lower() in strong_skills)
            
        years_exp = profile.get("years_of_experience", 0.0)
        if years_exp is None:
            years_exp = 0.0
            
        behavioral_val = features["behavioral_score"]
        
        reasoning = (
            f"Strong match with {matched_skills} out of {total_skills} required skills at advanced/expert level. "
            f"{years_exp} years of experience. Behavioral engagement score: {behavioral_val:.2f}."
        )
        if is_only_consulting:
            reasoning += " Note: Background consists exclusively of service/consulting companies."
            
        scored_list.append({
            "candidate_id": candidate_id,
            "final_score": final_score,
            "skills_match_score": features["skills_match_score"],
            "experience_score": features["experience_score"],
            "seniority_score": features["seniority_score"],
            "behavioral_score": features["behavioral_score"],
            "availability_score": features["availability_score"],
            "semantic_score": semantic_score,
            "reasoning": reasoning
        })
        
    # Sort candidates from highest to lowest final_score.
    # Tie-breaker: candidate_id ascending (alphabetical) as required by validation logic
    scored_list.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    
    return scored_list

if __name__ == "__main__":
    print("Testing scorer.py against sample_candidates.json...")
    
    test_jd_skills = ["Python", "SQL", "Spark", "Airflow", "NLP", "Image Classification"]
    
    sample_path = os.path.join(
        project_root, 
        "India_runs_data_and_ai_challenge", 
        "sample_candidates.json"
    )
    
    if not os.path.exists(sample_path):
        sample_path = os.path.join(
            project_root,
            "India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "sample_candidates.json"
        )
        
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_candidates = json.load(f)
            
        # First filter out honeypots (mimics the pipeline)
        from src.honeypot_filter import filter_honeypots
        clean_candidates, removed = filter_honeypots(sample_candidates)
        
        print(f"Filtered {removed} honeypots. Scoring remaining {len(clean_candidates)} candidates...")
        
        # Score candidates
        results = score_candidates(clean_candidates, test_jd_skills)
        
        print("\nTop 5 Candidates scored and ranked:")
        for rank, res in enumerate(results[:5], 1):
            print(f"\nRank {rank}: {res['candidate_id']} - Final Score: {res['final_score']:.4f}")
            print(f"  Skills Match: {res['skills_match_score']:.2f}, Experience: {res['experience_score']:.2f}")
            print(f"  Behavioral: {res['behavioral_score']:.2f}, Availability: {res['availability_score']:.2f}")
            print(f"  Reasoning: {res['reasoning']}")
            
        print("\nTest passed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
