"""
feature_extractor.py

This module contains the logic to extract numerical features (scores between 0.0 and 1.0)
from a candidate's profile, career history, and Redrob behavioral signals.
It is part of the Redrob x Hack2Skill Intelligent Candidate Discovery & Ranking pipeline.
"""

import os
import json
import datetime

def extract_features(candidate, jd_skills):
    """
    Extracts numeric scores (0.0 to 1.0) for a candidate profile based on job description skills.
    
    Parameters:
        candidate (dict): The candidate profile dictionary.
        jd_skills (list): A list of skill strings required for the job.
        
    Returns:
        features (dict): A dictionary of numeric feature scores.
    """
    # 1. Skills Match Score
    # What % of jd_skills appear in candidate's skills list with proficiency = "advanced" or "expert"
    skills_match_score = 0.0
    candidate_skills = candidate.get("skills", [])
    if jd_skills and candidate_skills:
        # Build a set of advanced/expert skills the candidate has
        strong_candidate_skills = set()
        for skill in candidate_skills:
            name = skill.get("name")
            proficiency = skill.get("proficiency", "").lower()
            if name and proficiency in ("advanced", "expert"):
                strong_candidate_skills.add(name.strip().lower())
                
        # Count matches
        matched_count = 0
        for jd_skill in jd_skills:
            if jd_skill.strip().lower() in strong_candidate_skills:
                matched_count += 1
        skills_match_score = matched_count / len(jd_skills)
        
    # 2. Experience Score
    # Normalize years_of_experience to a 0-1 scale (cap at 15 years = 1.0)
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience", 0.0)
    if years_exp is None:
        years_exp = 0.0
    # Capping at 15 years and normalizing
    experience_score = min(15.0, max(0.0, float(years_exp))) / 15.0
    
    # 3. Seniority Score
    # Score based on most recent job title seniority
    # (intern=0.1, junior=0.3, mid=0.5, senior=0.7, lead/principal=0.9, director+=1.0)
    title = profile.get("current_title", "").strip().lower()
    
    # Fallback to the most recent role in career history if current title is empty
    if not title:
        career_history = candidate.get("career_history", [])
        for role in career_history:
            if role.get("is_current"):
                title = role.get("title", "").strip().lower()
                break
                
    seniority_score = 0.0
    if title:
        # Check from highest to lowest seniority to handle overlapping keywords correctly
        if any(keyword in title for keyword in ("director", "vp", "vice president", "cto", "cio", "cxo", "chief", "founder", "head of")):
            seniority_score = 1.0
        elif any(keyword in title for keyword in ("lead", "principal", "staff", "architect", "head")):
            seniority_score = 0.9
        elif any(keyword in title for keyword in ("senior", "sr")):
            seniority_score = 0.7
        elif any(keyword in title for keyword in ("junior", "jr", "associate", "entry", "trainee")):
            seniority_score = 0.3
        elif any(keyword in title for keyword in ("intern", "coop", "co-op")):
            seniority_score = 0.1
        else:
            # Default to mid-level
            seniority_score = 0.5
            
    # 4. Behavioral Score
    # Average of github_activity_score, interview_completion_rate, recruiter_response_rate
    # Normalize each from 0-100 to 0-1
    signals = candidate.get("redrob_signals", {})
    
    github = signals.get("github_activity_score")
    if github is None or github < 0:
        github_val = 0.0
    else:
        github_val = min(100.0, float(github)) / 100.0
        
    def clean_rate(rate):
        if rate is None or rate < 0:
            return 0.0
        # If rate is already in 0-1 range, keep it. If it is 0-100, normalize it.
        if rate > 1.0:
            return min(100.0, float(rate)) / 100.0
        return float(rate)
        
    interview_val = clean_rate(signals.get("interview_completion_rate"))
    recruiter_val = clean_rate(signals.get("recruiter_response_rate"))
    
    behavioral_score = (github_val + interview_val + recruiter_val) / 3.0
    
    # 5. Availability Score
    # Score based on:
    # - open_to_work_flag (1.0 if True)
    # - notice_period_days (shorter = better, cap at 90 days)
    # - last_active_date recency (active in last 30 days = 1.0, 90 days = 0.5, older = 0.1)
    open_to_work = signals.get("open_to_work_flag", False)
    open_to_work_score = 1.0 if open_to_work else 0.0
    
    notice_days = signals.get("notice_period_days")
    if notice_days is None:
        notice_score = 0.0
    else:
        # Shorter notice is better. Capped at 90 days.
        notice_score = max(0.0, (90.0 - float(notice_days)) / 90.0)
        
    last_active_str = signals.get("last_active_date")
    recency_score = 0.1 # Default to older
    if last_active_str:
        try:
            last_active = datetime.datetime.strptime(last_active_str, "%Y-%m-%d").date()
            # Reference date should handle year 2026 correctly
            ref_date = datetime.date.today()
            if ref_date < datetime.date(2026, 6, 16):
                ref_date = datetime.date(2026, 6, 16)
                
            days_diff = (ref_date - last_active).days
            if days_diff <= 30:
                recency_score = 1.0
            elif days_diff <= 90:
                recency_score = 0.5
        except (ValueError, TypeError):
            pass
            
    availability_score = (open_to_work_score + notice_score + recency_score) / 3.0
    
    return {
        "skills_match_score": skills_match_score,
        "experience_score": experience_score,
        "seniority_score": seniority_score,
        "behavioral_score": behavioral_score,
        "availability_score": availability_score
    }

if __name__ == "__main__":
    print("Testing feature_extractor.py against sample_candidates.json...")
    
    # Required skills from JD
    test_jd_skills = ["Python", "SQL", "Spark", "Airflow", "NLP", "Image Classification"]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
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
            
        print(f"Extracted feature vectors for the first 3 candidates:")
        for idx in range(min(3, len(sample_candidates))):
            cand = sample_candidates[idx]
            features = extract_features(cand, test_jd_skills)
            print(f"\nCandidate: {cand.get('candidate_id')} ({cand.get('profile', {}).get('anonymized_name')})")
            print(f"Current Title: {cand.get('profile', {}).get('current_title')}")
            print(f"Years of Experience: {cand.get('profile', {}).get('years_of_experience')}")
            print(f"Features: {json.dumps(features, indent=2)}")
        print("\nTest passed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
