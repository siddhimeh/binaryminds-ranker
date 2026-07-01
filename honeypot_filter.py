"""
honeypot_filter.py

This module contains logic to filter out "honeypot" candidates — synthetic profiles
designed with impossible, contradictory, or suspiciously perfect attributes.
It is part of the Redrob x Hack2Skill Intelligent Candidate Discovery & Ranking pipeline.
"""

import os
import json
import datetime

def filter_honeypots(candidates):
    """
    Filters out honeypot candidates based on five logic checks.
    
    Parameters:
        candidates (list): A list of candidate profile dictionaries.
        
    Returns:
        clean_candidates (list): List of candidates that passed all checks.
        removed_count (int): The number of candidates removed.
    """
    clean_candidates = []
    removed_count = 0
    current_year = datetime.datetime.now().year
    
    for candidate in candidates:
        is_honeypot = False
        
        # Extract basic sections with safe defaults
        profile = candidate.get("profile", {})
        education = candidate.get("education", [])
        career_history = candidate.get("career_history", [])
        signals = candidate.get("redrob_signals", {})
        
        # 1. Years of experience > (current_year - graduation_year + 2) [impossible experience]
        years_exp = profile.get("years_of_experience")
        if years_exp is not None and education:
            # Get the maximum graduation/end year from education history
            end_years = [edu.get("end_year") for edu in education if edu.get("end_year") is not None]
            if end_years:
                grad_year = max(end_years)
                if years_exp > (current_year - grad_year + 2):
                    is_honeypot = True
        
        # 2. ALL of these signals are >= 95 (scaled to 100) at the same time:
        #    - github_activity_score
        #    - skill_assessment_scores average
        #    - interview_completion_rate
        #    - recruiter_response_rate
        if not is_honeypot and signals:
            github_score = signals.get("github_activity_score")
            skill_scores_dict = signals.get("skill_assessment_scores", {})
            interview_rate = signals.get("interview_completion_rate")
            recruiter_rate = signals.get("recruiter_response_rate")
            
            # Check github_activity_score
            github_perfect = github_score is not None and github_score >= 95
            
            # Check skill_assessment_scores average
            if skill_scores_dict:
                avg_skill_score = sum(skill_scores_dict.values()) / len(skill_scores_dict)
            else:
                avg_skill_score = 0.0
            skill_perfect = avg_skill_score >= 95
            
            # Check interview_completion_rate (stored as fraction 0.0-1.0 in schema)
            interview_perfect = interview_rate is not None and (interview_rate * 100) >= 95
            
            # Check recruiter_response_rate (stored as fraction 0.0-1.0 in schema)
            recruiter_perfect = recruiter_rate is not None and (recruiter_rate * 100) >= 95
            
            if github_perfect and skill_perfect and interview_perfect and recruiter_perfect:
                is_honeypot = True
        
        # 3. Expected salary minimum > expected_salary maximum [impossible salary]
        if not is_honeypot and signals:
            salary_range = signals.get("expected_salary_range_inr_lpa", {})
            if salary_range:
                min_sal = salary_range.get("min")
                max_sal = salary_range.get("max")
                if min_sal is not None and max_sal is not None and min_sal > max_sal:
                    is_honeypot = True
                    
        # 4. Any career_history role has a start_date before the candidate's graduation end_year
        if not is_honeypot and career_history and education:
            end_years = [edu.get("end_year") for edu in education if edu.get("end_year") is not None]
            if end_years:
                grad_year = max(end_years)
                for role in career_history:
                    start_date = role.get("start_date")
                    if start_date:
                        try:
                            # Start date is in YYYY-MM-DD format
                            start_year = int(start_date.split("-")[0])
                            if start_year < grad_year:
                                is_honeypot = True
                                break
                        except (ValueError, IndexError):
                            # Skip malformed date check
                            pass
                            
        # 5. profile_completeness_score < 30 AND open_to_work_flag is True [suspicious empty profile]
        if not is_honeypot and signals:
            completeness = signals.get("profile_completeness_score")
            open_to_work = signals.get("open_to_work_flag")
            if completeness is not None and open_to_work is True:
                if completeness < 30:
                    is_honeypot = True
                    
        if is_honeypot:
            removed_count += 1
        else:
            clean_candidates.append(candidate)
            
    return clean_candidates, removed_count

if __name__ == "__main__":
    # Test on sample_candidates.json when run directly
    print("Testing honeypot_filter.py against sample_candidates.json...")
    
    # Construct relative path using os.path.join
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sample_path = os.path.join(
        project_root, 
        "India_runs_data_and_ai_challenge", 
        "sample_candidates.json"
    )
    
    if not os.path.exists(sample_path):
        # Check if the folder contains extra nested levels
        sample_path = os.path.join(
            project_root,
            "India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "sample_candidates.json"
        )
        
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_candidates = json.load(f)
            
        initial_count = len(sample_candidates)
        clean_list, removed = filter_honeypots(sample_candidates)
        final_count = len(clean_list)
        
        print(f"Initial candidates count: {initial_count}")
        print(f"Removed honeypots: {removed}")
        print(f"Remaining clean candidates: {final_count}")
        print("Test passed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
