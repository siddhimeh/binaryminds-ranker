"""
streamlit_app.py

This is a Streamlit web application providing a premium, interactive dashboard
for judges to upload candidate profiles (in JSON format), filter out honeypots,
rank candidates, and download the resulting CSV.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# pyrefly: ignore [missing-import]
from honeypot_filter import filter_honeypots
# pyrefly: ignore [missing-import]
from scorer import score_candidates

# Define required skills from JD
JD_SKILLS = [
    "Python", "embeddings", "sentence-transformers", "vector databases", 
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch", "Elasticsearch", 
    "FAISS", "NDCG", "MRR", "MAP", "fine-tuning", "LoRA", "QLoRA", "PEFT", 
    "learning-to-rank", "XGBoost", "distributed systems", "NLP"
]

# Set page configuration with a premium dark theme feel
st.set_page_config(
    page_title="BinaryMinds — Intelligent Candidate Ranker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-themed, glassmorphic visual excellence
st.markdown("""
<style>
    /* Import modern Google font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main background styling */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
        color: #e2e8f0;
    }
    
    /* Premium Header Title Gradient */
    .gradient-text {
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Card design */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Weight breakdown visual blocks */
    .weight-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #60a5fa;
    }
    .weight-name {
        font-weight: 600;
        color: #e2e8f0;
    }
    .weight-value {
        font-weight: 700;
        color: #60a5fa;
    }
    
    /* Metric container styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(96, 165, 250, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .metric-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #a78bfa;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    /* Tables styling */
    .stTable {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: Weight Breakdown ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=70)
    st.markdown("<h3 style='margin-top:0.5rem;'>Ranking Weights</h3>", unsafe_allow_html=True)
    st.markdown("These weights represent the relative importance of each dimension in the ranking model:")
    
    weights = [
        ("Skills Match Score", "30%", "#60a5fa"),
        ("Experience Score", "25%", "#34d399"),
        ("Semantic Similarity", "20%", "#a78bfa"),
        ("Behavioral Signals", "15%", "#f472b6"),
        ("Availability Score", "10%", "#fbbf24")
    ]
    
    for name, val, color in weights:
        st.markdown(f"""
        <div class="weight-item" style="border-left-color: {color};">
            <span class="weight-name">{name}</span>
            <span class="weight-value" style="color: {color};">{val}</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("**Team**: BinaryMinds")
    st.markdown("**Members**:")
    st.markdown("- Siddhi Mehrotra")
    st.markdown("- Riddhi Mehrotra")

# --- MAIN CONTENT ---
st.markdown("<h1 class='gradient-text'>BinaryMinds — Intelligent Candidate Ranker</h1>", unsafe_allow_html=True)

# Description card
st.markdown("""
<div class="glass-card">
    <h4>About the System</h4>
    <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">
        This dashboard processes candidate profiles through an advanced engineering pipeline. 
        It filters out fraudulent profile configurations (honeypots), extracts standard features, 
        evaluates compliance against the Job Description requirements, and ranks them by composite score.
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader
st.subheader("1. Upload Candidate Profiles")
uploaded_file = st.file_uploader(
    "Choose a JSON file containing candidate profiles (e.g. sample_candidates.json)",
    type=["json"],
    help="Upload candidates list to process"
)

if uploaded_file is not None:
    try:
        # Load JSON candidates
        candidates = json.load(uploaded_file)
        st.success(f"Successfully loaded {len(candidates)} candidates from file.")
        
        st.subheader("2. Run Candidate Ranking")
        if st.button("Rank Candidates", type="primary", use_container_width=True):
            with st.spinner("Processing candidates (filtering honeypots, extracting features, and scoring)..."):
                # Run Step 1: Honeypot Filter
                clean_candidates, removed_count = filter_honeypots(candidates)
                
                # Run Step 2: Candidate Scorer
                # Since Streamlit upload is offline and doesn't load a full JD embeddings model,
                # we run feature scoring with default semantic score of 0.5.
                scored_results = score_candidates(clean_candidates, JD_SKILLS)
                
                # Display metrics
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-label">Uploaded</div>
                        <div class="metric-number" style="color: #60a5fa;">{len(candidates)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Honeypots Removed</div>
                        <div class="metric-number" style="color: #f87171;">{removed_count}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Clean Candidates</div>
                        <div class="metric-number" style="color: #34d399;">{len(clean_candidates)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Build display dataframe
                display_rows = []
                for rank, res in enumerate(scored_results, 1):
                    display_rows.append({
                        "Rank": rank,
                        "Candidate ID": res["candidate_id"],
                        "Score": round(res["final_score"], 4),
                        "Reasoning": res["reasoning"],
                        "Skills Score": round(res["skills_match_score"], 2),
                        "Exp Score": round(res["experience_score"], 2),
                        "Behavioral": round(res["behavioral_score"], 2),
                        "Availability": round(res["availability_score"], 2)
                    })
                
                df_all = pd.DataFrame(display_rows)
                
                # Show top 10
                st.subheader("Top 10 Ranked Candidates")
                st.dataframe(
                    df_all.head(10)[["Rank", "Candidate ID", "Score", "Reasoning"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Detailed breakdown section
                st.subheader("Detailed Feature Breakdowns (All Candidates)")
                st.dataframe(df_all, use_container_width=True, hide_index=True)
                
                # Download CSV
                st.subheader("3. Export Ranked CSV")
                # Format to final submission schema
                df_csv = df_all.copy()
                df_csv["rank"] = df_csv["Rank"]
                df_csv["candidate_id"] = df_csv["Candidate ID"]
                df_csv["score"] = df_csv["Score"]
                df_csv["reasoning"] = df_csv["Reasoning"]
                df_csv = df_csv[["candidate_id", "rank", "score", "reasoning"]]
                
                csv_data = df_csv.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Full CSV Results",
                    data=csv_data,
                    file_name="ranked_candidates_submission.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
else:
    st.info("Please upload a candidates JSON file to start the ranking process.")
