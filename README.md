# BinaryMinds — Intelligent Candidate Ranker
**Team:** Siddhi Mehrotra, Riddhi Mehrotra  
**Hackathon:** Redrob x Hack2Skill — Intelligent Candidate Discovery & Ranking Challenge  
**Submission Deadline:** 2 July 2026

## Project Overview
An intelligent, end-to-end recruitment system designed to identify and rank candidate matches for senior applied AI roles. It streams and scores candidates under tight performance limits, filtering fraudulent profiles, extracting behavioral and professional features, and computing semantic similarities using compact offline language models.

## Project Structure
```text
├── README.md                           # Project instructions and documentation
├── requirements.txt                    # Project dependencies
├── app/
│   └── streamlit_app.py                # Visual Streamlit dashboard for judges
├── src/
│   ├── honeypot_filter.py              # Removes impossible/suspicious candidate profiles
│   ├── feature_extractor.py            # Extracts normalized numeric score features
│   ├── scorer.py                       # Computes final weighted scores and reasonings
│   ├── semantic_similarity.py          # Computes offline sentence transformer similarities
│   └── ranker.py                       # Main end-to-end execution pipeline
├── output/
│   └── submission.csv                  # Final ranked candidates submission file
└── India_runs_data_and_ai_challenge/   # Challenge datasets and specifications
```

## How to Install
```bash
pip install -r requirements.txt
```

## How to Download the AI Model (do once)
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## How to Run the Ranker
```bash
python src/ranker.py
```

## How to Run the Demo
```bash
streamlit run app/streamlit_app.py
```

## How It Works
1. **Honeypot Filter**: Removes candidates with impossible years of experience, perfect/suspicious mock assessment profiles, invalid expected salary ranges, job start dates prior to graduation, or empty/incomplete profiles flagged as open to work.
2. **Feature Extraction**: Extracts normalized scores representing required skills match, total experience capped at 15 years, profile seniority level, behavioral activity (GitHub commits, interview and response rates), and availability.
3. **Initial Scoring**: Computes weighted scores using default parameters on streaming candidate blocks of 5,000 to dynamically shortlist the top 2,000 candidates.
4. **Semantic Similarity**: Processes the shortlist of 2,000 candidates, generating embeddings from combined summaries and career descriptions to compute cosine similarities against the Job Description.
5. **Final Ranking**: Combines feature weights with actual semantic similarity scores to produce a final composite ranking.
6. **Output & Validation**: Generates a valid `submission.csv` containing the top 100 candidates ranked, scored, and explained, and runs local auto-validation checks.

## Constraints Met
- **Runs in under 5 minutes on CPU only**: Completed on 100,000 candidates in 152.74 seconds.
- **No internet required during ranking**: Sentence-transformers model weights run locally.
- **Peak RAM under 16 GB**: Batch streaming limits peak memory footprint.
- **No GPU required**: Entire pipeline computes on standard CPU architecture.
