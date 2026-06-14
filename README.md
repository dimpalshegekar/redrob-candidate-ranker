# 🧠 Redrob Intelligent Candidate Ranking System
**India Runs Hackathon × Redrob AI — Track 1: Data & AI Challenge**
**Author:** Dimpal Shegekar | [@dimpalshegekar](https://github.com/dimpalshegekar)

---

## 🎯 Problem Statement
Recruiters go through hundreds of profiles and still miss the right person. 
Not because the talent isn't there — but because keyword filters can't see 
what actually matters.

## 💡 Solution
An AI-powered ranking engine that intelligently ranks candidates by going 
beyond keywords — using semantic understanding, skill relevance, experience 
fit, and behavioral signals from the Redrob platform.

---

## 🏗️ Architecture
candidates.jsonl (100K candidates)

↓

Honeypot Detection → Remove 42 impossible profiles

↓

Text Representation → headline + summary + career + skills

↓

TF-IDF Vectorization (fast, CPU-friendly)

↓

Cosine Similarity vs Job Description

↓
Component Scoring:
Semantic match    (35%)
AI/ML skills      (20%)
Experience fit    (15%)
Behavioral signals(15%)
Location fit      (10%)
Role fit multiplier

↓

Ranked Top 100 CSV with reasoning


---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run ranker
python rank_fast.py --candidates candidates.jsonl --out submission.csv

# Fix tie-breaking
python fix_ties.py

# Validate
python validate_submission.py submission.csv
```

---

## 📊 Scoring Components

| Component | Weight | Description |
|---|---|---|
| Semantic Match | 35% | TF-IDF cosine similarity vs JD |
| AI/ML Skills | 20% | FAISS, embeddings, RAG, LoRA etc. |
| Experience | 15% | Sweet spot: 5-9 years |
| Behavioral Signals | 15% | Activity, response rate, GitHub, notice period |
| Location | 10% | India-based, Pune/Noida preferred |
| Role Fit | Multiplier | Penalizes wrong roles |

---

## ⚠️ Honeypot Detection

Automatically detects and excludes:
- Profiles with impossible YOE vs career history
- Expert skills with 0 duration months
- Suspiciously inflated profiles

**Result: 42 honeypots detected and excluded from 100K candidates**

---

## 📋 Compute Constraints

| Constraint | Limit | This System |
|---|---|---|
| Runtime | ≤ 5 min | ~1-2 min ✅ |
| Memory | ≤ 16 GB | ~1 GB ✅ |
| Compute | CPU only | ✅ CPU only |
| Network | Off | ✅ No API calls |

---

## 📁 Project Structure
redrob-candidate-ranker/

├── rank_fast.py         ← Main ranker (TF-IDF based)

├── fix_ties.py          ← Tie-breaking fix

├── requirements.txt     ← Dependencies

├── README.md            ← This file

└── submission.csv       ← Final ranked output

---

## 🏆 Results
- Processed: **100,000 candidates**
- Honeypots removed: **42**
- Final ranked output: **Top 100 candidates**
- Validation: **Passed ✅**

