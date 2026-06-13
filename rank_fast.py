"""
Redrob Fast Ranker — TF-IDF + Signal Scoring
Runs in under 2 minutes on CPU!
Author: Dimpal Shegekar
Usage: python rank_fast.py --candidates candidates.jsonl --out submission.csv
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Starting Redrob Fast Ranker...")

JD_TEXT = """
Senior AI Engineer at Redrob AI. Production experience required with 
embeddings retrieval systems sentence-transformers vector databases 
pinecone weaviate qdrant milvus faiss elasticsearch opensearch 
hybrid search ranking systems python pytorch tensorflow hugging face 
transformers bert nlp rag retrieval augmented generation fine-tuning 
lora qlora peft llm recommendation system information retrieval 
learning to rank ndcg mrr bm25 reranking vector search semantic search 
dense retrieval sparse retrieval mlflow weights biases bentoml 
5 to 9 years experience product company shipped end to end ranking 
search recommendation systems real users evaluation frameworks
"""

CORE_AI_SKILLS = {
    "milvus","faiss","pinecone","weaviate","qdrant","opensearch",
    "elasticsearch","vector database","vector search","hybrid search",
    "semantic search","dense retrieval","embeddings","sentence-transformers",
    "bge","e5","ranking","recommendation","information retrieval",
    "learning to rank","reranking","ndcg","mrr","bm25","llm",
    "fine-tuning","lora","qlora","peft","transformers","bert","gpt",
    "nlp","rag","mlflow","weights & biases","bentoml","python",
    "pytorch","tensorflow","hugging face","huggingface","rag",
    "retrieval augmented generation","tts","speech recognition",
    "image classification","nlp","gans","statistical modeling",
}

NEGATIVE_TITLES = {
    "marketing manager","hr manager","content writer","operations manager",
    "graphic designer","business analyst","sales manager","accountant",
    "mechanical engineer","customer support","seo","brand manager",
}

SERVICES_COMPANIES = {
    "tcs","infosys","wipro","accenture","cognizant","capgemini",
    "tech mahindra","hcl","mindtree","mphasis","hexaware",
}

def is_honeypot(c):
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    yoe = profile.get("years_of_experience", 0)
    total_months = sum(j.get("duration_months", 0) for j in career)
    total_years = total_months / 12
    if yoe > 3 and total_years > 0:
        if yoe > total_years * 2.5:
            return True
    skills = c.get("skills", [])
    impossible = sum(1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
    if impossible >= 3:
        return True
    return False

def score_experience(c):
    yoe = c.get("profile", {}).get("years_of_experience", 0)
    if 5 <= yoe <= 9: return 1.0
    elif 4 <= yoe < 5: return 0.8
    elif 9 < yoe <= 12: return 0.75
    elif 3 <= yoe < 4: return 0.5
    elif yoe > 12: return 0.6
    else: return 0.1

def score_ai_skills(c):
    skills = c.get("skills", [])
    total = 0.0
    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)
        end = s.get("endorsements", 0)
        relevant = any(k in name for k in CORE_AI_SKILLS) or name in CORE_AI_SKILLS
        if relevant:
            pw = {"advanced":1.0,"intermediate":0.7,"beginner":0.3}.get(prof, 0.3)
            dw = min(dur/60, 1.0)
            eb = min(end/50, 0.3)
            total += pw * (0.7 + 0.3*dw) + eb
    return min(total/8.0, 1.0)

def score_role_fit(c):
    title = c.get("profile", {}).get("current_title", "").lower()
    career = c.get("career_history", [])
    for neg in NEGATIVE_TITLES:
        if neg in title:
            return 0.1
    companies = [j.get("company","").lower() for j in career]
    services_only = all(any(sc in co for sc in SERVICES_COMPANIES)
        for co in companies if co) if companies else False
    if services_only and len(companies) >= 2:
        return 0.4
    return 1.0

def score_behavioral(c):
    s = c.get("redrob_signals", {})
    score = 0.0

    last = s.get("last_active_date","")
    if last:
        try:
            dt = datetime.strptime(last, "%Y-%m-%d")
            days = (datetime(2026,6,13) - dt).days
            r = 1.0 if days<=30 else 0.8 if days<=90 else 0.5 if days<=180 else 0.3 if days<=365 else 0.0
            score += 0.25*r
        except: pass

    if s.get("open_to_work_flag", False): score += 0.15
    rr = s.get("recruiter_response_rate", 0)
    score += 0.15 * min(rr/0.8, 1.0)
    pc = s.get("profile_completeness_score", 0)/100
    score += 0.10*pc
    notice = s.get("notice_period_days", 90)
    ns = 1.0 if notice<=30 else 0.8 if notice<=60 else 0.6 if notice<=90 else 0.3 if notice<=120 else 0.1
    score += 0.10*ns
    gh = s.get("github_activity_score", -1)
    if gh >= 0: score += 0.10*(gh/100)
    icr = s.get("interview_completion_rate", 0)
    score += 0.08*icr
    if s.get("willing_to_relocate", False): score += 0.07
    return min(score, 1.0)

def score_location(c):
    country = c.get("profile",{}).get("country","").lower()
    location = c.get("profile",{}).get("location","").lower()
    if country != "india": return 0.4
    preferred = ["pune","noida","delhi","bangalore","bengaluru",
                 "mumbai","hyderabad","gurugram","gurgaon","chennai"]
    if any(city in location for city in preferred): return 1.0
    return 0.7

def build_text(c):
    p = c.get("profile",{})
    career = c.get("career_history",[])
    skills = c.get("skills",[])
    parts = [
        p.get("headline",""),
        p.get("summary",""),
        p.get("current_title",""),
        " ".join(j.get("description","") for j in career[:3]),
        " ".join(s.get("name","") for s in skills),
    ]
    return " ".join(x for x in parts if x)[:1500]

def generate_reasoning(c, rank):
    p = c.get("profile",{})
    s = c.get("redrob_signals",{})
    career = c.get("career_history",[])
    title = p.get("current_title","Unknown")
    yoe = p.get("years_of_experience",0)
    company = p.get("current_company","Unknown")
    notice = s.get("notice_period_days","?")
    open_work = s.get("open_to_work_flag", False)
    skills = c.get("skills",[])
    rel = [s["name"] for s in skills
           if any(k in s["name"].lower() for k in CORE_AI_SKILLS)
           and s.get("proficiency") in ["advanced","intermediate"]][:3]
    skill_str = ", ".join(rel) if rel else "limited AI skills"
    parts = [f"{title} with {yoe:.1f} yrs at {company}"]
    if rel: parts.append(f"skilled in {skill_str}")
    if open_work: parts.append("open to work")
    if isinstance(notice, (int,float)):
        if notice <= 60: parts.append(f"notice {notice}d")
        else: parts.append(f"long notice {notice}d")
    return "; ".join(parts) + "."

def main(candidates_path, output_path):
    # Load
    print(f"Loading {candidates_path}...")
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates.")

    # Honeypot filter
    print("Detecting honeypots...")
    clean = [c for c in candidates if not is_honeypot(c)]
    removed = len(candidates) - len(clean)
    print(f"Removed {removed} honeypots. {len(clean):,} remaining.")
    candidates = clean

    # Build texts
    print("Building texts...")
    texts = [build_text(c) for c in candidates]

    # TF-IDF (fast!)
    print("Computing TF-IDF similarity (fast mode)...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        sublinear_tf=True,
        min_df=2
    )
    all_texts = [JD_TEXT] + texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vec = tfidf_matrix[0]
    cand_vecs = tfidf_matrix[1:]
    semantic_scores = cosine_similarity(jd_vec, cand_vecs)[0]
    print("TF-IDF done!")

    # Score all
    print("Computing component scores...")
    results = []
    for i, c in enumerate(candidates):
        sem = float(semantic_scores[i])
        exp = score_experience(c)
        ai = score_ai_skills(c)
        role = score_role_fit(c)
        beh = score_behavioral(c)
        loc = score_location(c)

        final = (0.35*sem + 0.20*ai + 0.15*exp + 0.15*beh + 0.10*loc) * role
        results.append({"candidate": c, "score": final})

    # Sort
    results.sort(key=lambda x: x["score"], reverse=True)
    top100 = results[:100]

    # Normalize scores
    max_s = top100[0]["score"]
    min_s = top100[-1]["score"]
    rng = max_s - min_s if max_s != min_s else 1.0

    # Write CSV
    print(f"Writing {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id","rank","score","reasoning"])
        for rank, item in enumerate(top100, 1):
            c = item["candidate"]
            raw = item["score"]
            norm = round(0.40 + 0.59*(raw-min_s)/rng, 4)
            reasoning = generate_reasoning(c, rank)
            writer.writerow([c["candidate_id"], rank, norm, reasoning])

    print(f"\n✅ Done! submission written to {output_path}")
    print(f"Top candidate: {top100[0]['candidate']['candidate_id']}")
    print(f"\nNow run: python validate_submission.py {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    main(args.candidates, args.out)
