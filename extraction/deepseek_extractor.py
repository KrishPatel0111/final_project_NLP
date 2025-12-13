# extraction/deepseek_extractor_production.py

import json
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
import re
from datetime import datetime
import requests
from pathlib import Path

load_dotenv()

CONFIG = {
    'templates_dir': 'data/prompt_previews',
    'n_examples': 2,
    'output_file': 'data/extractions/deepseek_extractions_production.jsonl',
    'metadata_file': 'data/extractions/extraction_metadata.json',
    'model_name': 'deepseek-chat',
    'temperature': 0.1,
    'save_every': 1,
    'api_url': "https://api.deepseek.com/chat/completions"
}

TEMPLATES = {}

# ---------------- TEMPLATE LOADING ---------------- #

def load_templates():
    global TEMPLATES
    template_dir = CONFIG['templates_dir']
    domains = ["educational", "politics", "cultural", "sport", "technology", "social"]

    print(f"\n📋 Loading prompt templates from: {template_dir}/")

    for domain in domains:
        filename = f"{domain}_ex_no_{CONFIG['n_examples']}.txt"
        path = os.path.join(template_dir, filename)

        if os.path.exists(path):
            TEMPLATES[domain] = open(path, 'r', encoding='utf-8').read()
            print(f"   ✅ Loaded {domain}")
        else:
            print(f"   ⚠️ Missing: {filename}")

    return len(TEMPLATES) > 0

# ---------------- PROMPT BUILDER ---------------- #

def format_target(article_title, article_text, summary_text, domain):
    return f"""ARTICLE TITLE: {article_title}
DOMAIN: {domain}

ORIGINAL ARTICLE (identify cues from here):
{article_text}

SUMMARY (check if cues are preserved here):
{summary_text}

OUTPUT YOUR JSON ANALYSIS:
"""

# ---------------- JSON CLEANING ---------------- #

def fix_json_errors(json_text):
    json_text = json_text.replace('"Contexttype"', '"Contextual"')
    json_text = json_text.replace('"Cue type"', '"Cue Type"')
    json_text = json_text.replace("```json", "").replace("```", "")
    json_text = re.sub(r",(\s*[}\]])", r"\1", json_text)
    return json_text.strip()

# ---------------- DEEPSEEK CALL ---------------- #

def deepseek_call(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Missing DEEPSEEK_API_KEY in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": CONFIG["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": CONFIG["temperature"]
    }

    resp = requests.post(CONFIG["api_url"], headers=headers, json=payload)

    if resp.status_code != 200:
        print("❌ DeepSeek API error:", resp.text)
        return None

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except:
        print("❌ Unexpected DeepSeek response:", resp.text)
        return None

# ---------------- EXTRACTION ---------------- #

def extract_with_deepseek(article_title, article_text, summary_text, domain):
    domain = domain.lower().strip()
    template = TEMPLATES.get(domain, TEMPLATES.get("politics", ""))

    prompt = template + format_target(article_title, article_text, summary_text, domain)

    raw_output = deepseek_call(prompt)
    if raw_output is None:
        return None

    try:
        return json.loads(raw_output)
    except:
        fixed = fix_json_errors(raw_output)
        try:
            return json.loads(fixed)
        except:
            print("\n❌ JSON still invalid after fix:")
            print(raw_output[:200])
            return None

# ---------------- DATA LOADING ---------------- #

def load_data():
    summaries_file = "data/outputs/all_summaries.jsonl"
    articles_file = "data/guardian_articles_raw.csv"

    print("\n📊 Loading data...")

    if not os.path.exists(summaries_file):
        print("❌ Missing summaries file")
        return None
    if not os.path.exists(articles_file):
        print("❌ Missing articles file")
        return None

    summaries = pd.read_json(summaries_file, lines=True)
    articles = pd.read_csv(articles_file)

    merged = summaries.merge(articles[['title', 'text', 'domain']], on='title', how='inner')

    # ------------------------------------------------------------------
    # FIX FOR KeyError: 'domain'
    # Pandas renames duplicates as domain_x (summaries) and domain_y (articles)
    # We WANT the article CSV domain.
    # ------------------------------------------------------------------
    if "domain_x" in merged.columns and "domain_y" in merged.columns:
        merged = merged.drop(columns=["domain_x"])
        merged = merged.rename(columns={"domain_y": "domain"})
    elif "domain" not in merged.columns:
        for c in merged.columns:
            if "domain" in c.lower():
                merged = merged.rename(columns={c: "domain"})
                break
    # ------------------------------------------------------------------

    print(f"   ✅ Merged: {len(merged)} rows")
    return merged

# ---------------- PROGRESS TRACKING ---------------- #

def load_existing_extractions():
    output_file = CONFIG["output_file"]
    if not os.path.exists(output_file):
        return set()

    done = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add((rec["article_title"], rec["summary_model"], rec["summary_style"]))
    return done

def save_extraction(entry):
    output = CONFIG["output_file"]
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def save_metadata(stats):
    metadata_file = CONFIG["metadata_file"]
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

# ---------------- EXTRACTION LOOP ---------------- #

def run_extraction():

    print("="*80)
    print("🚀 RUNNING DEEPSEEK EXTRACTION (LIMIT = 2 ARTICLES)")
    print("="*80)

    if not load_templates():
        print("❌ Templates missing")
        return

    df = load_data()
    if df is None:
        return

    # ---- LIMIT TO 2 ARTICLES ---- #
    df = df.head(2)
    print(f"🚧 TEST MODE: Only processing {len(df)} articles")
    # ------------------------------ #

    done = load_existing_extractions()

    df["key"] = df.apply(lambda r: (r["title"], r["model"], r["style"]), axis=1)
    df = df[~df["key"].isin(done)]

    print(f"\n📌 Remaining after resume filter: {len(df)}")

    stats = {"successful": 0, "failed": 0, "start": datetime.now().isoformat()}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):

        result = extract_with_deepseek(row["title"], row["text"], row["summary"], row["domain"])

        if result:
            entry = {
                "article_title": row["title"],
                "summary_model": row["model"],
                "summary_style": row["style"],
                "domain": row["domain"],
                "extraction_model": CONFIG["model_name"],
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_result": result
            }
            save_extraction(entry)
            stats["successful"] += 1
        else:
            print(f"\n❌ FAILED on: {row['title']}")
            stats["failed"] += 1
            break

    stats["end"] = datetime.now().isoformat()
    save_metadata(stats)

    print("\n🎉 Extraction finished")
    print(stats)

# ---------------- MAIN ---------------- #

def main():
    load_dotenv()
    run_extraction()

if __name__ == "__main__":
    main()
