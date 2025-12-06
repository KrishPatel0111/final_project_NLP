# extraction/deepseek_cue_extraction.py

import json
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
import re
from datetime import datetime
import requests

load_dotenv()

CONFIG = {
    "templates_dir": "data/prompt_previews",
    "n_examples": 2,
    "output_file": "data/extractions/deepseek_article_only_cues.jsonl",
    "api_url": "https://api.deepseek.com/chat/completions",
    "model_name": "deepseek-chat",
    "temperature": 0.1,
}


TEMPLATES = {}

# --------------------------------------------------------------------
# TEMPLATE LOADING
# --------------------------------------------------------------------
def load_templates():
    global TEMPLATES
    template_dir = CONFIG["templates_dir"]
    domains = ["educational", "politics", "cultural", "sport", "technology", "social"]

    print(f"\n📋 Loading templates from: {template_dir}")

    for domain in domains:
        fname = f"{domain}_ex_no_{CONFIG['n_examples']}.txt"
        path = os.path.join(template_dir, fname)

        if os.path.exists(path):
            TEMPLATES[domain] = open(path, "r", encoding="utf-8").read()
            print(f"   ✅ Loaded template for {domain}")
        else:
            print(f"   ⚠️ Missing: {fname}")

    if len(TEMPLATES) == 0:
        print("❌ No templates loaded")
        return False

    return True


# --------------------------------------------------------------------
# PROMPT BUILDER — ARTICLE ONLY (NO PRESERVATION, NO SUMMARY)
# --------------------------------------------------------------------
def build_prompt(domain, title, article_text):
    template = TEMPLATES.get(domain.lower(), "")

    prompt = (
        template
        + f"\n\nARTICLE TITLE: {title}\n"
        f"DOMAIN: {domain}\n\n"
        f"ARTICLE TEXT:\n{article_text}\n\n"
        "🎯 TASK:\n"
        "- Extract ALL contextual cues from the ARTICLE ONLY.\n"
        "- Extract ALL cultural cues from the ARTICLE ONLY.\n"
        "- DO NOT reference or compare to the summary.\n"
        "- DO NOT mention preserved/lost cues.\n"
        "- DO NOT classify or explain.\n"
        "- Output ONLY cues.\n\n"
        "🎯 STRICT JSON OUTPUT:\n"
        "{\n"
        '  \"contextual_cues\": [ ... ],\n'
        '  \"cultural_cues\": [ ... ]\n'
        "}\n"
    )

    return prompt


# --------------------------------------------------------------------
# CLEAN JSON
# --------------------------------------------------------------------
def fix_json_errors(text):
    text = text.replace("```json", "").replace("```", "")
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text.strip()


# --------------------------------------------------------------------
# CALL DEEPSEEK
# --------------------------------------------------------------------
def deepseek_call(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Missing DEEPSEEK_API_KEY in .env file")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": CONFIG["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": CONFIG["temperature"],
    }

    r = requests.post(CONFIG["api_url"], headers=headers, json=payload)

    if r.status_code != 200:
        print("❌ DeepSeek Error:", r.text)
        return None

    try:
        return r.json()["choices"][0]["message"]["content"]
    except:
        print("❌ Unexpected DeepSeek Output:", r.text)
        return None


# --------------------------------------------------------------------
# EXTRACTION LOGIC
# --------------------------------------------------------------------
def extract_from_article(domain, title, text):
    prompt = build_prompt(domain, title, text)
    raw = deepseek_call(prompt)

    if raw is None:
        return None

    try:
        return json.loads(raw)
    except:
        fixed = fix_json_errors(raw)
        try:
            return json.loads(fixed)
        except:
            print("\n❌ Still invalid JSON:")
            print(raw[:200])
            return None


# --------------------------------------------------------------------
# LOAD DATA (FIX DOMAIN_X / DOMAIN_Y)
# --------------------------------------------------------------------
def load_data():
    print("\n📊 Loading data...")

    summaries = pd.read_json("data/outputs/all_summaries.jsonl", lines=True)
    articles = pd.read_csv("data/guardian_articles_raw.csv")

    print("\n🔍 Columns in summaries:", summaries.columns.tolist())
    print("🔍 Columns in articles:", articles.columns.tolist())

    merged = summaries.merge(
        articles[["title", "text", "domain"]],
        on="title",
        how="inner"
    )

    print("\n🔍 Columns AFTER merge:", merged.columns.tolist())

    # FIX DOMAIN COLUMNS
    if "domain_y" in merged.columns:
        merged = merged.rename(columns={"domain_y": "domain"})
    elif "domain" not in merged.columns:
        raise KeyError("❌ No domain column available after merge!")

    if "domain_x" in merged.columns:
        merged = merged.drop(columns=["domain_x"])

    print("\n🔍 Columns AFTER FIX:", merged.columns.tolist())

    merged = merged.drop_duplicates(subset="title")

    merged = merged.head(5)
    print(f"\n📌 Loaded {len(merged)} articles for extraction")

    return merged


# --------------------------------------------------------------------
# SAVE EXTRACTION
# --------------------------------------------------------------------
def save_extraction(entry):
    os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
    with open(CONFIG["output_file"], "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------
# MAIN EXTRACTION LOOP
# --------------------------------------------------------------------
def run_extraction():
    print("=" * 80)
    print("🚀 RUNNING ARTICLE-ONLY CUE EXTRACTION USING DEEPSEEK")
    print("=" * 80)

    if not load_templates():
        return

    df = load_data()

    print("\nExtracting cues...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        result = extract_from_article(row["domain"], row["title"], row["text"])

        if result:
            entry = {
                "title": row["title"],
                "domain": row["domain"],
                "extraction_timestamp": datetime.now().isoformat(),
                "cues": result,
            }
            save_extraction(entry)
        else:
            print(f"\n❌ Extraction failed for: {row['title']}")
            break

    print("\n🎉 Extraction complete! Saved to:")
    print(CONFIG["output_file"])


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    run_extraction()


if __name__ == "__main__":
    main()
