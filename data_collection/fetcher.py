import os
import time
import re
import requests
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

# ---------------------
# Config
# ---------------------
load_dotenv("/home/krish/uni/cse5525/final/final_project_NLP/keys.env")
API_KEY = os.getenv("GUARDIAN_API_KEY")
assert API_KEY, "Missing GUARDIAN_API_KEY in env"

TARGET_PER_DOMAIN = 25
LOOKBACK_DAYS = 540   # ~18 months
FROM_DATE = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()

domains = {
    "politics": "politics",
    "culture": "culture|music|artanddesign|stage",
    "education": "education",
    "sports": "sport",
    "tech": "technology",
    "social": "world|global-development|commentisfree"
}

# Small, high-precision cultural cue list (expand later)
CUE_TERMS = {
    "queer","lgbtq","latinx","indigenous","diaspora","immigrant","hijab","halal",
    "heritage","tradition","festival","ballroom","drag","pride","community",
    "afro","caribbean","bilingual","rural","working-class","faith","synagogue",
    "mosque","temple","caste","tribal"
}
cue_regex = re.compile(r"\b(" + "|".join(re.escape(t) for t in CUE_TERMS) + r")\b", re.I)

def cultural_score(text: str) -> int:
    if not text:
        return 0
    return len(cue_regex.findall(text))

def fetch_domain(domain_name: str, section_filter: str) -> list:
    """Fetch newest Guardian 'article' items for a domain, skip hosted/sponsored,
    and prefer culturally dense content."""
    collected = []
    seen_ids = set()
    page = 1
    max_pages = 20   # hard cap to avoid infinite loops

    # Optional: bias query toward cultural terms to pull richer articles earlier
    q_bias = " OR ".join(list(CUE_TERMS)[:10])  # keep URL short; adjust as needed

    while len(collected) < TARGET_PER_DOMAIN and page <= max_pages:
        url = (
            "https://content.guardianapis.com/search"
            f"?section={section_filter}"
            f"&page-size=50&page={page}"
            f"&order-by=newest"
            f"&from-date={FROM_DATE}"
            f"&show-fields=bodyText,headline,byline"
            f"&show-tags=keyword"
            f"&api-key={API_KEY}"
            f"&q={requests.utils.quote(q_bias)}" # type: ignore
        )
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            print(f"[{domain_name}] HTTP {r.status_code}: {r.text[:200]}")
            break

        data = r.json()
        if "response" not in data or "results" not in data["response"]:
            print(f"[{domain_name}] Unexpected payload: {str(data)[:200]}")
            break

        resp = data["response"]
        results = resp.get("results", [])
        if not results:
            break

        for item in results:
            # Guardrails: only normal articles, skip hosted/sponsored and liveblogs
            if item.get("type") != "article":
                continue
            if item.get("isHosted", False):
                continue  # partner/sponsored content
            fields = item.get("fields") or {}
            body = fields.get("bodyText") or ""
            title = fields.get("headline") or ""
            url_web = item.get("webUrl") or ""
            if not body or not title or not url_web:
                continue
            if item.get("id") in seen_ids:
                continue

            # Quick cultural density heuristic
            score = cultural_score(title + "\n" + body)
            # You can use a threshold (e.g., >=2) OR collect all with scores and rank later
            collected.append({
                "id": item.get("id"),
                "title": title,
                "text": body,
                "domain": domain_name,
                "url": url_web,
                "sectionName": item.get("sectionName"),
                "publicationDate": item.get("webPublicationDate"),
                "score_cultural": score
            })
            seen_ids.add(item.get("id"))

        # Continue paging if needed
        if len(collected) < TARGET_PER_DOMAIN:
            page += 1
            time.sleep(0.25)
        else:
            break

    # Post-filter: keep the top culturally dense pieces, newest first
    collected.sort(key=lambda x: (x["score_cultural"], x["publicationDate"] or ""), reverse=True)
    return collected[:TARGET_PER_DOMAIN]

# ---------------------
# Run all domains
# ---------------------
all_articles = []
for domain, section in domains.items():
    print(f"Fetching {domain}…")
    items = fetch_domain(domain, section)
    # If too few, relax by removing q_bias in a second pass (fallback)
    if len(items) < TARGET_PER_DOMAIN:
        print(f"{domain}: got {len(items)}; running fallback without cultural bias query")
        # quick fallback: re-run without q bias to fill the quota
        # (reuse same function but with empty CUE_TERMS temporarily)
        
        bak_terms, bak_regex = CUE_TERMS, cue_regex
        CUE_TERMS = set()  # disable bias
        cue_regex = re.compile("$^")  # match nothing
        items2 = fetch_domain(domain, section)
        CUE_TERMS, cue_regex = bak_terms, bak_regex
        # Merge & re-rank by cultural score (original regex works again)
        merged = items + items2
        for it in merged:
            if it["score_cultural"] == 0:
                it["score_cultural"] = cultural_score(it["title"] + "\n" + it["text"])
        merged.sort(key=lambda x: (x["score_cultural"], x["publicationDate"] or ""), reverse=True)
        items = []
        seen = set()
        for it in merged:
            if it["id"] not in seen:
                items.append(it); seen.add(it["id"])
            if len(items) >= TARGET_PER_DOMAIN:
                break
    all_articles.extend(items)

df = pd.DataFrame(all_articles)
# Basic sanity filters
df = df[df["text"].str.split().str.len() >= 250]  # drop very short pieces
df.drop_duplicates(subset=["url"], inplace=True)

# Save
os.makedirs("data", exist_ok=True)
df.to_csv("data/guardian_articles_raw.csv", index=False)
print(df.groupby("domain").size())
print("Saved to data/guardian_articles_raw.csv")
