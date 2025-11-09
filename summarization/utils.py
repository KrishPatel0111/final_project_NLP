# summarization/utils.py
from anyio import Path
import os, re, json, pandas as pd

def ensure_dir(path: str):
    if path and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s or ""))

def clamp_chars(s: str, limit: int = 8000) -> str:
    s = (s or "").strip()
    return s[:limit]

def style_from_one_paragraph(text: str, style: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    if style == "bullets":
        return "- " + "\n- ".join(parts[:7]) if parts else ""
    if style == "three_sentence":
        return " ".join(parts[:3]) if parts else ""
    return text

def iter_articles(csv_path: str, limit: int = 0):
    df = pd.read_csv(csv_path)
    if limit and limit > 0:
        df = df.head(limit)
    for i, row in df.iterrows():
        yield int(i), str(row["title"]), str(row["text"]), row.get("domain",""), row.get("url","") # type: ignore

def write_record(out_f, rec: dict):
    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
def get_root_dir() -> str:
    from pathlib import Path

    PROJECT_HOME = Path(__file__).resolve()
    while not (PROJECT_HOME / ".git").exists():
        PROJECT_HOME = PROJECT_HOME.parent
        
    return str(PROJECT_HOME)
