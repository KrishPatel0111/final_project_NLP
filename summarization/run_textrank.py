# summarization/textrank_runner.py
import math
from tqdm import tqdm
import spacy, pytextrank as ptr
from sum_config import STYLES
from utils import iter_articles, write_record, ensure_dir, get_root_dir
import os

_nlp = spacy.load("en_core_web_sm")
_nlp.add_pipe("textrank")

def pick(text, k):
    doc = _nlp(text)
    return [s.text.strip() for s in doc._.textrank.summary(limit_phrases=15, limit_sentences=k)]

def summarize(text, style):
    sents_total = max(1, sum(1 for _ in _nlp(text).sents))
    if style in ("bullets", "one_paragraph"):
        k = max(5, min(7, math.ceil(0.22 * sents_total)))
        s = pick(text, k)
        return ("- " + "\n- ".join(s[:7])) if style == "bullets" else " ".join(s[:7])
    if style == "three_sentence":
        return " ".join(pick(text, 3)[:3])
    return ""

def run(in_csv="data/guardian_articles_raw.csv", out_jsonl="data/outputs/textrank.jsonl", limit=0):
    root_dir = get_root_dir()
    out_jsonl = os.path.join(root_dir, out_jsonl)
    in_csv = os.path.join(root_dir, in_csv)
    ensure_dir(out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, title, body, domain, url in tqdm(iter_articles(in_csv, limit=limit)):
            for style in STYLES:
                s = summarize(body, style)
                write_record(out, {
                    "doc_id": i, "domain": domain, "url": url,
                    "model": "TextRank", "style": style, "title": title, "summary": s
                })

if __name__ == "__main__":
    run()
