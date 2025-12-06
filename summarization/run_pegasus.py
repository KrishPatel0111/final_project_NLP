from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sum_config import target_word_bounds, approx_tokens_from_words
from utils import iter_articles, write_record, ensure_dir, word_count, clamp_chars, get_root_dir
import os

MODEL = "google/pegasus-cnn_dailymail"
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto")
summ = pipeline("summarization", model=mdl, tokenizer=tok, device_map="auto")

tok.model_max_length = min(getattr(mdl.config, "max_position_embeddings", 1024) or 1024, 1024)

# Quality-first decoding defaults
BEAMS = 8
DIVERSITY_PENALTY = 0.1
NO_REPEAT_NGRAM = 4
REPETITION_PENALTY = 1.05
LENGTH_PENALTY = 2.0

def summarize_one(body, src_words):
    """
    PEGASUS is pre-trained for news summarization (not instruction-tuned).
    Generate ONE paragraph summary per article.
    """
    text = clamp_chars(body)
    lo, hi = target_word_bounds(src_words)
    min_new = max(approx_tokens_from_words(lo), 32)
    max_new = approx_tokens_from_words(hi)

    out = summ(
        text,
        do_sample=False,
        truncation=True,
        min_new_tokens=min_new,
        max_new_tokens=max_new,
        num_beams=BEAMS,
        diversity_penalty=DIVERSITY_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM,
        repetition_penalty=REPETITION_PENALTY,
        length_penalty=LENGTH_PENALTY,
        early_stopping=True,
    )[0]["summary_text"]

    out = out.replace("<n>", " ")  # ← ADD THIS LINE
    out = out.strip()
    return out

def run(in_csv="data/guardian_articles_raw.csv", out_jsonl="data/outputs/pegasus.jsonl", limit=0):
    root = get_root_dir()
    in_csv = os.path.join(root, in_csv)
    out_jsonl = os.path.join(root, out_jsonl)
    print("[pegasus] IN:", in_csv)
    print("[pegasus] OUT:", out_jsonl)

    ensure_dir(out_jsonl)

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, title, body, domain, url in tqdm(iter_articles(in_csv, limit=limit)):
            src_words = word_count(body)
            s = summarize_one(body, src_words)
            
            write_record(out, {
                "doc_id": i,
                "domain": domain,
                "url": url,
                "model": "PEGASUS-cnn_dailymail",
                "style": "one_paragraph",  # Only one style
                "title": title,
                "summary": s
            })
            out.flush()

if __name__ == "__main__":
    run()
    