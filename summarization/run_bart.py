# summarization/bart_runner.py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sum_config import STYLES, target_word_bounds, approx_tokens_from_words
from utils import iter_articles, write_record, ensure_dir, style_from_one_paragraph, word_count, clamp_chars

MODEL = "facebook/bart-large-cnn"
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto")
summ = pipeline("summarization", model=mdl, tokenizer=tok, device_map="auto")

def summarize_one(body, style, src_words):
    if style == "one_paragraph":
        lo, hi = target_word_bounds(src_words)
        max_new = approx_tokens_from_words(hi)
    elif style == "bullets":
        max_new = 220
    else:
        max_new = 120
    out = summ(clamp_chars(body), do_sample=False, truncation=True, max_new_tokens=max_new)[0]["summary_text"]
    return style_from_one_paragraph(out, style)

def run(in_csv="data/guardian_articles_raw.csv", out_jsonl="outputs/bart.jsonl", limit=0):
    ensure_dir(out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, title, body, domain, url in tqdm(iter_articles(in_csv, limit=limit)):
            src_words = word_count(body)
            for style in STYLES:
                s = summarize_one(body, style, src_words)
                write_record(out, {
                    "doc_id": i, "domain": domain, "url": url,
                    "model": "BART-large-cnn", "style": style, "title": title, "summary": s
                })

if __name__ == "__main__":
    run()
