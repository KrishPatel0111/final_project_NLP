# summarization/mistral_runner.py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sum_config import STYLES, target_word_bounds, approx_tokens_from_words
from utils import iter_articles, write_record, ensure_dir, word_count, clamp_chars

MODEL = "mistralai/Mistral-7B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto")

def prompt(style, title, body):
    base = (
        "You are a careful, faithful summarizer. Preserve named entities, numbers, dates, and culturally relevant context "
        "(identity, heritage, religion, community, local references). Do NOT add facts.\n\n"
        f"TITLE: {title}\n\nARTICLE:\n{body}\n\n"
    )
    ask = {"bullets": "Produce 5-7 bullet points.",
           "one_paragraph": "Produce one paragraph of 5-7 sentences.",
           "three_sentence": "Produce exactly three sentences."}[style]
    return base + ask

def summarize_one(title, body, style, src_words):
    if style == "one_paragraph":
        lo, hi = target_word_bounds(src_words)
        max_new = approx_tokens_from_words(hi)
    elif style == "bullets":
        max_new = 220
    else:
        max_new = 120
    p = prompt(style, title, clamp_chars(body))
    inputs = tok(p, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new, temperature=0.2, do_sample=False,
                           repetition_penalty=1.05, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(p):].strip()

def run(in_csv="data/guardian_articles_raw.csv", out_jsonl="outputs/mistral.jsonl", limit=0):
    ensure_dir(out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, title, body, domain, url in tqdm(iter_articles(in_csv, limit=limit)):
            src_words = word_count(body)
            for style in STYLES:
                s = summarize_one(title, body, style, src_words)
                write_record(out, {
                    "doc_id": i, "domain": domain, "url": url,
                    "model": "Mistral-7B-Instruct", "style": style, "title": title, "summary": s
                })

if __name__ == "__main__":
    run()
